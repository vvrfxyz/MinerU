import time # 导入时间模块
import cv2 # 导入 OpenCV 库
from loguru import logger # 导入 loguru 日志库
from tqdm import tqdm # 导入 tqdm 进度条库

# 从项目内部导入常量和模型名称
from magic_pdf.config.constants import MODEL_NAME
# 从项目内部导入模型初始化单例
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
# 从项目内部导入模型工具函数
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram, crop_img, get_res_list_from_layout_res)
# 从项目内部导入 OCR 工具函数
from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.ocr_utils import (
    get_adjusted_mfdetrec_res, get_ocr_result_list)

# 定义基础批处理大小常量 (可能用于与 batch_ratio 相乘)
YOLO_LAYOUT_BASE_BATCH_SIZE = 1 # YOLO 版面分析模型的基础批处理大小
MFD_BASE_BATCH_SIZE = 1 # 公式检测模型的基础批处理大小
MFR_BASE_BATCH_SIZE = 16 # 公式识别模型的基础批处理大小


# 定义执行批量分析的类
class BatchAnalyze:
    # 初始化方法
    def __init__(self, model_manager, batch_ratio: int, show_log, layout_model, formula_enable, table_enable):
        self.model_manager = model_manager # 模型单例管理器
        self.batch_ratio = batch_ratio # 批处理比例因子 (根据显存计算得到)
        self.show_log = show_log # 是否显示日志
        self.layout_model = layout_model # 指定的版面分析模型名称
        self.formula_enable = formula_enable # 是否启用公式处理
        self.table_enable = table_enable # 是否启用表格识别

    # 定义类的调用方法，处理一批图像信息
    def __call__(self, images_with_extra_info: list) -> list:
        # 如果输入列表为空，直接返回空列表
        if len(images_with_extra_info) == 0:
            return []

        images_layout_res = [] # 用于存储所有图像的版面分析初步结果
        layout_start_time = time.time() # 记录版面分析开始时间
        # 从第一张图像信息中获取 OCR 标志和语言，用于获取/初始化对应的模型实例
        _, fst_ocr, fst_lang = images_with_extra_info[0]
        # 使用 ModelSingleton 获取或初始化模型实例 (会根据参数缓存)
        self.model = self.model_manager.get_model(fst_ocr, self.show_log, fst_lang, self.layout_model, self.formula_enable, self.table_enable)

        # 提取所有图像的 NumPy 数组列表
        images = [image for image, _, _ in images_with_extra_info]

        # --- 1. 批量版面分析 ---
        if self.model.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            # 如果是 LayoutLMv3，目前似乎还是逐张处理（需要确认 LayoutLMv3 模型是否支持批量推理）
            for image in images:
                layout_res = self.model.layout_model(image, ignore_catids=[])
                images_layout_res.append(layout_res)
        elif self.model.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # 如果是 DocLayout_YOLO，调用其批量预测方法
            layout_images = [] # 准备输入图像列表
            for image_index, image in enumerate(images):
                layout_images.append(image)

            # 调用模型的 batch_predict 方法
            # 注意：实际使用的 batch size 是基础大小，batch_ratio 似乎未在此处使用
            images_layout_res += self.model.layout_model.batch_predict(
                layout_images, YOLO_LAYOUT_BASE_BATCH_SIZE
            )

        # 记录版面分析耗时 (注释掉了)
        # logger.info(
        #     f'layout time: {round(time.time() - layout_start_time, 2)}, image num: {len(images)}'
        # )

        # --- 2. 批量公式处理 (如果启用) ---
        if self.model.apply_formula:
            # --- 2a. 批量公式检测 (MFD) ---
            mfd_start_time = time.time() # 记录开始时间
            # 调用 MFD 模型的批量预测方法
            # 注意：实际使用的 batch size 是基础大小，batch_ratio 似乎未在此处使用
            images_mfd_res = self.model.mfd_model.batch_predict(
                images, MFD_BASE_BATCH_SIZE
            )
            # 记录 MFD 耗时 (注释掉了)
            # logger.info(
            #     f'mfd time: {round(time.time() - mfd_start_time, 2)}, image num: {len(images)}'
            # )

            # --- 2b. 批量公式识别 (MFR) ---
            mfr_start_time = time.time() # 记录开始时间
            # 调用 MFR 模型的批量预测方法
            images_formula_list = self.model.mfr_model.batch_predict(
                images_mfd_res, # 输入是 MFD 检测结果列表
                images, # 输入原始图像列表
                batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE, # 使用计算得到的批处理大小
            )
            mfr_count = 0 # 计数器，统计总共识别了多少公式
            # 将识别出的公式结果合并到对应图像的版面分析结果中
            for image_index in range(len(images)):
                images_layout_res[image_index] += images_formula_list[image_index]
                mfr_count += len(images_formula_list[image_index])
            # 记录 MFR 耗时和数量 (注释掉了)
            # logger.info(
            #     f'mfr time: {round(time.time() - mfr_start_time, 2)}, image num: {mfr_count}'
            # )

        # --- 清理显存 (注释掉了) ---
        # clean_vram(self.model.device, vram_threshold=8)

        # --- 准备后续处理（OCR检测、表格识别、OCR识别）的数据结构 ---
        ocr_res_list_all_page = [] # 存储所有页面需要进行 OCR 检测的区域信息
        table_res_list_all_page = [] # 存储所有页面需要进行表格识别的区域信息

        # 遍历每张图像的初步版面分析结果
        for index in range(len(images)):
            _, ocr_enable, _lang = images_with_extra_info[index] # 获取该页是否需要OCR和语言信息
            layout_res = images_layout_res[index] # 获取该页的版面结果
            np_array_img = images[index] # 获取该页的图像 NumPy 数组

            # 从版面结果中分离出需要 OCR 的区域、表格区域和公式区域
            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )

            # 将需要 OCR 的区域信息打包存储
            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list, # 区域列表
                                          'lang':_lang, # 语言
                                          'ocr_enable':ocr_enable, # 是否需要最终识别文本
                                          'np_array_img':np_array_img, # 原始图像
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res, # 该页公式结果（用于避让）
                                          'layout_res':layout_res, # 该页的完整版面结果（后续会将 OCR 结果添加进去）
                                          })

            # 将需要表格识别的区域信息打包存储
            for table_res in table_res_list:
                table_img, _ = crop_img(table_res, np_array_img) # 裁剪出表格图像
                table_res_list_all_page.append({'table_res':table_res, # 表格区域信息
                                                'lang':_lang, # 语言 (可能影响表格内文字识别)
                                                'table_img':table_img, # 裁剪出的表格图像
                                              })

        # --- 3. 批量文本框检测 (OCR-det) ---
        # 注意：这里的实现是逐个区域进行检测，并非真正的模型层面批量处理，但逻辑上是处理一个批次的页面数据
        det_start = time.time() # 记录开始时间
        det_count = 0 # 计数器
        # 使用 tqdm 显示处理进度
        for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
            _lang = ocr_res_list_dict['lang'] # 获取当前处理页面的语言
            # 获取对应语言的 OCR 原子模型实例 (如果不同页面语言不同，会动态获取)
            atom_model_manager = AtomModelSingleton()
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name='ocr',
                ocr_show_log=False, # 不显示日志
                det_db_box_thresh=0.3, # 检测阈值
                lang=_lang # 指定语言
            )
            # 遍历当前页面中需要 OCR 的每个区域
            for res in ocr_res_list_dict['ocr_res_list']:
                # 裁剪图像区域并添加 padding
                new_image, useful_list = crop_img(
                    res, ocr_res_list_dict['np_array_img'], crop_paste_x=50, crop_paste_y=50
                )
                # 调整公式坐标以适应裁剪后的图像
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                )

                # 转换为 BGR 格式
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                # 调用 OCR 模型进行文本检测 (rec=False)
                ocr_res = ocr_model.ocr(
                    new_image, mfd_res=adjusted_mfdetrec_res, rec=False
                )[0]

                # --- 结果整合 ---
                if ocr_res: # 如果检测到文本框
                    # 将检测结果坐标转换回原始页面坐标系，并进行格式化
                    # 同时，如果该页面需要 OCR 识别 (ocr_enable=True)，会将裁剪后的小图像 (new_image) 和语言信息暂存起来，用于后续的批量识别
                    ocr_result_list = get_ocr_result_list(ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], new_image, _lang)
                    # 将格式化后的检测结果 (包含 category_id=15 的文本行) 追加到对应页面的版面结果中
                    ocr_res_list_dict['layout_res'].extend(ocr_result_list)
            det_count += len(ocr_res_list_dict['ocr_res_list']) # 累加处理的区域数量
        # 记录文本检测耗时 (注释掉了)
        # logger.info(f'ocr-det time: {round(time.time()-det_start, 2)}, image num: {det_count}')


        # --- 4. 批量表格识别 (如果启用) ---
        # 注意：这里的实现也是逐个表格区域进行识别
        if self.model.apply_table:
            table_start = time.time() # 记录开始时间
            table_count = 0 # 计数器
            # 使用 tqdm 显示处理进度
            for table_res_dict in tqdm(table_res_list_all_page, desc="Table Predict"):
                _lang = table_res_dict['lang'] # 获取语言
                # 获取表格识别所需的 OCR 引擎（可能与主流程 OCR 参数不同）
                atom_model_manager = AtomModelSingleton()
                ocr_engine = atom_model_manager.get_atom_model(
                    atom_model_name='ocr',
                    ocr_show_log=False,
                    det_db_box_thresh=0.5, # 使用不同的检测阈值
                    det_db_unclip_ratio=1.6, # 使用不同的 unclip ratio
                    lang=_lang
                )
                # 获取表格识别模型实例 (这里写死了 rapid_table 和 slanet_plus)
                # TODO: 这里模型路径为空，设备为 cpu，可能需要根据实际配置调整
                table_model = atom_model_manager.get_atom_model(
                    atom_model_name='table',
                    table_model_name='rapid_table',
                    table_model_path='', # 模型路径未指定？
                    table_max_time=400, # 最大时间
                    device='cpu', # 在 CPU 上运行表格识别？
                    ocr_engine=ocr_engine, # 传入专用的 OCR 引擎
                    table_sub_model_name='slanet_plus' # 指定子模型
                )
                # 对裁剪出的表格图像进行预测
                html_code, table_cell_bboxes, logic_points, elapse = table_model.predict(table_res_dict['table_img'])
                # --- 表格结果处理 ---
                if html_code: # 如果返回了 HTML 代码
                    # 简单检查 HTML 结尾是否有效
                    expected_ending = html_code.strip().endswith(
                        '</html>'
                    ) or html_code.strip().endswith('</table>')
                    if expected_ending:
                        # 将有效的 HTML 代码添加到对应表格区域的结果字典中
                        table_res_dict['table_res']['html'] = html_code
                    else:
                        logger.warning(
                            'table recognition processing fails, not found expected HTML table end'
                        )
                else:
                    logger.warning(
                        'table recognition processing fails, not get html return'
                    )
            # 记录表格识别耗时 (注释掉了)
            # logger.info(f'table time: {round(time.time() - table_start, 2)}, image num: {len(table_res_list_all_page)}')

        # --- 准备批量 OCR 识别数据 ---
        # 创建字典，按语言对需要识别的文本行进行分组
        need_ocr_lists_by_lang = {}  # key: lang, value: list of layout_res_item (category_id=15)
        img_crop_lists_by_lang = {}  # key: lang, value: list of cropped image (np_img)

        # 遍历所有页面的所有版面元素
        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                # 筛选出之前标记为需要 OCR 的文本行 (category_id=15)
                if layout_res_item['category_id'] in [15]:
                    # 检查是否暂存了裁剪图像 ('np_img') 和语言 ('lang')
                    if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                        lang = layout_res_item['lang'] # 获取语言

                        # 如果该语言尚未在字典中，则初始化空列表
                        if lang not in need_ocr_lists_by_lang:
                            need_ocr_lists_by_lang[lang] = []
                            img_crop_lists_by_lang[lang] = []

                        # 将文本行信息和对应的裁剪图像添加到相应语言的列表中
                        need_ocr_lists_by_lang[lang].append(layout_res_item)
                        img_crop_lists_by_lang[lang].append(layout_res_item['np_img'])

                        # 从原始字典中移除暂存的图像和语言字段，避免冗余
                        layout_res_item.pop('np_img')
                        layout_res_item.pop('lang')


        # --- 5. 批量文本识别 (OCR-rec) ---
        if len(img_crop_lists_by_lang) > 0: # 如果有需要识别的文本行

            rec_time = 0 # 初始化识别总耗时
            rec_start = time.time() # 记录开始时间
            total_processed = 0 # 计数器

            # 按语言分组进行批量识别
            for lang, img_crop_list in img_crop_lists_by_lang.items():
                if len(img_crop_list) > 0: # 如果该语言有需要识别的图像
                    # 获取对应语言的 OCR 模型实例
                    atom_model_manager = AtomModelSingleton()
                    ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name='ocr',
                        ocr_show_log=False,
                        det_db_box_thresh=0.3, # 检测阈值 (虽然这里只做识别，但可能初始化需要)
                        lang=lang
                    )
                    # 调用 OCR 模型的 ocr 方法进行批量识别 (det=False)
                    # tqdm_enable=True 可以在 OCR 内部显示识别进度条
                    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]

                    # 检查返回结果数量是否与输入数量一致
                    assert len(ocr_res_list) == len(
                        need_ocr_lists_by_lang[lang]), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}'

                    # 将识别结果 (文本和置信度) 更新回对应的文本行字典中
                    for index, layout_res_item in enumerate(need_ocr_lists_by_lang[lang]):
                        ocr_text, ocr_score = ocr_res_list[index] # 获取识别结果
                        layout_res_item['text'] = ocr_text # 更新文本内容
                        layout_res_item['score'] = float(round(ocr_score, 2)) # 更新置信度得分

                    total_processed += len(img_crop_list) # 累加处理的图像数量

            rec_time += time.time() - rec_start # 计算总识别耗时
            # 记录 OCR 识别耗时和处理数量 (注释掉了)
            # logger.info(f'ocr-rec time: {round(rec_time, 2)}, total images processed: {total_processed}')


        # 返回包含了所有处理结果（版面、公式、表格HTML、OCR文本行）的列表，每个元素对应输入的一张图像
        return images_layout_res [4]