# -*- coding: utf-8 -*-
# 导入时间模块，用于计时
import time
# 导入OpenCV库，用于图像处理
import cv2
import os
# 导入loguru库，用于日志记录
from loguru import logger
# 导入tqdm库，用于显示进度条
from tqdm import tqdm
# 从常量配置文件中导入模型名称常量
from magic_pdf.config.constants import MODEL_NAME
# 导入模型初始化相关的单例类
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
# 导入模型相关的工具函数
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram, crop_img, get_res_list_from_layout_res)
# 导入OCR相关的工具函数
from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.ocr_utils import (
    get_adjusted_mfdetrec_res, get_ocr_result_list)

# 定义YOLO布局模型的基础批处理大小
YOLO_LAYOUT_BASE_BATCH_SIZE = 1
# 定义数学公式检测模型的基础批处理大小
MFD_BASE_BATCH_SIZE = 1
# 定义数学公式识别模型的基础批处理大小
MFR_BASE_BATCH_SIZE = 16

# 定义批量分析类
class BatchAnalyze:
    # 初始化方法
    def __init__(self, model_manager, batch_ratio: int, show_log, layout_model, formula_enable, table_enable):
        # 模型管理器实例
        self.model_manager = model_manager
        # 批处理比例因子
        self.batch_ratio = batch_ratio
        # 是否显示日志标志
        self.show_log = show_log
        # 使用的布局模型名称
        self.layout_model = layout_model
        # 是否启用公式处理标志
        self.formula_enable = formula_enable
        # 是否启用表格处理标志
        self.table_enable = table_enable

    # 可调用方法，用于执行批量分析
    def __call__(self, images_with_extra_info: list) -> list:
        # 如果输入列表为空，则返回空列表
        if len(images_with_extra_info) == 0:
            return []
        
        # --- 开始：图片切片功能新增 ---
        output_dir = "/data/MinerU/output/cut/"
        try:
            os.makedirs(output_dir, exist_ok=True)
            # print(f"创建或确认输出目录: {output_dir}") # 用于调试
        except OSError as e:
            print(f"错误：无法创建目录 {output_dir}: {e}")
            # 根据需要决定是抛出异常还是继续但不保存切片
            output_dir = None # 设置为None，后续跳过保存步骤
        # --- 结束：图片切片功能新增 ---

        # 初始化存储布局结果的列表
        images_layout_res = []
        # 记录布局分析开始时间
        layout_start_time = time.time()
        
        # 获取第一张图像的OCR启用状态和语言信息
        _, fst_ocr, fst_lang = images_with_extra_info[0]
        # 根据输入信息获取相应的模型实例
        self.model = self.model_manager.get_model(fst_ocr, self.show_log, fst_lang, self.layout_model, self.formula_enable, self.table_enable)
        
        # 提取所有图像数据
        images = [image for image, _,_ in images_with_extra_info]
        
        # 判断使用的布局模型类型
        if self.model.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            # 如果是LayoutLMv3模型
            # 遍历每张图像进行布局分析
            for image in images:
                # 调用LayoutLMv3模型进行预测，忽略指定的类别ID
                layout_res = self.model.layout_model(image, ignore_catids=[])
                # 将结果添加到列表中
                images_layout_res.append(layout_res)
        elif self.model.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # 如果是DocLayout_YOLO模型
            layout_images = []
            # 准备用于布局分析的图像列表
            for image_index, image in enumerate(images):
                layout_images.append(image)
            # 调用YOLO模型进行批量预测
            images_layout_res += self.model.layout_model.batch_predict(
                # layout_images, self.batch_ratio * YOLO_LAYOUT_BASE_BATCH_SIZE # 原批处理大小计算方式
                layout_images, YOLO_LAYOUT_BASE_BATCH_SIZE # 使用固定的基础批处理大小
            )
            
        # 记录布局分析耗时和处理的图像数量 (注释掉的代码)
        # logger.info(
        #     f'layout time: {round(time.time() - layout_start_time, 2)}, image num: {len(images)}'
        # )
        
        # 如果启用了公式处理
        if self.model.apply_formula:
            # --- 公式检测 ---
            # 记录公式检测开始时间
            mfd_start_time = time.time()
            # 调用公式检测模型进行批量预测
            images_mfd_res = self.model.mfd_model.batch_predict(
                # images, self.batch_ratio * MFD_BASE_BATCH_SIZE # 原批处理大小计算方式
                images, MFD_BASE_BATCH_SIZE # 使用固定的基础批处理大小
            )
            # 记录公式检测耗时和处理的图像数量 (注释掉的代码)
            # logger.info(
            #     f'mfd time: {round(time.time() - mfd_start_time, 2)}, image num: {len(images)}'
            # )

            # --- 公式识别 ---
            # 记录公式识别开始时间
            mfr_start_time = time.time()
            # 调用公式识别模型进行批量预测
            images_formula_list = self.model.mfr_model.batch_predict(
                images_mfd_res, # 公式检测结果
                images, # 原始图像
                batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE, # 计算实际批处理大小
            )
            # 初始化公式识别计数器
            mfr_count = 0
            # 遍历每张图像的公式识别结果
            for image_index in range(len(images)):
                # 将公式识别结果合并到对应的布局结果中
                images_layout_res[image_index] += images_formula_list[image_index]
                # 累加识别出的公式数量
                mfr_count += len(images_formula_list[image_index])
            # 记录公式识别耗时和识别出的公式数量 (注释掉的代码)
            # logger.info(
            #     f'mfr time: {round(time.time() - mfr_start_time, 2)}, image num: {mfr_count}'
            # )
            
        # 清理显存 (注释掉的代码)
        # clean_vram(self.model.device, vram_threshold=8)
        
        # 初始化用于存储所有页面OCR结果的列表
        ocr_res_list_all_page = []
        # 初始化用于存储所有页面表格结果的列表
        table_res_list_all_page = []
        
        # 遍历所有图像及其布局结果
        for index in range(len(images)):
            # 获取当前图像的OCR启用状态和语言
            _, ocr_enable, _lang = images_with_extra_info[index]
            # 获取当前图像的布局结果
            layout_res = images_layout_res[index]
            # 获取当前图像的NumPy数组表示
            np_array_img = images[index]
            # 从布局结果中提取OCR区域、表格区域和单页公式检测/识别结果
            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )
            # 将OCR相关信息存入列表
            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list, # OCR区域列表
                                          'lang':_lang, # 语言
                                          'ocr_enable':ocr_enable, # OCR是否启用
                                          'np_array_img':np_array_img, # 图像NumPy数组
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res, # 单页公式结果
                                          'layout_res':layout_res, # 完整的布局结果 (用于后续更新)
                                          })
            # 遍历当前页面的表格区域
            for table_res in table_res_list:
                # 裁剪出表格图像
                table_img, _= crop_img(table_res, np_array_img)
                # 将表格相关信息存入列表
                table_res_list_all_page.append({'table_res':table_res, # 表格区域信息
                                                'lang':_lang, # 语言
                                                'table_img':table_img, # 裁剪出的表格图像
                                              })
                                              
        # --- 文本框检测 (OCR Detection) ---
        # 记录检测开始时间
        det_start = time.time()
        # 初始化检测计数器
        det_count = 0
        # 遍历需要进行OCR处理的页面信息 (使用tqdm显示进度条)
        # for ocr_res_list_dict in ocr_res_list_all_page:
        for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
            # 处理每个需要进行OCR的区域
            # 获取当前页面的语言
            _lang = ocr_res_list_dict['lang']
            # 获取原子模型管理器单例
            atom_model_manager = AtomModelSingleton()
            # 获取对应语言的OCR模型 (主要用于检测)
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name='ocr', # 模型类型为OCR
                ocr_show_log=False, # 不显示OCR内部日志
                det_db_box_thresh=0.3, # DBNet检测阈值
                lang=_lang # 指定语言
            )
            # 遍历当前页面的OCR区域列表
            for res in ocr_res_list_dict['ocr_res_list']:
                # 裁剪出包含上下文的图像区域 (crop_paste_x/y 添加了padding)
                new_image, useful_list = crop_img(
                    res, ocr_res_list_dict['np_array_img'], crop_paste_x=50, crop_paste_y=50
                )
                # 调整公式检测/识别结果的坐标以适应裁剪后的图像
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                )
                # --- 执行OCR文本检测 ---
                # 将图像颜色空间从RGB转换为BGR (OpenCV默认使用BGR)
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                # 调用OCR模型进行检测 (rec=False表示只进行检测，不进行识别)
                # adjusted_mfdetrec_res 传递了调整后的公式区域信息，可能用于辅助文本检测
                ocr_res = ocr_model.ocr(
                    new_image, mfd_res=adjusted_mfdetrec_res, rec=False
                )[0] # [0] 获取第一页（也是唯一一页）的结果
                
                # --- 整合结果 ---
                # 如果检测到了文本框
                if ocr_res:
                    # 将检测到的文本框坐标转换回原始图像坐标系，并格式化为结果列表
                    ocr_result_list = get_ocr_result_list(ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], new_image, _lang)
                    # 将新检测到的文本框结果添加到当前页面的布局结果中
                    ocr_res_list_dict['layout_res'].extend(ocr_result_list)
            # 累加处理的OCR区域数量
            det_count += len(ocr_res_list_dict['ocr_res_list'])
        # 记录OCR检测耗时和处理的区域数量 (注释掉的代码)
        # logger.info(f'ocr-det time: {round(time.time()-det_start, 2)}, image num: {det_count}')

        # --- 表格识别 (Table Recognition) ---
        # 如果启用了表格处理
        if self.model.apply_table:
            # 记录表格识别开始时间
            table_start = time.time()
            # 初始化表格计数器
            table_count = 0
            # 遍历需要进行表格识别的区域信息 (使用tqdm显示进度条)
            # for table_res_list_dict in table_res_list_all_page:
            for table_res_dict in tqdm(table_res_list_all_page, desc="Table Predict"):
                # 获取当前表格区域对应的语言
                _lang = table_res_dict['lang']
                # 获取原子模型管理器单例
                atom_model_manager = AtomModelSingleton()
                # 获取用于表格内文本识别的OCR引擎
                ocr_engine = atom_model_manager.get_atom_model(
                    atom_model_name='ocr', # 模型类型为OCR
                    ocr_show_log=False, # 不显示OCR内部日志
                    det_db_box_thresh=0.5, # 检测阈值
                    det_db_unclip_ratio=1.6, # 文本框扩展比例
                    lang=_lang # 指定语言
                )
                # 获取表格识别模型
                table_model = atom_model_manager.get_atom_model(
                    atom_model_name='table', # 模型类型为表格
                    table_model_name='rapid_table', # 表格模型名称
                    table_model_path='', # 表格模型路径 (如果需要指定)
                    table_max_time=400, # 最大处理时间
                    device='cpu', # 指定设备为CPU (可能需要根据实际情况调整)
                    ocr_engine=ocr_engine, # 传入配置好的OCR引擎
                    table_sub_model_name='slanet_plus' # 表格识别子模型名称
                )
                # 调用表格模型进行预测，获取HTML代码、单元格边界框、逻辑坐标点和耗时
                html_code, table_cell_bboxes, logic_points, elapse = table_model.predict(table_res_dict['table_img'])
                
                # --- 判断表格识别结果是否有效 ---
                # 如果返回了HTML代码
                if html_code:
                    # 检查HTML代码是否以预期的标签结束
                    expected_ending = html_code.strip().endswith(
                        '</html>'
                    ) or html_code.strip().endswith('</table>')
                    # 如果结束标签符合预期
                    if expected_ending:
                        # 将HTML代码存入表格结果字典中
                        table_res_dict['table_res']['html'] = html_code
                    else:
                        # 如果结束标签不符合预期，记录警告日志
                        logger.warning(
                            'table recognition processing fails, not found expected HTML table end'
                        )
                else:
                    # 如果没有返回HTML代码，记录警告日志
                    logger.warning(
                        'table recognition processing fails, not get html return'
                    )
            # 记录表格识别耗时和处理的表格数量 (注释掉的代码)
            # logger.info(f'table time: {round(time.time() - table_start, 2)}, image num: {len(table_res_list_all_page)}')

        # --- OCR 文本识别 (Text Recognition) ---
        # 创建字典用于按语言存储需要进行OCR识别的项
        need_ocr_lists_by_lang = {}  # 按语言存储布局项的字典
        img_crop_lists_by_lang = {}  # 按语言存储裁剪图像的字典
        
        # 遍历所有页面的布局结果
        for layout_res in images_layout_res:
            # 遍历当前页面的每个布局项
            for layout_res_item in layout_res:
                # 如果布局项的类别ID是15 (通常表示需要OCR识别的文本行)
                if layout_res_item['category_id'] in [15]:
                    # 并且该项包含裁剪后的图像 ('np_img') 和语言信息 ('lang')
                    if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                        # 获取语言
                        lang = layout_res_item['lang']
                        # 如果该语言尚未在字典中，则初始化列表
                        if lang not in need_ocr_lists_by_lang:
                            need_ocr_lists_by_lang[lang] = []
                            img_crop_lists_by_lang[lang] = []
                        # 将布局项和对应的裁剪图像添加到相应语言的列表中
                        need_ocr_lists_by_lang[lang].append(layout_res_item)
                        img_crop_lists_by_lang[lang].append(layout_res_item['np_img'])
                        # 从原始布局项中移除图像和语言字段，以减少内存占用和避免冗余
                        layout_res_item.pop('np_img')
                        layout_res_item.pop('lang')
                        
        # 如果存在需要进行OCR识别的图像
        if len(img_crop_lists_by_lang) > 0:
            # --- 按语言批量进行OCR识别 ---
            # 初始化识别总耗时
            rec_time = 0
            # 记录识别开始时间
            rec_start = time.time()
            # 初始化总处理计数器
            total_processed = 0
            # 分别处理每种语言
            for lang, img_crop_list in img_crop_lists_by_lang.items():
                # 如果当前语言有需要识别的图像
                if len(img_crop_list) > 0:
                    # 获取对应语言的OCR模型 (主要用于识别)
                    atom_model_manager = AtomModelSingleton()
                    ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name='ocr', # 模型类型为OCR
                        ocr_show_log=False, # 不显示OCR内部日志
                        det_db_box_thresh=0.3, # 检测阈值 (虽然这里主要做识别，但模型初始化可能需要)
                        lang=lang # 指定语言
                    )
                    # 对该语言的图像列表进行批量OCR识别 (det=False表示只进行识别)
                    # tqdm_enable=True 会在OCR内部显示识别进度条
                    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0] # [0] 获取结果
                    
                    # --- 验证结果数量 ---
                    # 断言确保OCR返回的结果数量与输入图像数量一致
                    assert len(ocr_res_list) == len(
                        need_ocr_lists_by_lang[lang]), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}'
                    
                    # --- 处理OCR识别结果 ---
                    # 遍历当前语言的布局项列表
                    for index, layout_res_item in enumerate(need_ocr_lists_by_lang[lang]):
                        # 从OCR结果列表中获取对应的文本和置信度
                        ocr_text, ocr_score = ocr_res_list[index]
                        # 将识别出的文本更新到布局项中
                        layout_res_item['text'] = ocr_text
                        # 将置信度（保留两位小数）更新到布局项中
                        layout_res_item['score'] = float(round(ocr_score, 2))
                    # 累加已处理的图像数量
                    total_processed += len(img_crop_list)
            # 累加识别耗时
            rec_time += time.time() - rec_start
            # 记录OCR识别耗时和总处理数量 (注释掉的代码)
            # logger.info(f'ocr-rec time: {round(rec_time, 2)}, total images processed: {total_processed}')
            
        # --- 开始：图片切片保存 ---
        if output_dir: # 仅在目录创建成功时执行
            print(f"开始保存切片到 {output_dir}...") # 用于调试
            slice_count = 0
            for image_idx, layout_res in enumerate(images_layout_res):
                original_image = images[image_idx] # 获取对应的原始图像 (NumPy Array)
                # 检查原始图像是否有效
                if original_image is None or original_image.size == 0:
                    print(f"警告：跳过图像 {image_idx} 的切片，因为原始图像无效。")
                    continue

                image_height, image_width = original_image.shape[:2]

                for item_idx, item in enumerate(layout_res):
                    # 检查 'poly' 键是否存在且不为 None
                    if 'poly' in item and item['poly'] is not None:
                        try:
                            poly = item['poly']
                            # 确保 poly 是一个列表并且包含足够的元素来提取坐标
                            # (至少需要索引 0, 1, 4, 5，所以长度至少为6)
                            if isinstance(poly, list) and len(poly) >= 6:
                                # 从 'poly' 列表中提取边界框坐标
                                # poly format: [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                                x1 = int(poly[0])  # xmin
                                y1 = int(poly[1])  # ymin
                                x2 = int(poly[4])  # xmax (from bottom-right or top-right corner)
                                y2 = int(poly[5])  # ymax (from bottom-right corner)
                            else:
                                print(f"警告：跳过图像 {image_idx} 项目 {item_idx} - 'poly' 格式无效或长度不足: {poly}")
                                continue # 跳过这个无效的项目

                            # --- 后续代码基本保持不变 ---

                            # 边界检查和修正 (防止坐标超出图像范围)
                            # !! 确保 image_width 和 image_height 在这里是有效的 !!
                            # !! 如果它们是在循环外定义的，确保它们对应当前的 original_image !!
                            # image_height, image_width = original_image.shape[:2] # <--- 可能需要在这里或循环开始处获取当前图像尺寸

                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(image_width, x2)
                            y2 = min(image_height, y2)

                            # 检查修正后的框是否有效 (宽度和高度大于0)
                            if x1 >= x2 or y1 >= y2:
                                # print(f"警告：跳过图像 {image_idx} 项目 {item_idx} 的无效边界框 (修正后): [{x1},{y1},{x2},{y2}], 原始poly: {poly}")
                                continue # 跳过无效或零尺寸的框

                            # 裁剪图像
                            # NumPy 索引是 [y1:y2, x1:x2]
                            slice_img = original_image[y1:y2, x1:x2]

                            # 检查裁剪结果是否为空
                            if slice_img is None or slice_img.size == 0:
                                # print(f"警告：图像 {image_idx} 项目 {item_idx} 裁剪结果为空, 原始poly: {poly}, 修正后: [{x1},{y1},{x2},{y2}]")
                                continue

                            # 构建文件名
                            category_id = item.get('category_id', 'unknown')
                            filename = f"image_{image_idx}_item_{item_idx}_catid_{category_id}.png"
                            output_path = os.path.join(output_dir, filename)

                            # 保存图像
                            # 假设 original_image 是 RGB 格式，转换为 BGR 进行保存
                            # 如果 original_image 已经是 BGR，则移除 cvtColor
                            save_success = cv2.imwrite(output_path, cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR))
                            # save_success = cv2.imwrite(output_path, slice_img) # 如果原始图像是BGR

                            if not save_success:
                                print(f"错误：无法将切片保存到 {output_path}")
                            else:
                                slice_count += 1
                                # print(f"成功保存切片: {output_path}") # 用于详细调试

                        except (ValueError, TypeError) as ve:
                            print(f"错误：处理图像 {image_idx} 项目 {item_idx} 时坐标转换失败: {ve}")
                            print(f"  项目数据: {item}")
                        except Exception as e:
                            print(f"错误：处理图像 {image_idx} 项目 {item_idx} 时发生意外异常: {e}")
                            print(f"  项目数据: {item}")
                    # else: # 可以选择性地记录那些没有 'poly' 键的项目用于调试
                    #     print(f"信息：跳过图像 {image_idx} 项目 {item_idx}，无 'poly' 键。 Item: {item}")


            print(f"完成切片保存，共保存 {slice_count} 个切片。") # 用于调试
        # --- 结束：图片切片保存 ---
        logger.info(f'images_layout_res: {images_layout_res}')
        # 返回包含所有分析结果（布局、公式、表格、OCR文本）的列表
        return images_layout_res