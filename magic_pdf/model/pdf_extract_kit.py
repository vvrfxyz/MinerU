# flake8: noqa  # 指示 flake8 忽略此文件的代码风格检查
import os  # 导入操作系统接口模块
import time  # 导入时间相关模块

import cv2  # 导入 OpenCV 库，用于图像处理
import torch  # 导入 PyTorch 深度学习框架
import yaml  # 导入 YAML 文件处理库
from loguru import logger  # 导入 loguru 日志库

# 设置环境变量，禁止 albumentations 库检查更新
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# 从项目内部导入常量定义
from magic_pdf.config.constants import *
# 从项目内部导入原子模型枚举
from magic_pdf.model.model_list import AtomicModel
# 从项目内部导入模型初始化单例
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
# 从项目内部导入模型工具函数
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram, crop_img, get_res_list_from_layout_res)
# 从项目内部导入 OCR 工具函数
from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.ocr_utils import (
    get_adjusted_mfdetrec_res, get_ocr_result_list)


# 定义一个自定义的 PDF Extract Kit (PEK) 模型类
class CustomPEKModel:

    # 初始化方法
    def __init__(self, ocr: bool = False, show_log: bool = False, **kwargs):
        """
        ======== 模型初始化 ========
        """
        # 获取当前文件 (pdf_extract_kit.py) 的绝对路径
        current_file_path = os.path.abspath(__file__)
        # 获取当前文件所在的目录 (model)
        current_dir = os.path.dirname(current_file_path)
        # 获取上一级目录 (magic_pdf) 作为项目根目录
        root_dir = os.path.dirname(current_dir)
        # 构建模型配置目录的路径
        model_config_dir = os.path.join(root_dir, 'resources', 'model_config')
        # 构建 model_configs.yaml 文件的完整路径
        config_path = os.path.join(model_config_dir, 'model_configs.yaml')
        # 读取 YAML 配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        # 初始化解析配置

        # --- 版面分析 (Layout) 配置 ---
        # 从 kwargs 获取版面分析配置，如果未提供则为空字典
        self.layout_config = kwargs.get('layout_config', {})
        # 获取版面分析模型名称，默认为 DocLayout_YOLO
        self.layout_model_name = self.layout_config.get(
            'model', MODEL_NAME.DocLayout_YOLO
        )

        # --- 公式 (Formula) 配置 ---
        # 从 kwargs 获取公式配置
        self.formula_config = kwargs.get('formula_config', {})
        # 获取公式检测 (MFD) 模型名称，默认为 YOLO_V8_MFD
        self.mfd_model_name = self.formula_config.get(
            'mfd_model', MODEL_NAME.YOLO_V8_MFD
        )
        # 获取公式识别 (MFR) 模型名称，默认为 UniMerNet_v2_Small
        self.mfr_model_name = self.formula_config.get(
            'mfr_model', MODEL_NAME.UniMerNet_v2_Small
        )
        # 是否启用公式处理，默认为 True
        self.apply_formula = self.formula_config.get('enable', True)

        # --- 表格 (Table) 配置 ---
        # 从 kwargs 获取表格配置
        self.table_config = kwargs.get('table_config', {})
        # 是否启用表格识别，默认为 False
        self.apply_table = self.table_config.get('enable', False)
        # 表格识别的最大处理时间（秒），默认为常量 TABLE_MAX_TIME_VALUE
        self.table_max_time = self.table_config.get('max_time', TABLE_MAX_TIME_VALUE)
        # 获取表格识别模型名称，默认为 RAPID_TABLE
        self.table_model_name = self.table_config.get('model', MODEL_NAME.RAPID_TABLE)
        # 获取表格识别的子模型名称（如果适用），默认为 None
        self.table_sub_model_name = self.table_config.get('sub_model', None)

        # --- OCR 配置 ---
        # 是否启用 OCR（由初始化参数传入）
        self.apply_ocr = ocr
        # 获取 OCR 语言（由初始化参数传入），默认为 None
        self.lang = kwargs.get('lang', None)

        # 记录初始化信息
        logger.info(
            'DocAnalysis init, this may take some times, layout_model: {}, apply_formula: {}, apply_ocr: {}, '
            'apply_table: {}, table_model: {}, lang: {}'.format(
                self.layout_model_name,
                self.apply_formula,
                self.apply_ocr,
                self.apply_table,
                self.table_model_name,
                self.lang,
            )
        )
        # 初始化解析方案
        # 获取运行设备（如 'cpu', 'cuda:0', 'mps'），默认为 'cpu'
        self.device = kwargs.get('device', 'cpu')

        logger.info('using device: {}'.format(self.device))
        # 获取模型文件存放目录，默认为 'resources/models'
        models_dir = kwargs.get(
            'models_dir', os.path.join(root_dir, 'resources', 'models')
        )
        logger.info('using models_dir: {}'.format(models_dir))

        # 获取原子模型管理器（单例模式）
        atom_model_manager = AtomModelSingleton()

        # --- 初始化公式识别模型 (如果启用) ---
        if self.apply_formula:
            # 初始化公式检测 (MFD) 模型
            self.mfd_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFD, # 指定模型类型为 MFD
                mfd_weights=str( # 拼接 MFD 模型权重文件的完整路径
                    os.path.join(
                        models_dir, self.configs['weights'][self.mfd_model_name]
                    )
                ),
                device=self.device, # 指定运行设备
            )

            # 初始化公式解析 (MFR) 模型
            # 获取 MFR 模型权重文件的目录
            mfr_weight_dir = str(
                os.path.join(models_dir, self.configs['weights'][self.mfr_model_name])
            )
            # 获取 MFR 模型配置文件的路径 (UniMERNet)
            mfr_cfg_path = str(os.path.join(model_config_dir, 'UniMERNet', 'demo.yaml'))

            self.mfr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFR, # 指定模型类型为 MFR
                mfr_weight_dir=mfr_weight_dir, # 传入权重目录
                mfr_cfg_path=mfr_cfg_path, # 传入配置文件路径
                device=self.device, # 指定运行设备
            )

        # --- 初始化版面分析 (Layout) 模型 ---
        # 根据配置选择并初始化 Layout 模型
        if self.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            # 初始化 LayoutLMv3 模型
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout, # 指定模型类型为 Layout
                layout_model_name=MODEL_NAME.LAYOUTLMv3, # 指定具体模型名称
                layout_weights=str( # 拼接 LayoutLMv3 权重文件路径
                    os.path.join(
                        models_dir, self.configs['weights'][self.layout_model_name]
                    )
                ),
                layout_config_file=str( # 拼接 LayoutLMv3 配置文件路径
                    os.path.join(
                        model_config_dir, 'layoutlmv3', 'layoutlmv3_base_inference.yaml'
                    )
                ),
                # 如果设备是 Apple Silicon (mps)，则强制在 CPU 上运行 LayoutLMv3
                device='cpu' if str(self.device).startswith("mps") else self.device,
            )
        elif self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # 初始化 DocLayout_YOLO 模型
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout, # 指定模型类型为 Layout
                layout_model_name=MODEL_NAME.DocLayout_YOLO, # 指定具体模型名称
                doclayout_yolo_weights=str( # 拼接 DocLayout_YOLO 权重文件路径
                    os.path.join(
                        models_dir, self.configs['weights'][self.layout_model_name]
                    )
                ),
                device=self.device, # 指定运行设备
            )
        # --- 初始化 OCR 模型 ---
        # 即使不启用全局 OCR (apply_ocr=False)，也可能需要 OCR 模型进行文本检测
        self.ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR, # 指定模型类型为 OCR
            ocr_show_log=show_log, # 是否显示 OCR 日志
            det_db_box_thresh=0.3, # 文本检测阈值
            lang=self.lang # 指定语言
        )
        # --- 初始化表格识别 (Table) 模型 (如果启用) ---
        if self.apply_table:
            # 获取表格模型权重/配置文件的目录或路径
            table_model_dir = self.configs['weights'][self.table_model_name]
            self.table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Table, # 指定模型类型为 Table
                table_model_name=self.table_model_name, # 指定具体表格模型名称
                table_model_path=str(os.path.join(models_dir, table_model_dir)), # 拼接模型路径
                table_max_time=self.table_max_time, # 设置最大处理时间
                device=self.device, # 指定运行设备
                ocr_engine=self.ocr_model, # 将已初始化的 OCR 模型实例传递给表格模型（可能用于识别单元格内的文本）
                table_sub_model_name=self.table_sub_model_name # 传入子模型名称（如果需要）
            )

        logger.info('DocAnalysis init done!') # 记录初始化完成

    # 定义类的调用方法，处理单个图像
    def __call__(self, image):
        # --- 1. 版面分析 ---
        layout_start = time.time() # 记录开始时间
        layout_res = [] # 初始化版面分析结果列表
        # 根据模型类型调用相应的预测方法
        if self.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            # 调用 LayoutLMv3 模型
            layout_res = self.layout_model(image, ignore_catids=[]) # ignore_catids 可能用于忽略某些类别
        elif self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # 调用 DocLayout_YOLO 模型
            layout_res = self.layout_model.predict(image)

        layout_cost = round(time.time() - layout_start, 2) # 计算耗时
        logger.info(f'layout detection time: {layout_cost}') # 记录耗时

        # --- 2. 公式处理 (如果启用) ---
        if self.apply_formula:
            # --- 2a. 公式检测 (MFD) ---
            mfd_start = time.time() # 记录开始时间
            mfd_res = self.mfd_model.predict(image) # 检测图像中的公式区域
            logger.info(f'mfd time: {round(time.time() - mfd_start, 2)}') # 记录耗时

            # --- 2b. 公式识别 (MFR) ---
            mfr_start = time.time() # 记录开始时间
            # 使用 MFR 模型识别检测到的公式 (mfd_res)
            formula_list = self.mfr_model.predict(mfd_res, image)
            # 将识别出的公式结果添加到版面分析结果列表中
            layout_res.extend(formula_list)
            mfr_cost = round(time.time() - mfr_start, 2) # 计算耗时
            logger.info(f'formula nums: {len(formula_list)}, mfr time: {mfr_cost}') # 记录公式数量和耗时

        # --- 清理显存 ---
        # 根据设备类型和显存占用阈值清理显存，防止内存溢出
        clean_vram(self.device, vram_threshold=6)

        # --- 3. 从版面分析结果中提取需要进一步处理的区域 ---
        # 将 layout_res 按照类型（需要 OCR、表格、公式）进行分类
        ocr_res_list, table_res_list, single_page_mfdetrec_res = (
            get_res_list_from_layout_res(layout_res)
        )

        # --- 4. OCR 处理 ---
        ocr_start = time.time() # 记录开始时间
        # 遍历所有需要进行 OCR 处理的区域 (通常是文本块、标题等)
        for res in ocr_res_list:
            # 裁剪出该区域的图像，并在周围添加一些空白（padding），可能有助于提高识别准确率
            new_image, useful_list = crop_img(res, image, crop_paste_x=50, crop_paste_y=50)
            # 调整该区域内已有的公式检测结果的坐标，以适应裁剪后的图像坐标系
            adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(single_page_mfdetrec_res, useful_list)

            # 将裁剪后的图像从 RGB 转换为 BGR (适配 OCR 模型输入)
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

            # 调用 OCR 模型进行识别
            if self.apply_ocr:
                # 如果启用 OCR，同时进行文本检测和文本识别
                # mfd_res=adjusted_mfdetrec_res 用于告知 OCR 模型跳过这些区域（因为它们是公式）
                ocr_res = self.ocr_model.ocr(new_image, mfd_res=adjusted_mfdetrec_res)[0]
            else:
                # 如果未启用 OCR，只进行文本检测 (rec=False)
                ocr_res = self.ocr_model.ocr(new_image, mfd_res=adjusted_mfdetrec_res, rec=False)[0]

            # --- 结果整合 ---
            # 如果 OCR 模型返回了结果 (检测到了文本框)
            if ocr_res:
                # 将 OCR 结果（文本框坐标、可能包含文本内容）转换回原始图像坐标系，并格式化
                ocr_result_list = get_ocr_result_list(ocr_res, useful_list)
                # 将格式化后的 OCR 结果添加到最终的版面分析结果列表中
                layout_res.extend(ocr_result_list)

        ocr_cost = round(time.time() - ocr_start, 2) # 计算 OCR (或仅检测) 的总耗时
        if self.apply_ocr:
            logger.info(f"ocr time: {ocr_cost}") # 记录 OCR 耗时
        else:
            logger.info(f"det time: {ocr_cost}") # 记录文本检测耗时

        # --- 5. 表格识别 (如果启用) ---
        if self.apply_table:
            table_start = time.time() # 记录开始时间
            # 遍历所有检测到的表格区域
            for res in table_res_list:
                # 裁剪出表格区域的图像
                new_image, _ = crop_img(res, image)
                single_table_start_time = time.time() # 记录单个表格处理开始时间
                html_code = None # 初始化 HTML 代码变量
                # 根据配置的表格模型调用相应的识别方法
                if self.table_model_name == MODEL_NAME.STRUCT_EQTABLE:
                    # 使用 StructER / EqTable 模型
                    with torch.no_grad(): # 在不计算梯度的情况下运行，节省内存
                        table_result = self.table_model.predict(new_image, 'html') # 预测并指定输出格式为 html
                        if len(table_result) > 0:
                            html_code = table_result[0] # 获取 HTML 结果
                elif self.table_model_name == MODEL_NAME.TABLE_MASTER:
                    # 使用 TableMaster 模型
                    html_code = self.table_model.img2html(new_image) # 直接将图像转换为 HTML
                elif self.table_model_name == MODEL_NAME.RAPID_TABLE:
                    # 使用 RapidTable 模型
                    # 预测并获取 HTML 代码、单元格边界框、逻辑坐标点和耗时
                    html_code, table_cell_bboxes, logic_points, elapse = self.table_model.predict(
                        new_image
                    )
                # 检查单个表格处理是否超时
                run_time = time.time() - single_table_start_time
                if run_time > self.table_max_time:
                    logger.warning(
                        f'table recognition processing exceeds max time {self.table_max_time}s'
                    )
                # --- 表格结果处理 ---
                # 判断是否成功获取了 HTML 代码
                if html_code:
                    # 检查 HTML 代码是否以预期的标签结束，简单判断是否为有效表格 HTML
                    expected_ending = html_code.strip().endswith(
                        '</html>'
                    ) or html_code.strip().endswith('</table>')
                    if expected_ending:
                        # 如果有效，将 HTML 代码存入该表格区域的结果字典中
                        res['html'] = html_code
                    else:
                        # 如果无效，记录警告
                        logger.warning(
                            'table recognition processing fails, not found expected HTML table end'
                        )
                else:
                    # 如果未能获取 HTML 代码，记录警告
                    logger.warning(
                        'table recognition processing fails, not get html return'
                    )
            logger.info(f'table time: {round(time.time() - table_start, 2)}') # 记录表格处理总耗时

        # 返回包含所有检测和识别结果的列表
        return layout_res [2]