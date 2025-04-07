import os  # 导入操作系统接口模块
import time  # 导入时间相关模块

import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 深度学习框架

# 设置 PaddlePaddle 相关的环境变量，关闭 jit 编译和 stride kernel 优化（可能为了兼容性或特定硬件）
os.environ['FLAGS_npu_jit_compile'] = '0'
os.environ['FLAGS_use_stride_kernel'] = '0'
# 设置 PyTorch 环境变量，允许在 MPS 设备上回退到 CPU 执行不支持的操作
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# 设置环境变量，禁止 albumentations 库检查更新
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


from loguru import logger  # 导入 loguru 日志库

# 从项目内部导入获取显存信息的工具函数
from magic_pdf.model.sub_modules.model_utils import get_vram
# 从项目内部导入支持的 PDF 解析方法枚举
from magic_pdf.config.enums import SupportedPdfParseMethod
# 导入模型配置（可能包含 __model_mode__ 和 __use_inside_model__ 等变量）
import magic_pdf.model as model_config
# 从项目内部导入数据集类
from magic_pdf.data.dataset import Dataset
# 从项目内部导入内存清理函数
from magic_pdf.libs.clean_memory import clean_memory
# 从项目内部导入配置读取函数
from magic_pdf.libs.config_reader import (get_device, get_formula_config,
                                          get_layout_config,
                                          get_local_models_dir,
                                          get_table_recog_config)
# 从项目内部导入模型枚举
from magic_pdf.model.model_list import MODEL

# 定义一个模型单例类，确保全局只有一个模型实例（根据配置缓存）
class ModelSingleton:
    _instance = None # 用于存储单例实例
    _models = {} # 用于缓存已初始化的模型

    # 重写 __new__ 方法以实现单例模式
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # 获取模型实例的方法
    def get_model(
        self,
        ocr: bool, # 是否启用 OCR
        show_log: bool, # 是否显示日志
        lang=None, # OCR 语言
        layout_model=None, # 指定的版面分析模型
        formula_enable=None, # 是否启用公式处理
        table_enable=None, # 是否启用表格识别
    ):
        # 使用配置参数创建一个唯一的键
        key = (ocr, show_log, lang, layout_model, formula_enable, table_enable)
        # 如果该配置的模型尚未初始化
        if key not in self._models:
            # 调用 custom_model_init 函数初始化模型，并将结果存入缓存
            self._models[key] = custom_model_init(
                ocr=ocr,
                show_log=show_log,
                lang=lang,
                layout_model=layout_model,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
        # 返回缓存中对应配置的模型实例
        return self._models[key]


# 自定义模型初始化函数
def custom_model_init(
    ocr: bool = False, # 是否启用 OCR
    show_log: bool = False, # 是否显示日志
    lang=None, # OCR 语言
    layout_model=None, # 指定的版面分析模型
    formula_enable=None, # 是否启用公式处理
    table_enable=None, # 是否启用表格识别
):
    model = None # 初始化模型类型变量
    # 根据全局配置 `__model_mode__` 判断使用哪个模型
    if model_config.__model_mode__ == 'lite':
        # 如果是 'lite' 模式，记录警告并选择 Paddle 模型
        logger.warning(
            'The Lite mode is provided for developers to conduct testing only, and the output quality is '
            'not guaranteed to be reliable.'
        )
        model = MODEL.Paddle # MODEL.Paddle 对应 CustomPaddleModel
    elif model_config.__model_mode__ == 'full':
        # 如果是 'full' 模式，选择 PEK 模型
        model = MODEL.PEK # MODEL.PEK 对应 CustomPEKModel

    # 检查是否允许使用内部模型（可能用于控制商业版本和开源版本的差异）
    if model_config.__use_inside_model__:
        model_init_start = time.time() # 记录模型初始化开始时间
        if model == MODEL.Paddle:
            # 如果选择 Paddle 模型，导入并初始化 CustomPaddleModel
            from magic_pdf.model.pp_structure_v2 import CustomPaddleModel
            custom_model = CustomPaddleModel(ocr=ocr, show_log=show_log, lang=lang)
        elif model == MODEL.PEK:
            # 如果选择 PEK 模型，导入 CustomPEKModel
            from magic_pdf.model.pdf_extract_kit import CustomPEKModel

            # --- 准备 PEK 模型初始化参数 ---
            # 从配置文件读取模型目录和运行设备
            local_models_dir = get_local_models_dir()
            device = get_device()

            # 获取并可能覆盖版面分析配置
            layout_config = get_layout_config()
            if layout_model is not None:
                layout_config['model'] = layout_model

            # 获取并可能覆盖公式配置
            formula_config = get_formula_config()
            if formula_enable is not None:
                formula_config['enable'] = formula_enable

            # 获取并可能覆盖表格识别配置
            table_config = get_table_recog_config()
            if table_enable is not None:
                table_config['enable'] = table_enable

            # 将所有配置整合成一个字典传递给 CustomPEKModel
            model_input = {
                'ocr': ocr,
                'show_log': show_log,
                'models_dir': local_models_dir,
                'device': device,
                'table_config': table_config,
                'layout_config': layout_config,
                'formula_config': formula_config,
                'lang': lang,
            }

            # 初始化 CustomPEKModel
            custom_model = CustomPEKModel(**model_input)
        else:
            # 如果模型名称无效，记录错误并退出
            logger.error('Not allow model_name!')
            exit(1)
        model_init_cost = time.time() - model_init_start # 计算初始化耗时
        logger.info(f'model init cost: {model_init_cost}') # 记录耗时
    else:
        # 如果不允许使用内部模型，记录错误并退出
        logger.error('use_inside_model is False, not allow to use inside model')
        exit(1)

    # 返回初始化好的模型实例
    return custom_model

# 文档分析函数（处理单个 Dataset 对象）
def doc_analyze(
    dataset: Dataset, # 输入的 Dataset 对象，包含文档的所有页面信息和图像
    ocr: bool = False, # 是否启用 OCR
    show_log: bool = False, # 是否显示日志
    start_page_id=0, # 开始处理的页面索引（包含）
    end_page_id=None, # 结束处理的页面索引（包含），None 表示到最后一页
    lang=None, # OCR 语言 ('auto' 或指定语言)
    layout_model=None, # 指定版面分析模型
    formula_enable=None, # 是否启用公式处理
    table_enable=None, # 是否启用表格识别
):
    # 确定结束页面索引
    end_page_id = (
        end_page_id
        if end_page_id is not None and end_page_id >= 0
        else len(dataset) - 1 # 如果未指定或无效，则设为最后一页的索引
    )

    # 从环境变量获取最小批处理推理大小，默认为 200
    MIN_BATCH_INFERENCE_SIZE = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 200))
    images = [] # 存储页面图像
    page_wh_list = [] # 存储页面原始宽高
    # 遍历指定范围内的页面
    for index in range(len(dataset)):
        if start_page_id <= index <= end_page_id:
            page_data = dataset.get_page(index) # 获取页面数据
            img_dict = page_data.get_image() # 获取页面图像及信息
            images.append(img_dict['img']) # 添加图像到列表
            page_wh_list.append((img_dict['width'], img_dict['height'])) # 添加宽高到列表

    # 准备带有额外信息（图像、是否OCR、语言）的图像列表
    images_with_extra_info = []
    if lang is None or lang == 'auto':
        # 如果语言是自动检测或未指定，使用数据集自带的语言信息
        images_with_extra_info = [(images[i], ocr, dataset._lang) for i in range(len(images))]
    else:
        # 否则，使用指定的语言
        images_with_extra_info = [(images[i], ocr, lang) for i in range(len(images))]


    # 如果图像数量达到最小批处理大小，则进行分批
    if len(images) >= MIN_BATCH_INFERENCE_SIZE:
        batch_size = MIN_BATCH_INFERENCE_SIZE
        batch_images = [images_with_extra_info[i:i+batch_size] for i in range(0, len(images_with_extra_info), batch_size)]
    else:
        # 否则，所有图像作为一个批次
        batch_images = [images_with_extra_info]

    results = [] # 存储所有页面的分析结果
    # 遍历每个批次
    for sn, batch_image in enumerate(batch_images):
        # 调用批处理分析函数
        _, result = may_batch_image_analyze(batch_image, sn, ocr, show_log,layout_model, formula_enable, table_enable)
        results.extend(result) # 将批处理结果添加到总结果列表

    model_json = [] # 存储最终格式化的结果
    # 遍历数据集的所有页面索引
    for index in range(len(dataset)):
        if start_page_id <= index <= end_page_id:
            # 如果页面在处理范围内，弹出对应的结果和宽高信息
            result = results.pop(0)
            page_width, page_height = page_wh_list.pop(0)
        else:
            # 如果页面不在处理范围内，设置空结果和默认宽高
            result = []
            page_height = 0
            page_width = 0

        # 构建单页结果字典
        page_info = {'page_no': index, 'width': page_width, 'height': page_height}
        page_dict = {'layout_dets': result, 'page_info': page_info}
        model_json.append(page_dict) # 添加到最终结果列表

    # 导入推理结果类
    from magic_pdf.operators.models import InferenceResult
    # 使用格式化结果和原始数据集创建 InferenceResult 对象并返回
    return InferenceResult(model_json, dataset)

# 批量文档分析函数（处理多个 Dataset 对象）
def batch_doc_analyze(
    datasets: list[Dataset], # 输入的 Dataset 对象列表
    parse_method: str, # 解析方法 ('auto', 'ocr', 'digital')
    show_log: bool = False, # 是否显示日志
    lang=None, # OCR 语言 ('auto' 或指定语言)
    layout_model=None, # 指定版面分析模型
    formula_enable=None, # 是否启用公式处理
    table_enable=None, # 是否启用表格识别
):
    # 从环境变量获取最小批处理推理大小，默认为 200
    MIN_BATCH_INFERENCE_SIZE = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 200))
    batch_size = MIN_BATCH_INFERENCE_SIZE
    images = [] # 存储所有文档的所有页面图像
    page_wh_list = [] # 存储所有页面原始宽高

    images_with_extra_info = [] # 存储带有额外信息的图像列表
    # 遍历每个数据集
    for dataset in datasets:
        # 遍历数据集中的每个页面
        for index in range(len(dataset)):
            # 确定当前页面的语言
            if lang is None or lang == 'auto':
                _lang = dataset._lang # 使用数据集自带语言
            else:
                _lang = lang # 使用指定语言

            # 获取页面数据和图像
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            images.append(img_dict['img'])
            page_wh_list.append((img_dict['width'], img_dict['height']))

            # 确定当前页面是否需要 OCR
            if parse_method == 'auto':
                # 如果是自动模式，根据数据集的分类判断是否需要 OCR
                needs_ocr = (dataset.classify() == SupportedPdfParseMethod.OCR)
            else:
                # 否则，根据指定的 parse_method 判断
                needs_ocr = (parse_method == 'ocr')

            # 将图像、是否 OCR 标志、语言添加到列表中
            images_with_extra_info.append((images[-1], needs_ocr, _lang))

    # 将所有页面的图像信息进行分批
    batch_images = [images_with_extra_info[i:i+batch_size] for i in range(0, len(images_with_extra_info), batch_size)]
    results = [] # 存储所有页面的分析结果
    # 遍历每个批次
    for sn, batch_image in enumerate(batch_images):
        # 调用批处理分析函数 (注意：这里的 ocr 参数为 True，实际是否执行 OCR 取决于 batch_image 中每个元素的布尔值)
        _, result = may_batch_image_analyze(batch_image, sn, True, show_log, layout_model, formula_enable, table_enable)
        results.extend(result) # 添加批处理结果

    infer_results = [] # 存储每个数据集的 InferenceResult 对象
    from magic_pdf.operators.models import InferenceResult # 导入结果类
    # 遍历原始的数据集列表
    for index in range(len(datasets)):
        dataset = datasets[index]
        model_json = [] # 存储当前数据集的格式化结果
        # 遍历当前数据集的页面数量
        for i in range(len(dataset)):
            # 从总结果列表中弹出对应页面的结果和宽高
            result = results.pop(0)
            page_width, page_height = page_wh_list.pop(0)
            # 构建单页结果字典
            page_info = {'page_no': i, 'width': page_width, 'height': page_height}
            page_dict = {'layout_dets': result, 'page_info': page_info}
            model_json.append(page_dict) # 添加到当前数据集的结果列表
        # 为当前数据集创建 InferenceResult 对象
        infer_results.append(InferenceResult(model_json, dataset))
    # 返回包含所有数据集结果的列表
    return infer_results


# 可能用于多 GPU 或分布式处理的批处理图像分析函数
def may_batch_image_analyze(
        images_with_extra_info: list[(np.ndarray, bool, str)], # 包含图像、是否OCR标志、语言的元组列表
        idx: int, # 当前批次的索引或 ID (可能用于设备选择)
        ocr: bool, # 全局 OCR 标志 (可能已弃用或有默认作用)
        show_log: bool = False, # 是否显示日志
        layout_model=None, # 指定版面分析模型
        formula_enable=None, # 是否启用公式处理
        table_enable=None): # 是否启用表格识别
    # 设置 CUDA 可见设备 (通常用于多 GPU 环境，这里被注释掉了)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)

    # 导入实际执行批处理分析的类
    from magic_pdf.model.batch_analyze import BatchAnalyze

    # 获取模型单例管理器
    model_manager = ModelSingleton()

    # 从输入中提取纯图像列表 (这行被注释掉了，因为 BatchAnalyze 直接使用 images_with_extra_info)
    # images = [image for image, _, _ in images_with_extra_info]
    batch_ratio = 1 # 初始化批处理比例因子
    device = get_device() # 获取运行设备

    # --- NPU (华为昇腾) 相关设置 ---
    if str(device).startswith('npu'):
        import torch_npu # 导入 NPU 支持库
        if torch_npu.npu.is_available():
            # 如果 NPU 可用，设置编译模式 (这里关闭了 jit 编译)
            torch.npu.set_compile_mode(jit_compile=False)

    # --- 动态调整批处理比例 (基于 GPU 显存) ---
    if str(device).startswith('npu') or str(device).startswith('cuda'):
        # 获取 GPU 显存大小 (GB)，优先从环境变量读取虚拟显存大小
        gpu_memory = int(os.getenv('VIRTUAL_VRAM_SIZE', round(get_vram(device))))
        if gpu_memory is not None:
            # 根据显存大小设置批处理比例因子，显存越大，比例因子越大
            if gpu_memory >= 16:
                batch_ratio = 16
            elif gpu_memory >= 12:
                batch_ratio = 8
            elif gpu_memory >= 8:
                batch_ratio = 4
            elif gpu_memory >= 6:
                batch_ratio = 2
            else:
                batch_ratio = 1
            logger.info(f'gpu_memory: {gpu_memory} GB, batch_ratio: {batch_ratio}')

    # 记录文档分析开始时间 (注释掉了)
    # doc_analyze_start = time.time()

    # 初始化 BatchAnalyze 类，传入模型管理器和计算得到的批处理比例等参数
    batch_model = BatchAnalyze(model_manager, batch_ratio, show_log, layout_model, formula_enable, table_enable)
    # 调用 BatchAnalyze 实例处理图像批次
    results = batch_model(images_with_extra_info)

    # --- 清理内存 ---
    # gc_start = time.time() # 记录 GC 开始时间 (注释掉了)
    clean_memory(get_device()) # 调用内存清理函数，释放不再使用的 GPU/CPU 内存
    # gc_time = round(time.time() - gc_start, 2) # 计算 GC 时间 (注释掉了)
    # logger.debug(f'gc time: {gc_time}') # 记录 GC 时间 (注释掉了)

    # 计算并记录处理时间和速度 (注释掉了)
    # doc_analyze_time = round(time.time() - doc_analyze_start, 2)
    # doc_analyze_speed = round(len(images_with_extra_info) / doc_analyze_time, 2) if doc_analyze_time > 0 else 0
    # logger.debug(
    #     f'doc analyze time: {round(time.time() - doc_analyze_start, 2)},'
    #     f' speed: {doc_analyze_speed} pages/second'
    # )

    # 返回批次索引和处理结果列表
    return idx, results [3]