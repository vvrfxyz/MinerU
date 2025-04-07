import os  # 导入操作系统接口模块，用于环境变量等操作
import time  # 导入时间模块，用于计时等操作

import numpy as np  # 导入NumPy库，用于高效的数值计算，特别是数组操作
import torch  # 导入PyTorch库，用于深度学习模型的构建和计算

# --- 环境配置 ---
# 设置环境变量 'FLAGS_npu_jit_compile' 为 '0'，可能是为了关闭PaddlePaddle在NPU（华为昇腾AI处理器）上的即时编译（JIT）功能
os.environ['FLAGS_npu_jit_compile'] = '0'
# 设置环境变量 'FLAGS_use_stride_kernel' 为 '0'，可能与PaddlePaddle的算子选择有关，禁用某种stride kernel优化
os.environ['FLAGS_use_stride_kernel'] = '0'
# 设置环境变量 'PYTORCH_ENABLE_MPS_FALLBACK' 为 '1'，允许PyTorch在Apple Silicon的MPS（Metal Performance Shaders）后端上遇到不支持的操作时，自动回退到CPU执行
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# 设置环境变量 'NO_ALBUMENTATIONS_UPDATE' 为 '1'，禁止图像增强库albumentations自动检查更新
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


from loguru import logger  # 导入loguru库，用于提供更方便、更强大的日志记录功能

from magic_pdf.model.sub_modules.model_utils import get_vram  # 从项目内部导入获取显存大小的工具函数
from magic_pdf.config.enums import SupportedPdfParseMethod  # 从项目内部导入支持的PDF解析方法枚举
import magic_pdf.model as model_config  # 导入项目内部的模型配置模块
from magic_pdf.data.dataset import Dataset  # 从项目内部导入数据集类
from magic_pdf.libs.clean_memory import clean_memory  # 从项目内部导入清理内存（特别是GPU显存）的工具函数
from magic_pdf.libs.config_reader import (  # 从项目内部导入配置读取相关的函数
    get_device,  # 获取计算设备（如'cpu', 'cuda:0', 'npu:0'）
    get_formula_config,  # 获取公式识别相关的配置
    get_layout_config,  # 获取版面分析相关的配置
    get_local_models_dir,  # 获取本地模型文件的存储目录
    get_table_recog_config  # 获取表格识别相关的配置
)
from magic_pdf.model.model_list import MODEL  # 从项目内部导入模型类型枚举或列表

# --- 模型单例类 ---
class ModelSingleton:
    """
    实现单例模式，确保全局只有一个模型管理器实例。
    用于缓存和复用已初始化的模型，避免重复加载，提高效率。
    """
    _instance = None  # 类变量，存储单例实例
    _models = {}  # 类变量，用字典缓存已初始化的模型，键是配置元组，值是模型实例

    def __new__(cls, *args, **kwargs):
        """
        重写__new__方法，实现单例逻辑。
        如果实例不存在，则创建新实例；否则，返回现有实例。
        """
        if cls._instance is None:  # 检查实例是否已创建
            cls._instance = super().__new__(cls)  # 如果未创建，调用父类的__new__方法创建实例
        return cls._instance  # 返回单例实例

    def get_model(
        self,
        ocr: bool,  # 是否启用OCR
        show_log: bool,  # 是否显示详细日志
        lang=None,  # 指定处理的语言，None表示自动检测或使用默认
        layout_model=None,  # 指定布局模型，None表示使用配置默认
        formula_enable=None,  # 是否启用公式识别，None表示使用配置默认
        table_enable=None,  # 是否启用表格识别，None表示使用配置默认
    ):
        """
        根据指定的配置获取模型实例。
        如果具有相同配置的模型已被初始化并缓存，则直接返回缓存的实例。
        否则，调用 custom_model_init 初始化新模型，缓存后再返回。

        Args:
            ocr (bool): 是否需要OCR功能。
            show_log (bool): 是否显示日志。
            lang (str, optional): 语言。默认为 None。
            layout_model (str, optional): 布局模型名称。默认为 None。
            formula_enable (bool, optional): 是否启用公式识别。默认为 None。
            table_enable (bool, optional): 是否启用表格识别。默认为 None。

        Returns:
            object: 初始化或缓存的模型实例。
        """
        # 使用配置参数创建一个元组作为缓存的键
        key = (ocr, show_log, lang, layout_model, formula_enable, table_enable)
        if key not in self._models:  # 检查此配置的模型是否已在缓存中
            # 如果不在缓存中，调用 custom_model_init 初始化模型
            self._models[key] = custom_model_init(
                ocr=ocr,
                show_log=show_log,
                lang=lang,
                layout_model=layout_model,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
            logger.info(f"Model cache initialized for key: {key}") # 记录日志，表示为这个配置初始化了模型
        else:
            logger.info(f"Using cached model for key: {key}") # 记录日志，表示使用了缓存的模型
        return self._models[key]  # 返回缓存或新初始化的模型实例


# --- 模型初始化函数 ---
def custom_model_init(
    ocr: bool = False,  # 是否启用OCR，默认为False
    show_log: bool = False,  # 是否显示详细日志，默认为False
    lang=None,  # 指定处理的语言，None表示自动检测或使用默认
    layout_model=None,  # 指定布局模型，None表示使用配置默认
    formula_enable=None,  # 是否启用公式识别，None表示使用配置默认
    table_enable=None,  # 是否启用表格识别，None表示使用配置默认
):
    """
    根据配置初始化并返回具体的模型实例（Paddle或PEK）。

    Args:
        ocr (bool): 是否需要OCR功能。
        show_log (bool): 是否显示日志。
        lang (str, optional): 语言。默认为 None。
        layout_model (str, optional): 布局模型名称。默认为 None。
        formula_enable (bool, optional): 是否启用公式识别。默认为 None。
        table_enable (bool, optional): 是否启用表格识别。默认为 None。

    Returns:
        object: 初始化后的模型实例。
    """
    model = None  # 初始化模型变量
    # 根据全局配置 `model_config.__model_mode__` 决定使用哪个模型框架
    if model_config.__model_mode__ == 'lite':  # 如果是 'lite' 模式
        logger.warning(  # 记录警告信息
            'The Lite mode is provided for developers to conduct testing only, and the output quality is '
            'not guaranteed to be reliable.'
        )
        model = MODEL.Paddle  # 选择 Paddle 模型
    elif model_config.__model_mode__ == 'full':  # 如果是 'full' 模式
        model = MODEL.PEK  # 选择 PEK (PDF Extract Kit) 模型

    # 检查是否允许使用内部模型（根据全局配置 `model_config.__use_inside_model__`）
    if model_config.__use_inside_model__:
        model_init_start = time.time()  # 记录模型初始化开始时间
        if model == MODEL.Paddle:  # 如果选择的是 Paddle 模型
            from magic_pdf.model.pp_structure_v2 import CustomPaddleModel  # 导入 Paddle 模型类

            # 实例化 Paddle 模型
            custom_model = CustomPaddleModel(ocr=ocr, show_log=show_log, lang=lang)
        elif model == MODEL.PEK:  # 如果选择的是 PEK 模型
            from magic_pdf.model.pdf_extract_kit import CustomPEKModel  # 导入 PEK 模型类

            # --- 获取 PEK 模型所需的配置 ---
            local_models_dir = get_local_models_dir()  # 获取本地模型存储路径
            device = get_device()  # 获取计算设备

            layout_config = get_layout_config()  # 获取布局分析配置
            if layout_model is not None:  # 如果函数参数指定了布局模型，则覆盖配置文件中的设置
                layout_config['model'] = layout_model

            formula_config = get_formula_config()  # 获取公式识别配置
            if formula_enable is not None:  # 如果函数参数指定了是否启用公式识别，则覆盖配置文件中的设置
                formula_config['enable'] = formula_enable

            table_config = get_table_recog_config()  # 获取表格识别配置
            if table_enable is not None:  # 如果函数参数指定了是否启用表格识别，则覆盖配置文件中的设置
                table_config['enable'] = table_enable

            # --- 准备 PEK 模型的初始化参数 ---
            model_input = {
                'ocr': ocr,  # 是否启用OCR
                'show_log': show_log,  # 是否显示日志
                'models_dir': local_models_dir,  # 模型文件目录
                'device': device,  # 计算设备
                'table_config': table_config,  # 表格识别配置
                'layout_config': layout_config,  # 布局分析配置
                'formula_config': formula_config,  # 公式识别配置
                'lang': lang,  # 语言
            }

            # 实例化 PEK 模型
            custom_model = CustomPEKModel(**model_input)
        else:  # 如果模型类型不是 Paddle 或 PEK
            logger.error('Not allow model_name!')  # 记录错误日志
            exit(1)  # 退出程序
        model_init_cost = time.time() - model_init_start  # 计算模型初始化耗时
        logger.info(f'model init cost: {model_init_cost}')  # 记录模型初始化耗时日志
    else:  # 如果不允许使用内部模型
        logger.error('use_inside_model is False, not allow to use inside model')  # 记录错误日志
        exit(1)  # 退出程序

    return custom_model  # 返回初始化好的模型实例

# --- 单文档分析函数 ---
def doc_analyze(
    dataset: Dataset,  # 输入的 Dataset 对象，包含文档信息
    ocr: bool = False,  # 是否对整个文档强制使用OCR，默认为False
    show_log: bool = False,  # 是否显示详细日志，默认为False
    start_page_id=0,  # 要分析的起始页面索引（从0开始），默认为0
    end_page_id=None,  # 要分析的结束页面索引（包含），None表示到最后一页，默认为None
    lang=None,  # 指定语言，None表示自动检测或使用默认
    layout_model=None,  # 指定布局模型，None表示使用配置默认
    formula_enable=None,  # 是否启用公式识别，None表示使用配置默认
    table_enable=None,  # 是否启用表格识别，None表示使用配置默认
):
    """
    对单个文档（由Dataset对象表示）进行页面分析。

    Args:
        dataset (Dataset): 包含文档页面信息的 Dataset 对象。
        ocr (bool): 是否对所有页面强制使用OCR。
        show_log (bool): 是否显示日志。
        start_page_id (int): 开始处理的页面索引。
        end_page_id (int, optional): 结束处理的页面索引（包含）。默认为 None，表示处理到最后一页。
        lang (str, optional): 语言。默认为 None。
        layout_model (str, optional): 布局模型名称。默认为 None。
        formula_enable (bool, optional): 是否启用公式识别。默认为 None。
        table_enable (bool, optional): 是否启用表格识别。默认为 None。

    Returns:
        InferenceResult: 包含分析结果和原始Dataset对象的封装对象。
    """
    # 确定结束页面索引，如果 end_page_id 为 None 或负数，则设为最后一页的索引
    end_page_id = (
        end_page_id
        if end_page_id is not None and end_page_id >= 0
        else len(dataset) - 1
    )

    # 从环境变量获取最小批处理推理大小，默认为200
    MIN_BATCH_INFERENCE_SIZE = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 200))
    images = []  # 存储需要处理的页面图像
    page_wh_list = []  # 存储对应页面的宽度和高度

    # 遍历数据集中的所有页面
    for index in range(len(dataset)):
        # 只处理指定范围内的页面
        if start_page_id <= index <= end_page_id:
            page_data = dataset.get_page(index)  # 获取页面数据
            img_dict = page_data.get_image()  # 获取页面图像及其信息
            images.append(img_dict['img'])  # 添加图像到列表
            page_wh_list.append((img_dict['width'], img_dict['height']))  # 添加页面宽高到列表

    # 准备带有附加信息（图像、是否OCR、语言）的图像列表
    if lang is None or lang == 'auto': # 如果未指定语言或设为自动
        # 使用数据集自带的语言信息
        images_with_extra_info = [(images[i], ocr, dataset._lang) for i in range(len(images))]
    else: # 如果指定了语言
        # 使用指定的语言
        images_with_extra_info = [(images[i], ocr, lang) for i in range(len(images))]

    # 如果图像数量达到或超过最小批处理大小，则进行分批
    if len(images) >= MIN_BATCH_INFERENCE_SIZE:
        batch_size = MIN_BATCH_INFERENCE_SIZE  # 设置批处理大小
        # 将图像列表切分成多个批次
        batch_images = [images_with_extra_info[i:i+batch_size] for i in range(0, len(images_with_extra_info), batch_size)]
    else: # 如果图像数量不足一个批次
        # 将所有图像作为一个批次
        batch_images = [images_with_extra_info]

    results = []  # 存储所有批次的分析结果
    # 遍历每个批次进行处理
    for sn, batch_image in enumerate(batch_images):
        # 调用 may_batch_image_analyze 处理单个批次
        # 注意：这里的 ocr 参数传递给 may_batch_image_analyze 似乎有些冗余，因为 ocr 标志已经包含在 batch_image 的元组中了
        _, result = may_batch_image_analyze(batch_image, sn, ocr, show_log, layout_model, formula_enable, table_enable)
        results.extend(result)  # 将当前批次的结果添加到总结果列表中

    model_json = []  # 存储最终格式化的JSON输出
    # 再次遍历数据集的所有页面索引，以构建与原始页面顺序一致的输出
    for index in range(len(dataset)):
        if start_page_id <= index <= end_page_id:  # 如果页面在处理范围内
            result = results.pop(0)  # 从结果列表中取出对应的分析结果
            page_width, page_height = page_wh_list.pop(0)  # 取出对应的页面宽高
        else:  # 如果页面不在处理范围内
            result = []  # 分析结果为空列表
            page_height = 0  # 宽高设为0或保持默认
            page_width = 0

        # 构建页面信息字典
        page_info = {'page_no': index, 'width': page_width, 'height': page_height}
        # 构建包含布局检测结果和页面信息的字典
        page_dict = {'layout_dets': result, 'page_info': page_info}
        model_json.append(page_dict)  # 添加到最终输出列表

    from magic_pdf.operators.models import InferenceResult  # 导入推理结果封装类
    # 使用模型分析结果和原始数据集创建 InferenceResult 对象并返回
    return InferenceResult(model_json, dataset)

# --- 批量文档分析函数 ---
def batch_doc_analyze(
    datasets: list[Dataset],  # 输入的 Dataset 对象列表，每个对象代表一个文档
    parse_method: str,  # 解析方法 ('ocr', 'digital', 'auto')
    show_log: bool = False,  # 是否显示详细日志，默认为False
    lang=None,  # 指定语言，None表示自动检测或使用默认
    layout_model=None,  # 指定布局模型，None表示使用配置默认
    formula_enable=None,  # 是否启用公式识别，None表示使用配置默认
    table_enable=None,  # 是否启用表格识别，None表示使用配置默认
):
    """
    对一批文档（由Dataset对象列表表示）进行页面分析。会将所有文档的页面合并处理。

    Args:
        datasets (list[Dataset]): 包含多个文档信息的 Dataset 对象列表。
        parse_method (str): 解析方法，如 'ocr', 'digital', 'auto'。用于决定是否对页面使用OCR。
        show_log (bool): 是否显示日志。
        lang (str, optional): 语言。默认为 None。
        layout_model (str, optional): 布局模型名称。默认为 None。
        formula_enable (bool, optional): 是否启用公式识别。默认为 None。
        table_enable (bool, optional): 是否启用表格识别。默认为 None。

    Returns:
        list[InferenceResult]: 包含每个文档分析结果的 InferenceResult 对象列表。
    """
    # 从环境变量获取最小批处理推理大小，默认为200
    MIN_BATCH_INFERENCE_SIZE = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 200))
    batch_size = MIN_BATCH_INFERENCE_SIZE  # 设置批处理大小
    images = []  # 存储所有文档的所有页面图像
    page_wh_list = []  # 存储所有页面对应的宽度和高度

    images_with_extra_info = []  # 存储带有附加信息（图像、是否OCR、语言）的图像列表
    # 遍历输入的每个Dataset对象（每个文档）
    for dataset in datasets:
        # 遍历当前文档的每一页
        for index in range(len(dataset)):
            # 确定当前页面的语言
            if lang is None or lang == 'auto': # 如果未指定语言或设为自动
                _lang = dataset._lang # 使用数据集自带的语言
            else: # 如果指定了语言
                _lang = lang # 使用指定的语言

            page_data = dataset.get_page(index)  # 获取页面数据
            img_dict = page_data.get_image()  # 获取页面图像及其信息
            images.append(img_dict['img'])  # 添加图像到列表
            page_wh_list.append((img_dict['width'], img_dict['height']))  # 添加页面宽高到列表

            # 确定当前页面是否需要OCR
            if parse_method == 'auto': # 如果解析方法是自动
                # 根据数据集的分类结果判断是否需要OCR
                # dataset.classify() 可能返回 SupportedPdfParseMethod.OCR 或其他
                should_ocr = (dataset.classify() == SupportedPdfParseMethod.OCR)
            else: # 如果解析方法是指定的 'ocr' 或 'digital'
                # 如果 parse_method 是 'ocr'，则需要OCR
                should_ocr = (parse_method == 'ocr')

            # 将图像、OCR标志和语言打包成元组添加到列表
            images_with_extra_info.append((images[-1], should_ocr, _lang))

    # 将所有页面的图像列表切分成多个批次
    batch_images = [images_with_extra_info[i:i+batch_size] for i in range(0, len(images_with_extra_info), batch_size)]
    results = []  # 存储所有批次的分析结果
    # 遍历每个批次进行处理
    for sn, batch_image in enumerate(batch_images):
        # 调用 may_batch_image_analyze 处理单个批次
        # 注意：这里总是传递 ocr=True，但实际的OCR决策是在 images_with_extra_info 中每个元素决定的。
        # 这个 True 可能影响模型选择（如果模型初始化依赖这个顶层 ocr 参数）或日志记录等。需要结合 may_batch_image_analyze 内部逻辑理解。
        # 更合理的做法可能是让 may_batch_image_analyze 完全依赖 batch_image 中的 ocr 标志。
        _, result = may_batch_image_analyze(batch_image, sn, True, show_log, layout_model, formula_enable, table_enable)
        results.extend(result)  # 将当前批次的结果添加到总结果列表中

    infer_results = []  # 存储每个文档最终的 InferenceResult 对象
    from magic_pdf.operators.models import InferenceResult  # 导入推理结果封装类
    # 遍历原始的 Dataset 列表，将聚合的 results 分配回各自的文档
    for index in range(len(datasets)):
        dataset = datasets[index]  # 获取当前文档的 Dataset 对象
        model_json = []  # 存储当前文档的格式化JSON输出
        # 遍历当前文档的页数
        for i in range(len(dataset)):
            result = results.pop(0)  # 从总结果列表中按顺序取出属于当前页面的结果
            page_width, page_height = page_wh_list.pop(0)  # 取出对应的页面宽高
            # 构建页面信息字典
            page_info = {'page_no': i, 'width': page_width, 'height': page_height}
            # 构建包含布局检测结果和页面信息的字典
            page_dict = {'layout_dets': result, 'page_info': page_info}
            model_json.append(page_dict)  # 添加到当前文档的输出列表
        # 为当前文档创建 InferenceResult 对象并添加到最终结果列表
        infer_results.append(InferenceResult(model_json, dataset))
    return infer_results  # 返回包含所有文档分析结果的列表

# --- 图像批处理分析核心函数 ---
def may_batch_image_analyze(
            images_with_extra_info: list[(np.ndarray, bool, str)], # 输入：包含(图像, 是否OCR, 语言)元组的列表
            idx: int, # 当前批次的索引（似乎未使用？）
            ocr: bool, # 顶层OCR标志（可能影响模型获取或日志）
            show_log: bool = False, # 是否显示日志
            layout_model=None, # 指定布局模型
            formula_enable=None, # 是否启用公式识别
            table_enable=None): # 是否启用表格识别
    """
    对一批图像进行分析处理，是实际执行模型推理的地方。
    会根据显存动态调整内部处理批次的大小（batch_ratio）。

    Args:
        images_with_extra_info (list): 包含 (图像np.ndarray, 是否ocr, 语言str) 元组的列表。
        idx (int): 批次索引 (似乎未直接使用其值来分配设备)。
        ocr (bool): 一个总体的OCR标志，可能用于获取合适的模型实例。
        show_log (bool): 是否显示日志。
        layout_model (str, optional): 布局模型名称。
        formula_enable (bool, optional): 是否启用公式识别。
        table_enable (bool, optional): 是否启用表格识别。

    Returns:
        tuple: (批次索引 idx, 分析结果列表 list)。
    """
    # 设置可见的CUDA设备，这里被注释掉了，意味着可能使用默认或外部设置的设备
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)

    from magic_pdf.model.batch_analyze import BatchAnalyze  # 导入实际执行批量分析的类

    model_manager = ModelSingleton()  # 获取模型单例管理器实例

    # 从输入列表中提取图像 (这行代码被注释掉了，因为 BatchAnalyze 类可能直接接收 images_with_extra_info)
    # images = [image for image, _, _ in images_with_extra_info]
    batch_ratio = 1  # 初始化内部批处理比例因子，默认为1
    device = get_device()  # 获取计算设备

    # 如果设备是 NPU
    if str(device).startswith('npu'):
        import torch_npu  # 导入 NPU 相关库
        if torch_npu.npu.is_available():  # 检查 NPU 是否可用
            # 设置 NPU 的编译模式，这里关闭了 JIT 编译 (与文件开头的环境变量设置一致)
            torch.npu.set_compile_mode(jit_compile=False)

    # 如果设备是 NPU 或 CUDA (即 GPU 设备)
    if str(device).startswith('npu') or str(device).startswith('cuda'):
        # 获取 GPU 显存大小（GB），优先从环境变量 VIRTUAL_VRAM_SIZE 读取，否则通过 get_vram 函数获取
        gpu_memory = int(os.getenv('VIRTUAL_VRAM_SIZE', round(get_vram(device))))
        if gpu_memory is not None: # 如果成功获取显存大小
            # 根据显存大小动态调整批处理比例因子 batch_ratio
            if gpu_memory >= 16: # 显存 >= 16GB
                batch_ratio = 16
            elif gpu_memory >= 12: # 显存 >= 12GB
                batch_ratio = 8
            elif gpu_memory >= 8: # 显存 >= 8GB
                batch_ratio = 4
            elif gpu_memory >= 6: # 显存 >= 6GB
                batch_ratio = 2
            else: # 显存 < 6GB
                batch_ratio = 1
            logger.info(f'gpu_memory: {gpu_memory} GB, batch_ratio: {batch_ratio}') # 记录显存和确定的batch_ratio


    # 记录文档分析开始时间 (注释掉了)
    # doc_analyze_start = time.time()

    # 实例化 BatchAnalyze 类，传入模型管理器、批处理比例因子和配置参数
    # 注意：这里的 ocr 参数也被传入了 BatchAnalyze，需要看其内部如何使用
    batch_model = BatchAnalyze(model_manager, batch_ratio, show_log, layout_model, formula_enable, table_enable)
    # 调用 BatchAnalyze 实例处理图像批次，获取结果
    # BatchAnalyze 内部会使用 model_manager.get_model 来获取具体模型，并根据 batch_ratio 进行可能的内部批处理
    results = batch_model(images_with_extra_info)

    # 记录垃圾回收开始时间 (注释掉了)
    # gc_start = time.time()
    # 清理计算设备上的内存（主要是显存）
    clean_memory(get_device())
    # 计算并记录垃圾回收耗时 (注释掉了)
    # gc_time = round(time.time() - gc_start, 2)
    # logger.debug(f'gc time: {gc_time}')

    # 计算并记录文档分析总耗时和速度 (注释掉了)
    # doc_analyze_time = round(time.time() - doc_analyze_start, 2)
    # doc_analyze_speed = round(len(images) / doc_analyze_time, 2) # 注意: 如果用len(images)，需要取消上面 images 提取行的注释
    # logger.debug(
    #     f'doc analyze time: {round(time.time() - doc_analyze_start, 2)},'
    #     f' speed: {doc_analyze_speed} pages/second'
    # )

    # 返回批次索引和该批次的处理结果列表
    return idx, results