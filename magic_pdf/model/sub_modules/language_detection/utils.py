# Copyright (c) Opendatalab. All rights reserved. # 版权声明，保留所有权利归 Opendatalab 所有。
import os # 导入os模块，用于操作系统相关功能，如路径操作。
from pathlib import Path # 导入pathlib模块，提供面向对象的路径操作。
import yaml # 导入yaml模块，用于处理YAML配置文件。
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 设置环境变量，禁止albumentations库检查更新。
from magic_pdf.config.constants import MODEL_NAME # 从项目配置常量中导入模型名称。
from magic_pdf.data.utils import load_images_from_pdf # 从数据工具中导入从PDF加载图像的函数。
from magic_pdf.libs.config_reader import get_local_models_dir, get_device # 从库的配置读取器中导入获取本地模型目录和设备（CPU/GPU）的函数。
from magic_pdf.libs.pdf_check import extract_pages # 从库的PDF检查模块中导入提取PDF页面的函数。
from magic_pdf.model.model_list import AtomicModel # 从模型列表中导入原子模型枚举。
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton # 从模型初始化子模块中导入原子模型单例管理器。

def get_model_config():
    """
    获取模型配置信息。

    该函数负责读取模型的配置，包括本地模型存储路径、运行设备以及从YAML文件中加载的详细配置。

    Returns:
        tuple: 包含项目根目录、本地模型目录、运行设备和配置字典的元组。
               (root_dir, local_models_dir, device, configs)
    """
    local_models_dir = get_local_models_dir() # 获取本地存储模型的目录路径。
    device = get_device() # 获取运行设备（如 'cpu' 或 'cuda:0'）。
    current_file_path = os.path.abspath(__file__) # 获取当前脚本文件的绝对路径。
    # 获取项目的根目录（假设当前文件在项目的某个子目录下，向上追溯3级）。
    # 这种方式依赖于当前文件的具体位置，可能不够健壮。
    root_dir = Path(current_file_path).parents[3]
    # 构建模型配置文件的目录路径。
    model_config_dir = os.path.join(root_dir, 'resources', 'model_config')
    # 构建模型配置文件的完整路径 (model_configs.yaml)。
    config_path = os.path.join(model_config_dir, 'model_configs.yaml')
    # 使用 'utf-8' 编码打开配置文件进行读取。
    with open(config_path, 'r', encoding='utf-8') as f:
        # 使用yaml库安全地加载配置文件内容。FullLoader可以加载任意YAML标签。
        configs = yaml.load(f, Loader=yaml.FullLoader)
    # 返回项目根目录、本地模型目录、设备和加载的配置信息。
    return root_dir, local_models_dir, device, configs

def get_text_images(simple_images):
    """
    从页面图像列表中提取文本区域的图像。

    Args:
        simple_images (list): 包含页面图像信息的字典列表。
                              每个字典应至少包含一个 'img' 键，值为图像数据（如numpy数组）。

    Returns:
        list: 包含从输入图像中提取出的所有文本块图像（numpy数组）的列表。
    """
    # 调用函数获取模型配置信息，这里只关心模型目录、设备和配置字典。
    # _ 变量用于接收不关心的返回值（这里是root_dir）。
    _, local_models_dir, device, configs = get_model_config()
    # 获取原子模型单例管理器实例，确保模型只被加载一次。
    atom_model_manager = AtomModelSingleton()
    # 获取布局检测模型实例（这里指定使用 DocLayout_YOLO 模型）。
    temp_layout_model = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.Layout, # 指定原子模型类型为布局检测。
        layout_model_name=MODEL_NAME.DocLayout_YOLO, # 指定具体的布局模型名称。
        # 构建布局模型权重文件的完整路径。路径从配置中读取文件名，并与本地模型目录拼接。
        doclayout_yolo_weights=str(
            os.path.join(
                local_models_dir, configs['weights'][MODEL_NAME.DocLayout_YOLO]
            )
        ),
        device=device, # 指定模型运行的设备。
    )

    text_images = [] # 初始化一个空列表，用于存储提取到的文本区域图像。
    # 遍历输入的每个页面图像信息。
    for simple_image in simple_images:
        image = simple_image['img'] # 获取当前页面的图像数据（通常是numpy数组）。
        # 使用布局模型对当前页面图像进行预测，得到布局元素的检测结果（列表）。
        layout_res = temp_layout_model.predict(image)
        # 遍历当前页面检测到的所有布局元素结果。
        for res in layout_res:
            # 检查当前元素的类别ID是否为1。这里假设类别ID 1 代表文本块。
            if res['category_id'] in [1]:
                # 从检测结果的多边形坐标 'poly' 中提取边界框的左上角(x1, y1)和右下角(x2, y2)坐标。
                # 注意：'poly' 可能包含多个点，这里假设它至少包含能够确定矩形框的点，并按特定顺序排列。
                # _ 用于忽略不使用的坐标值。
                x1, y1, _,_ , x2, y2, _,_ = res['poly']
                # 进行初步的尺寸过滤：如果检测到的区域宽度和高度都小于100像素，则认为其过小或可能是噪声，予以忽略。
                if x2 - x1 < 100 and y2 - y1 < 100:
                    continue # 跳过当前结果，继续处理下一个。
                # 使用numpy切片从原图中截取文本块区域的图像。注意：图像坐标系通常y轴向下，x轴向右。
                # image[y1:y2, x1:x2] 表示截取y坐标从y1到y2（不含y2），x坐标从x1到x2（不含x2）的区域。
                text_images.append(image[y1:y2, x1:x2])
    # 返回包含所有提取到的、经过尺寸过滤的文本块图像的列表。
    return text_images

def auto_detect_lang(pdf_bytes: bytes):
    """
    自动检测PDF文件内容的语言。

    该函数通过对PDF进行采样、渲染成图像、提取文本区域、最后使用语言检测模型来判断语言。

    Args:
        pdf_bytes (bytes): 输入的PDF文件内容的字节流。

    Returns:
        str: 检测到的主要语言标识符（例如 'en', 'zh'）。具体格式取决于语言检测模型的输出。
    """
    # 从原始PDF字节流中提取部分页面（可能用于加速处理，具体逻辑在extract_pages内部）。
    sample_docs = extract_pages(pdf_bytes)
    # 将提取出的页面（通常是 PyMuPDF Document 对象）转换回字节流格式。
    sample_pdf_bytes = sample_docs.tobytes()
    # 将采样后的PDF字节流加载为图像列表。设置DPI (dots per inch) 为200，影响渲染图像的分辨率。
    simple_images = load_images_from_pdf(sample_pdf_bytes, dpi=200)
    # 调用 get_text_images 函数，从渲染的页面图像中提取出文本块的图像。
    text_images = get_text_images(simple_images)
    # 初始化语言检测模型（这里指定使用 YOLO_V11_LangDetect 模型）。
    langdetect_model = model_init(MODEL_NAME.YOLO_V11_LangDetect)
    # 使用语言检测模型对提取出的文本块图像列表进行语言检测。
    lang = langdetect_model.do_detect(text_images)
    # 返回检测到的语言结果。
    return lang

def model_init(model_name: str):
    """
    根据指定的模型名称初始化并返回模型实例。

    Args:
        model_name (str): 需要初始化的模型名称（应为 MODEL_NAME 枚举中的值）。

    Returns:
        object: 初始化后的模型实例。

    Raises:
        ValueError: 如果提供的 model_name 不被支持或未找到。
    """
    # 获取原子模型单例管理器实例。
    atom_model_manager = AtomModelSingleton()
    # 判断需要初始化的模型是否是语言检测模型 YOLO_V11_LangDetect。
    if model_name == MODEL_NAME.YOLO_V11_LangDetect:
        # 获取项目根目录和运行设备信息。本地模型目录和配置在此处未使用。
        root_dir, _, device,_ = get_model_config()
        # 通过原子模型管理器获取语言检测模型实例。
        model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.LangDetect, # 指定原子模型类型为语言检测。
            langdetect_model_name=MODEL_NAME.YOLO_V11_LangDetect, # 指定具体的语言检测模型名称。
            # 构建语言检测模型权重文件的完整路径。注意：这里权重路径是硬编码的，相对于项目根目录。
            langdetect_model_weight=str(os.path.join(root_dir, 'resources', 'yolov11-langdetect', 'yolo_v11_ft.pt')),
            device=device, # 指定模型运行的设备。
        )
    # 如果传入的模型名称不是当前函数支持的类型。
    else:
        # 抛出 ValueError 异常，提示找不到指定的模型名称。
        raise ValueError(f"model_name {model_name} not found")
    # 返回成功初始化后的模型实例。
    return model
