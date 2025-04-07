# 导入time库，用于计时
import time
# 导入torch库，用于PyTorch相关操作，例如检查GPU/NPU
import torch
# 导入loguru库，用于日志记录
from loguru import logger
# 导入numpy库，用于数值计算，特别是图像处理
import numpy as np
# 从自定义库中导入内存清理函数
from magic_pdf.libs.clean_memory import clean_memory

# 定义图像裁剪函数
def crop_img(input_res, input_np_img, crop_paste_x=0, crop_paste_y=0):
    """
    根据输入的多边形坐标裁剪图像，并在一个白色背景上粘贴裁剪后的图像，可以添加边距。

    Args:
        input_res (dict): 包含 'poly' 键的字典，'poly' 的值是一个列表，通常包含[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]形式的坐标。
        input_np_img (np.ndarray): 输入的原始图像，NumPy数组格式。
        crop_paste_x (int, optional): 在裁剪后的图像左右两侧添加的白色边距宽度。默认为0。
        crop_paste_y (int, optional): 在裁剪后的图像上下两侧添加的白色边距高度。默认为0。

    Returns:
        tuple: 包含两个元素的元组：
            - return_image (np.ndarray): 带边距的裁剪后图像。
            - return_list (list): 包含裁剪和粘贴相关信息的列表 [边距x, 边距y, 原始xmin, 原始ymin, 原始xmax, 原始ymax, 新宽度, 新高度]。
    """
    # 从输入结果中提取裁剪区域的左上角和右下角坐标
    crop_xmin, crop_ymin = int(input_res['poly'][0]), int(input_res['poly'][1])
    crop_xmax, crop_ymax = int(input_res['poly'][4]), int(input_res['poly'][5])
    # 计算添加边距后的新宽度和新高度
    crop_new_width = crop_xmax - crop_xmin + crop_paste_x * 2
    crop_new_height = crop_ymax - crop_ymin + crop_paste_y * 2
    # 创建一个白色背景的NumPy数组，尺寸为新宽度和新高度
    return_image = np.ones((crop_new_height, crop_new_width, 3), dtype=np.uint8) * 255
    # 使用NumPy切片从原始图像中裁剪出目标区域
    cropped_img = input_np_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    # 将裁剪出的图像粘贴到白色背景的中心位置（考虑边距）
    return_image[crop_paste_y:crop_paste_y + (crop_ymax - crop_ymin),
    crop_paste_x:crop_paste_x + (crop_xmax - crop_xmin)] = cropped_img
    # 创建一个列表，包含边距、原始坐标和新尺寸信息
    return_list = [crop_paste_x, crop_paste_y, crop_xmin, crop_ymin, crop_xmax, crop_ymax, crop_new_width,
                   crop_new_height]
    # 返回处理后的图像和信息列表
    return return_image, return_list

# 从布局检测结果中筛选出用于OCR、公式识别和表格识别的区域
def get_res_list_from_layout_res(layout_res):
    """
    根据布局检测结果，将不同类别的区域分别归类。

    Args:
        layout_res (list): 布局检测模型输出的结果列表，每个元素是一个包含 'category_id' 和 'poly' 的字典。

    Returns:
        tuple: 包含三个列表的元组：
            - ocr_res_list (list): 需要进行OCR的区域列表。
            - table_res_list (list): 表格区域列表。
            - single_page_mfdetrec_res (list): 数学公式区域列表（包含bbox信息）。
    """
    # 初始化用于存储OCR区域的列表
    ocr_res_list = []
    # 初始化用于存储表格区域的列表
    table_res_list = []
    # 初始化用于存储单页数学公式检测和识别区域的列表
    single_page_mfdetrec_res = []
    # 遍历布局检测结果中的每个区域
    for res in layout_res:
        # 获取区域的类别ID
        category_id = int(res['category_id'])
        # 如果类别ID是13或14，认为是数学公式区域
        if category_id in [13, 14]: # 通常 13: MFD_isolate (独立公式), 14: MFD_embedded (嵌入式公式)
            # 添加到公式列表中，只保留边界框(bbox)信息
            single_page_mfdetrec_res.append({
                "bbox": [int(res['poly'][0]), int(res['poly'][1]), # xmin, ymin
                         int(res['poly'][4]), int(res['poly'][5])], # xmax, ymax
            })
        # 如果类别ID是0, 1, 2, 4, 6, 7，认为是文本区域，需要进行OCR
        elif category_id in [0, 1, 2, 4, 6, 7]: # 通常 0: text, 1: title, 2: list, 4: caption, 6: header, 7: footer
            # 将整个结果添加到OCR列表中
            ocr_res_list.append(res)
        # 如果类别ID是5，认为是表格区域
        elif category_id in [5]: # 通常 5: table
            # 将整个结果添加到表格列表中
            table_res_list.append(res)
    # 返回分类后的三个列表
    return ocr_res_list, table_res_list, single_page_mfdetrec_res

# 清理显存函数
def clean_vram(device, vram_threshold=8):
    """
    检查设备的总显存，如果低于指定阈值，则尝试清理显存。

    Args:
        device (str or torch.device): 目标设备（例如 'cuda:0', 'npu:0', 'cpu'）。
        vram_threshold (int, optional): 显存阈值（单位：GB）。如果总显存小于等于此值，则触发清理。默认为8GB。
    """
    # 获取指定设备的总显存大小（GB）
    total_memory = get_vram(device)
    # 如果成功获取到显存大小，并且小于等于阈值
    if total_memory and total_memory <= vram_threshold:
        # 记录开始清理的时间
        gc_start = time.time()
        # 调用外部的显存清理函数
        clean_memory(device)
        # 计算清理耗时
        gc_time = round(time.time() - gc_start, 2)
        # 记录清理显存的耗时日志
        logger.info(f"gc time: {gc_time}")

# 获取设备显存大小函数
def get_vram(device):
    """
    获取指定设备的总显存大小（单位：GB）。

    Args:
        device (str or torch.device): 目标设备（例如 'cuda:0', 'npu:0', 'cpu'）。

    Returns:
        float or None: 设备的显存大小（GB），如果设备不支持或不可用则返回None。
    """
    # 检查是否有可用的CUDA设备，并且指定的设备不是CPU
    if torch.cuda.is_available() and device != 'cpu':
        # 获取CUDA设备的总显存（字节），并转换为GB
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # 将字节转换为 GB
        # 返回显存大小
        return total_memory
    # 检查设备是否为NPU（华为昇腾）
    elif str(device).startswith("npu"):
        try:
            # 尝试导入torch_npu库
            import torch_npu
            # 检查NPU设备是否可用
            if torch_npu.npu.is_available():
                # 获取NPU设备的总显存（字节），并转换为GB
                total_memory = torch_npu.npu.get_device_properties(device).total_memory / (1024 ** 3)  # 转为 GB
                # 返回显存大小
                return total_memory
        except ImportError:
            # 如果torch_npu库未安装，记录错误日志
            logger.error("torch_npu not installed, cannot get NPU memory info.")
            # 返回None
            return None
    # 如果是CPU或其他不支持的设备
    else:
        # 返回None
        return None