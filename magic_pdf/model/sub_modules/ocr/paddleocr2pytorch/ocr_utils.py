# Copyright (c) Opendatalab. All rights reserved.
import copy
import cv2
import numpy as np
from magic_pdf.pre_proc.ocr_dict_merge import merge_spans_to_line # 导入用于合并span到行的函数
from magic_pdf.libs.boxbase import __is_overlaps_y_exceeds_threshold # 导入检查Y轴重叠是否超过阈值的函数

def img_decode(content: bytes):
    """
    将图像的字节内容解码为OpenCV图像对象 (NumPy数组)。
    Args:
        content (bytes): 图像文件的字节流。
    Returns:
        np.ndarray: 解码后的OpenCV图像 (NumPy数组)。
    """
    np_arr = np.frombuffer(content, dtype=np.uint8) # 从字节缓冲区创建一个NumPy uint8数组
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED) # 使用OpenCV解码NumPy数组为图像，保持原始通道数（包括alpha通道）

def check_img(img):
    """
    检查输入的图像，确保它是一个有效的OpenCV BGR格式的NumPy数组。
    如果输入是字节流，则解码。如果是灰度图，则转换为BGR。
    Args:
        img (bytes or np.ndarray): 输入的图像，可以是字节流或者NumPy数组。
    Returns:
        np.ndarray: BGR格式的OpenCV图像 (NumPy数组)。
    """
    if isinstance(img, bytes): # 检查输入是否为字节流
        img = img_decode(img) # 如果是字节流，解码为图像
    if isinstance(img, np.ndarray) and len(img.shape) == 2: # 检查输入是否为NumPy数组且只有两个维度（灰度图）
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # 如果是灰度图，转换为BGR格式
    return img # 返回处理后的BGR图像

def alpha_to_color(img, alpha_color=(255, 255, 255)):
    """
    将带有alpha通道的图像（RGBA）与指定背景色（默认为白色）混合，去除alpha通道。
    Args:
        img (np.ndarray): 输入图像，可能是RGBA格式。
        alpha_color (tuple, optional): 用于混合的背景颜色 (B, G, R)。默认为白色 (255, 255, 255)。
    Returns:
        np.ndarray: 混合后的BGR图像。
    """
    # 检查图像是否有3个维度且第三个维度的大小为4（即RGBA格式）
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img) # 分离B, G, R, A四个通道
        alpha = A / 255.0 # 将Alpha通道的值归一化到0-1范围
        # 使用alpha混合公式计算新的R, G, B值: new_color = bg_color * (1 - alpha) + original_color * alpha
        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8) # 计算新的R通道
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8) # 计算新的G通道
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8) # 计算新的B通道
        img = cv2.merge((B, G, R)) # 合并新的B, G, R通道为BGR图像
    return img # 返回处理后的图像

def preprocess_image(_image):
    """
    对图像进行预处理，目前主要处理alpha通道。
    Args:
        _image (np.ndarray): 输入的OpenCV图像。
    Returns:
        np.ndarray: 预处理后的BGR图像。
    """
    alpha_color = (255, 255, 255) # 定义用于alpha混合的背景色（白色）
    _image = alpha_to_color(_image, alpha_color) #调用alpha_to_color函数处理透明度
    return _image # 返回处理后的图像

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    按从上到下、从左到右的顺序对文本框进行排序。
    args:
        dt_boxes(array): 检测到的文本框，形状为 [N, 4, 2]，每个框包含4个角点坐标。
    return:
        sorted boxes(array) with shape [N, 4, 2] 排序后的文本框列表。
    """
    num_boxes = dt_boxes.shape[0] # 获取文本框的数量
    # 主要排序：按左上角点的Y坐标（dt_boxes[i][0][1]）升序，Y坐标相同则按X坐标（dt_boxes[i][0][0]）升序
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes) # 转换为列表方便后续调整
    # 排序微调：处理同一行内（Y坐标相近）但X坐标顺序错误的框
    for i in range(num_boxes - 1): # 遍历排序后的框
        # 从当前框向前检查，看是否需要与前面的框交换位置
        for j in range(i, -1, -1):
            # 如果相邻两个框的左上角Y坐标差小于10（视为同一行）
            # 并且 后一个框的X坐标 小于 前一个框的X坐标（顺序错误）
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                # 交换这两个框的位置
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                # 如果Y坐标差较大或X坐标顺序正确，则停止向前比较
                break
    return _boxes # 返回最终排序的文本框列表

def bbox_to_points(bbox):
    """ 
    将bbox格式 [x0, y0, x1, y1] 转换为四个顶点的数组。
    Args:
        bbox (list or tuple): 包含左上角和右下角坐标 [x0, y0, x1, y1]。
    Returns:
        np.ndarray: 包含四个顶点坐标的数组 [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]，数据类型为float32。
    """
    x0, y0, x1, y1 = bbox # 解包bbox坐标
    # 按顺序（左上、右上、右下、左下）构造顶点数组
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype('float32')

def points_to_bbox(points):
    """ 
    将四个顶点的数组转换为bbox格式 [x0, y0, x1, y1]。
    假设输入的点是按顺序 [左上, 右上, 右下, 左下] 排列的。
    Args:
        points (np.ndarray or list): 包含四个顶点坐标的数组或列表。
    Returns:
        list: bbox格式的坐标 [x0, y0, x1, y1]。
    """
    x0, y0 = points[0] # 左上角的坐标
    x1, _ = points[1] # 右上角的x坐标（即x_max）
    _, y1 = points[2] # 右下角的y坐标（即y_max）
    return [x0, y0, x1, y1] # 返回bbox列表

def merge_intervals(intervals):
    """
    合并一维区间列表中重叠的区间。
    Args:
        intervals (list): 区间列表，每个区间为 [start, end]。
    Returns:
        list: 合并后的区间列表。
    """
    # 根据区间的起始值对区间进行排序
    intervals.sort(key=lambda x: x[0])
    merged = [] # 初始化合并后的区间列表
    for interval in intervals: # 遍历排序后的区间
        # 如果合并列表为空，或者当前区间与上一个合并区间的结束点不重叠
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval[:]) # 直接添加当前区间（使用切片创建副本）
        else:
            # 否则，存在重叠，将当前区间与上一个合并区间合并
            # 更新上一个合并区间的结束点为两者结束点的最大值
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged # 返回合并后的区间列表

def remove_intervals(original, masks):
    """
    从一个原始区间中移除一组掩码区间。
    Args:
        original (list): 原始区间 [start, end]。
        masks (list): 需要移除的掩码区间列表，每个区间为 [start, end]。
    Returns:
        list: 移除掩码区间后剩余的区间列表。
    """
    # 首先合并所有掩码区间，简化处理
    merged_masks = merge_intervals(masks)
    result = [] # 初始化结果列表，存储剩余的区间
    original_start, original_end = original # 获取原始区间的起始和结束
    current_start = original_start # 追踪当前处理的原始区间的起始点
    
    for mask in merged_masks: # 遍历合并后的掩码区间
        mask_start, mask_end = mask # 获取掩码区间的起始和结束
        
        # 如果掩码区间在当前处理的原始区间部分之前，则跳过
        if mask_end < current_start:
            continue
        # 如果掩码区间在当前处理的原始区间部分之后，则跳过（理论上排序后不会发生，但加上无妨）
        if mask_start > original_end:
            continue

        # 如果当前处理的原始区间起始点在掩码区间之前，说明有一段未被覆盖
        if current_start < mask_start:
            # 将[current_start, mask_start - 1]这一段添加到结果中
            result.append([current_start, mask_start - 1])
            
        # 更新下一次开始处理的原始区间起始点，跳过掩码区域
        # 取掩码结束点+1 和 当前起始点 中的较大值，防止掩码重叠导致回退
        current_start = max(current_start, mask_end + 1)
        
        # 如果 current_start 已经超出了 original_end，说明原始区间已被完全处理或覆盖
        if current_start > original_end:
            break

    # 循环结束后，如果 current_start 仍然小于等于 original_end，说明原始区间的末尾还有剩余部分
    if current_start <= original_end:
        result.append([current_start, original_end]) # 将最后剩余的部分添加到结果中
        
    return result # 返回最终剩余的区间列表


def update_det_boxes(dt_boxes, mfd_res):
    """
    更新检测到的文本框 (`dt_boxes`)，移除与 `mfd_res` 中元素（如公式、图表）在水平方向上重叠的部分。
    保留有角度的文本框不作处理。
    Args:
        dt_boxes (list): 检测到的文本框列表，每个元素是 [4, 2] 的角点坐标数组。
        mfd_res (list): 数学公式/图表检测结果列表，每个元素是包含 'bbox' [x0, y0, x1, y1] 的字典。
    Returns:
        list: 更新后的文本框列表，可能包含被分割的文本框。
    """
    new_dt_boxes = [] # 初始化更新后的文本框列表
    angle_boxes_list = [] # 初始化存储有角度文本框的列表
    
    for text_box in dt_boxes: # 遍历每个检测到的文本框
        # 检查文本框是否有角度
        if calculate_is_angle(text_box):
            angle_boxes_list.append(text_box) # 如果有角度，直接添加到angle_boxes_list，不处理
            continue # 处理下一个文本框
            
        # 将文本框的角点坐标转换为 bbox [x0, y0, x1, y1] 格式
        text_bbox = points_to_bbox(text_box)
        masks_list = [] # 初始化当前文本框需要移除的水平区间列表
        
        for mf_box in mfd_res: # 遍历数学公式/图表检测结果
            mf_bbox = mf_box['bbox'] # 获取公式/图表的bbox
            # 检查文本框和公式/图表框在Y轴上是否有显著重叠
            if __is_overlaps_y_exceeds_threshold(text_bbox, mf_bbox):
                # 如果Y轴重叠，则将公式/图表框的水平区间 [x_start, x_end] 添加到掩码列表
                masks_list.append([mf_bbox[0], mf_bbox[2]])
                
        # 获取文本框的原始水平区间 [x_start, x_end]
        text_x_range = [text_bbox[0], text_bbox[2]]
        # 从文本框的水平区间中移除所有重叠的掩码区间
        text_remove_mask_range = remove_intervals(text_x_range, masks_list)
        
        temp_dt_box = [] # 临时存储由剩余水平区间生成的新文本框
        # 遍历移除掩码后剩余的水平区间段
        for text_remove_mask in text_remove_mask_range:
            # 使用剩余的水平区间 [new_x_start, new_x_end] 和原始文本框的垂直区间 [y_start, y_end] 创建新的bbox
            new_bbox = [text_remove_mask[0], text_bbox[1], text_remove_mask[1], text_bbox[3]]
            # 将新的bbox转换为角点坐标格式，并添加到临时列表
            temp_dt_box.append(bbox_to_points(new_bbox))
            
        # 如果生成了新的（或未改变的）文本框段
        if len(temp_dt_box) > 0:
            # 将这些新文本框段添加到最终结果列表中
            new_dt_boxes.extend(temp_dt_box)
            
    # 将之前收集的有角度的文本框也添加到最终结果列表中
    new_dt_boxes.extend(angle_boxes_list)
    return new_dt_boxes # 返回更新后的文本框列表

def merge_overlapping_spans(spans):
    """
    合并在同一行上重叠的span（边界框）。
    假设输入的spans大致在同一水平线上。
    :param spans: 一个span坐标列表 [(x1, y1, x2, y2), ...]
    :return: 合并后的span列表
    """
    # 如果输入span列表为空，则返回空列表
    if not spans:
        return []
    # 按起始x坐标对span进行排序
    spans.sort(key=lambda x: x[0])
    # 初始化合并后的span列表
    merged = []
    for span in spans:
        # 解包span坐标
        x1, y1, x2, y2 = span
        # 如果合并列表为空，或者当前span与最后一个合并的span在水平方向上没有重叠（即当前x1大于上一个x2）
        if not merged or merged[-1][2] < x1:
            # 直接将当前span添加到合并列表
            merged.append(list(span)) # 使用list创建副本
        else:
            # 如果存在水平重叠，则合并当前span和上一个span
            last_span = merged.pop() # 弹出最后一个合并的span
            # 计算合并后span的坐标：取最小的(x1, y1)作为左上角，最大的(x2, y2)作为右下角（即外接矩形）
            merged_x1 = min(last_span[0], x1)
            merged_y1 = min(last_span[1], y1)
            merged_x2 = max(last_span[2], x2)
            merged_y2 = max(last_span[3], y2)
            # 将合并后的span添加回列表
            merged.append([merged_x1, merged_y1, merged_x2, merged_y2])
            
    # 将内部列表转换为元组（如果需要保持与原始注释一致的输出类型）
    merged_tuples = [tuple(span) for span in merged]
    # 返回合并后的span列表
    return merged_tuples


def merge_det_boxes(dt_boxes):
    """
    合并检测到的文本框。
    此函数接收一个检测到的边界框列表，每个框由四个角点定义。
    目标是将这些边界框合并成更大的文本区域（例如，文本行）。
    Parameters:
    dt_boxes (list): 包含多个文本检测框的列表，每个框由四个角点定义 ([4, 2] 数组)。
    Returns:
    list: 包含合并后文本区域的列表，每个区域由四个角点定义 ([4, 2] 数组)。
    """
    # 初始化用于存储转换后字典格式的文本框列表和有角度文本框的列表
    dt_boxes_dict_list = []
    angle_boxes_list = []
    
    for text_box in dt_boxes: # 遍历输入的检测框
        # 检查文本框是否有角度
        if calculate_is_angle(text_box):
            angle_boxes_list.append(text_box) # 有角度的框单独存放，后续直接添加
            continue # 处理下一个框
            
        # 将无角度文本框的角点坐标转换为 bbox [x0, y0, x1, y1] 格式
        text_bbox = points_to_bbox(text_box)
        # 创建符合 merge_spans_to_line 函数输入格式的字典
        text_box_dict = {
            'bbox': text_bbox,
            'type': 'text', # 标记类型为文本
        }
        dt_boxes_dict_list.append(text_box_dict) # 添加到字典列表
        
    # 调用外部函数 merge_spans_to_line 将邻近的文本区域合并成行
    # 这个函数会根据文本框的位置和大小将它们分组到不同的行中
    lines = merge_spans_to_line(dt_boxes_dict_list)
    
    # 初始化一个新的列表，用于存储最终合并后的文本区域
    new_dt_boxes = []
    for line in lines: # 遍历合并后的每一行
        line_bbox_list = [] # 收集当前行内所有span的bbox
        for span in line: # 遍历行内的每个span（文本片段）
            line_bbox_list.append(span['bbox']) # 提取bbox
            
        # 对当前行内的bbox列表进行重叠合并，处理行内可能存在的水平重叠片段
        merged_spans = merge_overlapping_spans(line_bbox_list)
        
        # 将合并后的span（bbox格式）转换回角点坐标格式，并添加到新的检测框列表中
        for span in merged_spans: # span 现在是 (x1, y1, x2, y2) 格式
            new_dt_boxes.append(bbox_to_points(list(span))) # 转换格式并添加
            
    # 将之前保留的有角度的文本框添加到最终结果列表中
    new_dt_boxes.extend(angle_boxes_list)
    return new_dt_boxes # 返回合并后的文本框列表

def get_adjusted_mfdetrec_res(single_page_mfdetrec_res, useful_list):
    """
    根据裁剪和粘贴参数调整单页数学公式检测结果 (`single_page_mfdetrec_res`) 的坐标。
    同时过滤掉调整后位于新画布外的结果。
    Args:
        single_page_mfdetrec_res (list): 单页的数学公式检测结果列表，每个元素是包含 'bbox' 的字典。
        useful_list (list): 包含坐标调整所需参数的列表：
                            [paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height]
                            paste_x, paste_y: 裁剪区域粘贴到新画布的左上角坐标。
                            xmin, ymin, xmax, ymax: 在原始页面上裁剪区域的bbox。
                            new_width, new_height: 新画布的尺寸。
    Returns:
        list: 调整坐标并过滤后的数学公式检测结果列表。
    """
    # 解包useful_list中的参数
    paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
    # 初始化调整后的结果列表
    adjusted_mfdetrec_res = []
    for mf_res in single_page_mfdetrec_res: # 遍历每个公式检测结果
        mf_xmin, mf_ymin, mf_xmax, mf_ymax = mf_res["bbox"] # 获取原始bbox坐标
        
        # 坐标转换：
        # 1. 减去裁剪区域的左上角坐标 (xmin, ymin)，得到相对于裁剪区域的坐标。
        # 2. 加上粘贴位置的坐标 (paste_x, paste_y)，得到在新画布上的坐标。
        x0 = mf_xmin - xmin + paste_x
        y0 = mf_ymin - ymin + paste_y
        x1 = mf_xmax - xmin + paste_x
        y1 = mf_ymax - ymin + paste_y
        
        # 过滤掉完全在新画布边界之外的公式框
        # 条件：右下角坐标小于0，或者左上角坐标大于新画布尺寸
        if any([x1 < 0, y1 < 0]) or any([x0 > new_width, y0 > new_height]):
            continue # 跳过这个结果
        else:
            # 如果公式框部分或全部在新画布内，则添加到调整后的结果列表
            adjusted_mfdetrec_res.append({
                "bbox": [x0, y0, x1, y1], # 使用调整后的坐标
            })
    return adjusted_mfdetrec_res # 返回调整后的结果列表

def get_ocr_result_list(ocr_res, useful_list, ocr_enable, new_image, lang):
    """
    处理OCR结果列表，调整坐标，根据需要进行图像裁剪，过滤低置信度结果，并格式化输出。
    坐标调整是将新画布上的坐标转换回原始页面的坐标。
    Args:
        ocr_res (list): OCR引擎返回的结果列表。每个元素可能包含 [角点坐标, [文本, 置信度]] 或仅 [角点坐标]。
        useful_list (list): 包含坐标调整所需参数的列表（同上）。
        ocr_enable (bool): 是否启用OCR识别（如果为True且ocr_res只提供框，会尝试裁剪图像）。
        new_image (np.ndarray): 经过裁剪和粘贴操作后的新画布图像。
        lang (str): 文本的语言标识。
    Returns:
        list: 格式化后的OCR结果列表，包含调整后的坐标、文本、置信度等信息。
    """
    # 解包useful_list中的参数
    paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
    ocr_result_list = [] # 初始化最终的OCR结果列表
    ori_im = new_image.copy() # 复制一份新画布图像，用于后续可能的裁剪操作

    for box_ocr_res in ocr_res: # 遍历OCR引擎的原始结果
        text = "" # 初始化文本
        score = 1.0 # 初始化置信度，默认为1
        img_crop = None # 初始化裁剪后的图像块，默认为None
        
        # 检查原始结果的格式
        if len(box_ocr_res) == 2: # 格式为 [points, [text, score]]
            p1, p2, p3, p4 = box_ocr_res[0] # 获取角点坐标
            text, score = box_ocr_res[1] # 获取文本和置信度
            # logger.info(f"text: {text}, score: {score}") # 日志记录（原始代码中被注释）
            if score < 0.6:  # 过滤掉置信度低于0.6的结果
                continue # 跳过此结果
        else: # 格式为 [points] (只有检测框，没有识别结果)
            p1, p2, p3, p4 = box_ocr_res # 获取角点坐标
            text, score = "", 1 # 文本设为空，置信度设为1
            # 如果启用了OCR，并且当前结果只有框
            if ocr_enable: 
                # 准备从 new_image 中裁剪出对应的图像区域
                tmp_box = copy.deepcopy(np.array([p1, p2, p3, p4]).astype('float32')) # 深拷贝并转换类型
                img_crop = get_rotate_crop_image(ori_im, tmp_box) # 调用函数进行旋转和裁剪

        # 检查文本框是否有角度
        poly = [p1, p2, p3, p4] # 组成多边形点列表
        if calculate_is_angle(poly): # 如果检测到角度
            # logger.info(f"average_angle_degrees: {average_angle_degrees}, text: {text}") # 日志记录（原始代码中被注释）
            # 注释：与x轴的夹角超过阈值（由calculate_is_angle判断），对边界做矫正
            # 矫正方法：计算几何中心，然后根据平均宽度和高度重新生成一个水平的矩形框
            x_center = sum(point[0] for point in poly) / 4 # 计算中心x坐标
            y_center = sum(point[1] for point in poly) / 4 # 计算中心y坐标
            # 计算平均高度（左边高+右边高）/ 2
            avg_height = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2 
            # 近似计算宽度（假设近似平行四边形，用上下边中点距离或直接用右上角x - 左上角x）
            # 原始代码使用 p3[0] - p1[0]，这对于旋转框可能不准确，但这里遵循原始逻辑
            avg_width = p3[0] - p1[0]  # 或者更稳健的方法是计算边的长度
            # 重新计算四个角点，使其成为一个以(x_center, y_center)为中心，宽高为avg_width, avg_height的水平矩形
            p1 = [x_center - avg_width / 2, y_center - avg_height / 2]
            p2 = [x_center + avg_width / 2, y_center - avg_height / 2]
            p3 = [x_center + avg_width / 2, y_center + avg_height / 2]
            p4 = [x_center - avg_width / 2, y_center + avg_height / 2]
            
        # 坐标转换：将新画布上的坐标（可能是矫正后的）转换回原始页面的坐标系
        # 1. 减去粘贴位置的坐标 (paste_x, paste_y)，得到相对于裁剪区域的坐标。
        # 2. 加上裁剪区域在原始页面的左上角坐标 (xmin, ymin)，得到在原始页面上的坐标。
        p1 = [p1[0] - paste_x + xmin, p1[1] - paste_y + ymin]
        p2 = [p2[0] - paste_x + xmin, p2[1] - paste_y + ymin]
        p3 = [p3[0] - paste_x + xmin, p3[1] - paste_y + ymin]
        p4 = [p4[0] - paste_x + xmin, p4[1] - paste_y + ymin]
        
        # 根据是否启用了OCR（并实际进行了裁剪）来构建结果字典
        if ocr_enable and img_crop is not None: # 如果需要OCR且裁剪了图像
            ocr_result_list.append({
                'category_id': 15, # 类别ID（硬编码为15）
                'poly': p1 + p2 + p3 + p4, # 调整后的8个坐标值列表
                'score': 1, # 置信度设为1（因为是后续识别）
                'text': text, # 文本（如果之前就有，否则为空）
                'np_img': img_crop, # 裁剪出的NumPy图像数组
                'lang': lang, # 语言标识
            })
        else: # 如果不需要OCR或已有识别结果
             ocr_result_list.append({
                'category_id': 15, # 类别ID
                'poly': p1 + p2 + p3 + p4, # 调整后的8个坐标值列表
                'score': float(round(score, 2)), # 保留两位小数的置信度
                'text': text, # 识别的文本
            })
            
    return ocr_result_list # 返回处理和格式化后的OCR结果列表

def calculate_is_angle(poly):
    """
    判断一个四边形（文本框）是否为有角度（非水平或垂直）。
    通过比较对角线顶点的垂直距离与平均边高的差异来判断。
    Args:
        poly (list): 包含四个顶点坐标的列表 [p1, p2, p3, p4]，假设顺序为左上、右上、右下、左下。
    Returns:
        bool: 如果判定为有角度，返回True；否则返回False。
    """
    p1, p2, p3, p4 = poly # 解包四个顶点坐标
    # 计算平均高度 = (左边高度 + 右边高度) / 2
    height = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
    # 检查对角线顶点 (p1, p3) 的垂直距离是否在平均高度的某个容差范围 (0.8 到 1.2 倍) 内
    if 0.8 * height <= (p3[1] - p1[1]) <= 1.2 * height:
        # 如果在该范围内，认为矩形接近水平或垂直，返回False
        return False
    else:
        # 如果超出该范围，认为矩形有明显角度，返回True
        # logger.info((p3[1] - p1[1])/height) # 日志记录（原始代码中被注释）
        return True

def get_rotate_crop_image(img, points):
    '''
    从原图中根据指定的四个顶点 `points` 裁剪出对应的区域，并进行透视变换矫正。
    如果裁剪出的图像高宽比过大（过高），则将其旋转90度。
    '''
    # 原始代码中的注释块，提供了另一种简单的裁剪方式（轴对齐裁剪），但当前函数实现的是透视变换裁剪
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2" # 确保输入是4个点
    
    # 计算目标矫正后图像的宽度：取上边和下边长度的最大值
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), # 上边长度
            np.linalg.norm(points[2] - points[3])  # 下边长度
            ))
    # 计算目标矫正后图像的高度：取左边和右边长度的最大值
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), # 左边长度
            np.linalg.norm(points[1] - points[2])  # 右边长度
            ))
    
    # 定义目标矩形的四个顶点坐标（顺序：左上、右上、右下、左下）
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
                          
    # 计算透视变换矩阵 M，将输入的 points 映射到目标矩形 pts_std
    M = cv2.getPerspectiveTransform(points, pts_std)
    
    # 应用透视变换
    dst_img = cv2.warpPerspective(
        img, # 输入图像
        M, # 变换矩阵
        (img_crop_width, img_crop_height), # 输出图像的尺寸
        borderMode=cv2.BORDER_REPLICATE, # 边界处理方式：复制边界像素
        flags=cv2.INTER_CUBIC # 插值方法：三次样条插值
        )
        
    # 获取裁剪并矫正后的图像尺寸
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    
    # 检查矫正后图像的高宽比，如果过高（例如竖排文本）
    if dst_img_width > 0 and dst_img_height * 1.0 / dst_img_width >= 1.5:
        # 将图像逆时针旋转90度
        dst_img = np.rot90(dst_img)
        
    return dst_img # 返回最终处理后的图像块