# 导入 copy 模块，用于对象复制，特别是深拷贝
import copy
# 导入 math 模块，提供数学运算功能，如三角函数
import math
# 导入 os 模块，用于与操作系统交互，如文件路径操作、环境变量设置
import os
# 导入 re 模块，用于正则表达式操作
import re
# 导入 statistics 模块，提供统计计算功能，如中位数
import statistics
# 导入 time 模块，用于时间相关操作，如计时
import time
# 导入 warnings 模块，用于控制警告信息的显示
import warnings
# 从 typing 模块导入 List 类型提示，用于明确列表类型
from typing import List

# 导入 cv2 模块 (OpenCV)，用于图像处理
import cv2
# 导入 fitz 模块 (PyMuPDF)，用于处理 PDF 文件
import fitz
# 导入 torch 模块 (PyTorch)，用于深度学习模型操作
import torch
# 导入 numpy 模块，用于数值计算，特别是数组操作
import numpy as np
# 导入 loguru 模块，用于日志记录
from loguru import logger
# 导入 tqdm 模块，用于显示进度条
from tqdm import tqdm

# 从项目配置中导入支持的 PDF 解析方法枚举
from magic_pdf.config.enums import SupportedPdfParseMethod
# 从项目配置中导入内容类型枚举，如块类型、内容类型
from magic_pdf.config.ocr_content_type import BlockType, ContentType
# 从项目中导入数据集和页面数据类
from magic_pdf.data.dataset import Dataset, PageableData
# 从项目库中导入边界框相关的工具函数
from magic_pdf.libs.boxbase import calculate_overlap_area_in_bbox1_area_ratio, __is_overlaps_y_exceeds_threshold
# 从项目库中导入内存清理函数
from magic_pdf.libs.clean_memory import clean_memory
# 从项目库中导入配置读取函数
from magic_pdf.libs.config_reader import get_local_layoutreader_model_dir, get_llm_aided_config, get_device
# 从项目库中导入字典转列表的工具函数
from magic_pdf.libs.convert_utils import dict_to_list
# 从项目库中导入计算 MD5 哈希值的工具函数
from magic_pdf.libs.hash_utils import compute_md5
# 从项目库中导入 PDF 图像处理工具函数
from magic_pdf.libs.pdf_image_tools import cut_image_to_pil_image
# 从项目中导入主要的模型处理类
from magic_pdf.model.magic_model import MagicModel
# 从项目后处理模块中导入 LLM 辅助处理函数 (公式、文本、标题)
from magic_pdf.post_proc.llm_aided import llm_aided_formula, llm_aided_text, llm_aided_title
# 从模型子模块中导入原子模型单例管理器
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
# 从项目后处理模块中导入段落切分函数
from magic_pdf.post_proc.para_split_v3 import para_split
# 从项目预处理模块中导入页面字典构建函数
from magic_pdf.pre_proc.construct_page_dict import ocr_construct_page_component_v2
# 从项目预处理模块中导入图像和表格裁剪函数
from magic_pdf.pre_proc.cut_image import ocr_cut_image_and_table
# 从项目预处理模块中导入准备用于布局分割的边界框函数
from magic_pdf.pre_proc.ocr_detect_all_bboxes import ocr_prepare_bboxes_for_layout_split_v2
# 从项目预处理模块中导入 OCR 字典合并相关的函数
from magic_pdf.pre_proc.ocr_dict_merge import fill_spans_in_blocks, fix_block_spans_v2, fix_discarded_block
# 从项目预处理模块中导入 OCR Span 列表修改相关的函数
from magic_pdf.pre_proc.ocr_span_list_modify import get_qa_need_list_v2, remove_overlaps_low_confidence_spans, \
    remove_overlaps_min_spans, remove_x_overlapping_chars

# 设置环境变量，禁止 albumentations 库检查更新
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新

# 定义一个函数，用于替换文本中的特殊字符 STX (\u0002) 和 ETX (\u0003)
def __replace_STX_ETX(text_str: str):
    """替换 \u0002 和 \u0003，因为使用 pymupdf 提取时这些字符会变成乱码。实际上它们原本是引号。
    缺点：这个问题目前只在英文文本中观察到；中文文本中尚未发现。
        Args:
            text_str (str): 原始文本
        Returns:
            _type_: 替换后的文本
    """  # noqa: E501 # 忽略 E501 错误 (行太长)
    # 如果输入字符串非空
    if text_str:
        # 将 \u0002 替换为单引号
        s = text_str.replace('\u0002', "'")
        # 将 \u0003 替换为单引号
        s = s.replace('\u0003', "'")
        # 返回替换后的字符串
        return s
    # 如果输入字符串为空，则直接返回
    return text_str

# 定义一个函数，用于拆分文本中的连写字符 (ligatures)
def __replace_ligatures(text: str):
    # 定义常见的连写字符及其对应的拆分形式
    ligatures = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬅ': 'ft', 'ﬆ': 'st'
    }
    # 使用正则表达式一次性替换所有定义的连写字符
    # re.escape 用于转义连写字符中的特殊正则字符
    # lambda m: ligatures[m.group()] 用于根据匹配到的连写字符返回其对应的拆分形式
    return re.sub('|'.join(map(re.escape, ligatures.keys())), lambda m: ligatures[m.group()], text)

# 定义一个函数，根据 span 内的字符信息 (chars) 生成内容 (content)
def chars_to_content(span):
    # 检查 span 中的 'chars' 列表是否为空
    if len(span['chars']) == 0:
        # 如果为空，则不做任何处理
        pass
    else:
        # 先按照字符边界框 (bbox) 的中心点 x 坐标对 'chars' 列表进行排序
        span['chars'] = sorted(span['chars'], key=lambda x: (x['bbox'][0] + x['bbox'][2]) / 2)
        # 计算每个字符的宽度 (x1 - x0)
        char_widths = [char['bbox'][2] - char['bbox'][0] for char in span['chars']]
        # 计算字符宽度的中位数
        median_width = statistics.median(char_widths)
        # 根据 x 轴重叠比率移除一部分可能错误的字符
        span = remove_x_overlapping_chars(span, median_width)
        # 初始化内容字符串
        content = ''
        # 遍历排序和清理后的字符列表
        for i, char in enumerate(span['chars']):
            # 获取当前字符
            char1 = char
            # 获取下一个字符，如果存在的话
            char2 = span['chars'][i + 1] if i + 1 < len(span['chars']) else None
            # 判断是否需要在字符间插入空格
            # 条件：存在下一个字符，且下一个字符的左边界 (x0) 与当前字符的右边界 (x1) 距离大于 0.25 倍字符宽度中位数，
            # 并且当前字符和下一个字符都不是空格
            if char2 and char2['bbox'][0] - char1['bbox'][2] > median_width * 0.25 and char['c'] != ' ' and char2['c'] != ' ':
                # 如果满足条件，则在当前字符后添加一个空格
                content += f"{char['c']} "
            else:
                # 否则，直接添加当前字符
                content += char['c']
        # 对生成的 content 应用连写字符替换
        span['content'] = __replace_ligatures(content)
    # 从 span 字典中删除 'chars' 键，因为它已经被处理并合并到 'content' 中
    del span['chars']

# 定义行尾标点符号集合
LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；', ']', '】', '}', '}', '>', '》', '、', ',', '，', '-', '—', '–',)
# 定义行首标点符号集合
LINE_START_FLAG = ('(', '（', '"', '“', '【', '{', '《', '<', '「', '『', '【', '[',)

# 定义一个函数，将页面上的所有字符 (all_chars) 填充到对应的 span 中
def fill_char_in_spans(spans, all_chars):
    # 简单地按照 span 的上边界 (y0) 从上到下进行排序
    spans = sorted(spans, key=lambda x: x['bbox'][1])
    # 遍历页面上的所有字符
    for char in all_chars:
        # 遍历排序后的所有 span
        for span in spans:
            # 判断当前字符是否属于当前 span (使用 calculate_char_in_span 函数)
            if calculate_char_in_span(char['bbox'], span['bbox'], char['c']):
                # 如果属于，则将该字符添加到 span 的 'chars' 列表中
                span['chars'].append(char)
                # 一个字符只属于一个 span，找到后跳出内层循环，处理下一个字符
                break
    # 初始化一个列表，用于存储可能需要 OCR 的 span
    need_ocr_spans = []
    # 再次遍历所有 span
    for span in spans:
        # 调用 chars_to_content 函数，根据 'chars' 生成 'content'
        chars_to_content(span)
        # 检查 span 是否可能为空或无效
        # 条件：span 的 content 长度乘以高度 小于 宽度乘以 0.5 (经验性判断，可能表示 span 很宽但内容很少或没有)
        if len(span['content']) * span['height'] < span['width'] * 0.5:
            # 如果满足条件，则认为这个 span 可能需要 OCR 补充内容
            # logger.info(f"maybe empty span: {len(span['content'])}, {span['height']}, {span['width']}") # 记录日志信息 (已注释)
            need_ocr_spans.append(span)
        # 删除临时的 'height' 和 'width' 键
        del span['height'], span['width']
    # 返回需要 OCR 的 span 列表
    return need_ocr_spans

# 定义一个函数，判断一个字符 (char) 是否属于一个 span，使用更鲁棒的中心点坐标判断
def calculate_char_in_span(char_bbox, span_bbox, char, span_height_radio=0.33):
    # 计算字符边界框的中心点 x 坐标
    char_center_x = (char_bbox[0] + char_bbox[2]) / 2
    # 计算字符边界框的中心点 y 坐标
    char_center_y = (char_bbox[1] + char_bbox[3]) / 2
    # 计算 span 边界框的中心点 y 坐标
    span_center_y = (span_bbox[1] + span_bbox[3]) / 2
    # 计算 span 的高度
    span_height = span_bbox[3] - span_bbox[1]
    # 主要判断逻辑：
    # 1. 字符中心 x 坐标在 span 的 x 范围内
    # 2. 字符中心 y 坐标在 span 的 y 范围内
    # 3. 字符中心 y 坐标与 span 中心 y 坐标的绝对差值小于 span 高度的 span_height_radio 倍 (默认为 1/3)
    if (
        span_bbox[0] < char_center_x < span_bbox[2]
        and span_bbox[1] < char_center_y < span_bbox[3]
        and abs(char_center_y - span_center_y) < span_height * span_height_radio  # 字符的中轴和span的中轴高度差不能超过指定比例的span高度
    ):
        # 如果满足以上条件，则认为字符属于该 span
        return True
    else:
        # 特殊处理：如果字符是行尾标点符号
        if char in LINE_STOP_FLAG:
            # 放宽判定条件：
            # 1. 字符的左边界 (x0) 在 span 右边界附近 (span_bbox[2] - span_height 到 span_bbox[2] 之间)
            # 2. 字符中心 x 坐标仍在 span 的 x 范围内
            # 3. 字符中心 y 坐标仍在 span 的 y 范围内
            # 4. y 轴中心点高度差限制同上
            # 目的是给靠近行尾的标点符号一个进入 span 的机会
            if (
                (span_bbox[2] - span_height) < char_bbox[0] < span_bbox[2] # 字符左边界在 span 右侧附近
                and char_center_x > span_bbox[0] # 字符中心仍在 span 内
                and span_bbox[1] < char_center_y < span_bbox[3] # y 轴中心在 span 内
                and abs(char_center_y - span_center_y) < span_height * span_height_radio # 高度差限制
            ):
                return True
        # 特殊处理：如果字符是行首标点符号
        elif char in LINE_START_FLAG:
            # 放宽判定条件：
            # 1. 字符的右边界 (x1) 在 span 左边界附近 (span_bbox[0] 到 span_bbox[0] + span_height 之间)
            # 2. 字符中心 x 坐标仍在 span 的 x 范围内
            # 3. 字符中心 y 坐标仍在 span 的 y 范围内
            # 4. y 轴中心点高度差限制同上
            # 目的是给靠近行首的标点符号一个进入 span 的机会
            if (
                span_bbox[0] < char_bbox[2] < (span_bbox[0] + span_height) # 字符右边界在 span 左侧附近
                and char_center_x < span_bbox[2] # 字符中心仍在 span 内
                and span_bbox[1] < char_center_y < span_bbox[3] # y 轴中心在 span 内
                and abs(char_center_y - span_center_y) < span_height * span_height_radio # 高度差限制
            ):
                return True
        else:
            # 其他情况，字符不属于该 span
            return False

# 定义一个函数，移除文本块 (text_blocks) 中倾斜角度过大的行 (line)
def remove_tilted_line(text_blocks):
    # 遍历所有文本块
    for block in text_blocks:
        # 初始化一个列表，用于存储需要移除的行
        remove_lines = []
        # 遍历当前块中的所有行
        for line in block['lines']:
            # 获取行的方向向量 (cosine, sine)
            cosine, sine = line['dir']
            # 计算方向向量对应的弧度值
            angle_radians = math.atan2(sine, cosine)
            # 将弧度值转换为角度值
            angle_degrees = math.degrees(angle_radians)
            # 如果角度的绝对值在 2 到 88 度之间 (即非水平或垂直)，则标记为需要移除
            if 2 < abs(angle_degrees) < 88:
                remove_lines.append(line)
        # 遍历标记为需要移除的行列表
        for line in remove_lines:
            # 从当前块的 'lines' 列表中移除该行
            block['lines'].remove(line)

# 定义一个函数，计算图像的对比度
def calculate_contrast(img, img_mode) -> float:
    """
    计算给定图像的对比度。
    :param img: 图像，类型为 numpy.ndarray
    :Param img_mode: 图像的色彩通道，'rgb' 或 'bgr'
    :return: 图像的对比度值
    """
    # 根据图像模式转换颜色空间
    if img_mode == 'rgb':
        # 将 RGB 图像转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img_mode == 'bgr':
        # 将 BGR 图像转换为灰度图 (OpenCV 默认)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # 如果模式无效，则抛出 ValueError 异常
        raise ValueError("Invalid image mode. Please provide 'rgb' or 'bgr'.")
    # 计算灰度图像的像素均值
    mean_value = np.mean(gray_img)
    # 计算灰度图像的像素标准差
    std_dev = np.std(gray_img)
    # 对比度定义为标准差除以平均值 (加上一个小的 epsilon 防止除零错误)
    contrast = std_dev / (mean_value + 1e-6)
    # logger.debug(f"contrast: {contrast}") # 记录对比度日志 (已注释)
    # 返回对比度值，保留两位小数
    return round(contrast, 2)

# 定义核心的文本 span 提取函数 (版本 2)
# @measure_time # 装饰器，用于测量函数执行时间 (已注释)
def txt_spans_extract_v2(pdf_page, spans, all_bboxes, all_discarded_blocks, lang):
    # --- 从 PyMuPDF 获取原始文本块信息 ---
    # 不同的 flags 会影响提取结果，如是否保留空白、是否处理连字符等
    # flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP # 保留空白，使用媒体框裁剪，cid 用 0xfffd 表示，连字符拆开
    # flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP # 保留连字符，保留空白，使用媒体框裁剪，cid 用 0xfffd 表示
    # flags=fitz.TEXTFLAGS_TEXT # 使用默认文本提取标志，发现自定义 flags 出现较多 0xfffd，可能是 pymupdf 可以自行处理内置字典的 pdf
    text_blocks_raw = pdf_page.get_text('rawdict', flags=fitz.TEXTFLAGS_TEXT)['blocks']
    # 移除原始文本块中倾斜的行
    remove_tilted_line(text_blocks_raw)

    # --- 提取所有非倾斜行的字符信息 ---
    all_pymu_chars = []
    for block in text_blocks_raw:
        for line in block['lines']:
            cosine, sine = line['dir'] # 获取行方向
            # 跳过非水平的行 (cosine 接近 1 或 -1，sine 接近 0)
            if abs(cosine) < 0.9 or abs(sine) > 0.1:
                continue
            # 遍历行内的 span
            for span in line['spans']:
                # 将 span 内的字符添加到总字符列表中
                all_pymu_chars.extend(span['chars'])

    # --- 计算页面 span 高度的中位数，用于后续判断 ---
    span_height_list = []
    for span in spans:
        # 跳过行内公式、图片、表格类型的 span
        if span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
            continue
        # 计算 span 高度并存储
        span_height = span['bbox'][3] - span['bbox'][1]
        span['height'] = span_height
        # 计算 span 宽度并存储
        span['width'] = span['bbox'][2] - span['bbox'][0]
        span_height_list.append(span_height)

    # 如果页面没有有效的 span 高度信息，则直接返回原始 spans
    if len(span_height_list) == 0:
        return spans
    else:
        # 计算 span 高度的中位数
        median_span_height = statistics.median(span_height_list)

    # --- 区分有用 span、无用 span 和垂直 span ---
    useful_spans = []       # 与有效 block 重叠的 span
    unuseful_spans = []     # 与丢弃 block 重叠的 span
    vertical_spans = []     # 垂直方向的 span (可能需要特殊处理)

    # 遍历所有 span
    for span in spans:
        # 跳过非文本类的 span
        if span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
            continue
        # 遍历所有检测到的边界框 (包括有效和丢弃的)
        for block in all_bboxes + all_discarded_blocks:
            # 跳过图片、表格、行间公式类型的 block
            if block[7] in [BlockType.ImageBody, BlockType.TableBody, BlockType.InterlineEquation]:
                continue
            # 计算 span 与 block 的重叠面积占 span 面积的比例
            if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block[0:4]) > 0.5:
                # 判断是否为垂直 span: 高度大于 3 倍中位数行高 且 高度大于 3 倍宽度
                if span['height'] > median_span_height * 3 and span['height'] > span['width'] * 3:
                    vertical_spans.append(span)
                # 如果与有效 block 重叠，则认为是 useful_spans
                elif block in all_bboxes:
                    useful_spans.append(span)
                # 如果与丢弃 block 重叠，则认为是 unuseful_spans
                else:
                    unuseful_spans.append(span)
                # 找到一个重叠的 block 就跳出内层循环
                break

    # --- 处理垂直 span ---
    # 垂直 span 通常跨越多行，直接用 PyMuPDF 提取的行 (line) 文本填充可能更准确
    if len(vertical_spans) > 0:
        # 使用 'dict' 格式获取文本块，包含更结构化的行信息
        text_blocks = pdf_page.get_text('dict', flags=fitz.TEXTFLAGS_TEXT)['blocks']
        all_pymu_lines = []
        # 收集页面上所有的行信息
        for block in text_blocks:
            for line in block['lines']:
                all_pymu_lines.append(line)
        # 遍历所有 PyMuPDF 提取的行
        for pymu_line in all_pymu_lines:
            # 遍历所有垂直 span
            for span in vertical_spans:
                # 如果行与垂直 span 重叠度高
                if calculate_overlap_area_in_bbox1_area_ratio(pymu_line['bbox'], span['bbox']) > 0.5:
                    # 将该行内所有 pymu_span 的文本拼接到垂直 span 的 content 中
                    for pymu_span in pymu_line['spans']:
                        span['content'] += pymu_span['text']
                    # 处理完一个垂直 span 就跳出内层循环
                    break
        # 移除内容为空的垂直 span
        for span in vertical_spans:
            if len(span['content']) == 0:
                spans.remove(span) # 注意：这里直接修改了传入的 spans 列表

    # --- 处理水平 span ---
    # 将 useful 和 unuseful 的 span 准备好，用于接收 PyMuPDF 提取的字符
    new_spans = []
    for span in useful_spans + unuseful_spans:
        # 只处理文本类型的 span
        if span['type'] in [ContentType.Text]:
            # 初始化 'chars' 列表，用于后续填充
            span['chars'] = []
            new_spans.append(span)

    # 调用 fill_char_in_spans 将 PyMuPDF 字符填充到 new_spans 中，并返回可能需要 OCR 的 span
    need_ocr_spans = fill_char_in_spans(new_spans, all_pymu_chars)

    # --- 对需要 OCR 的 span 进行处理 ---
    if len(need_ocr_spans) > 0:
        # # 初始化 OCR 模型 (注释掉了实际的模型加载，改为在后面批量处理)
        # atom_model_manager = AtomModelSingleton()
        # ocr_model = atom_model_manager.get_atom_model(
        #     atom_model_name='ocr',
        #     ocr_show_log=False,
        #     det_db_box_thresh=0.3,
        #     lang=lang
        # )
        # 遍历需要 OCR 的 span
        for span in need_ocr_spans:
            # 对 span 的 bbox 进行截图，得到 OpenCV 格式的图像 (BGR)
            span_img = cut_image_to_pil_image(span['bbox'], pdf_page, mode='cv2')
            # 计算截图的对比度
            # 如果对比度过低 (<= 0.17)，认为可能是空白或噪声区域，直接从 spans 列表中移除，不进行 OCR
            if calculate_contrast(span_img, img_mode='bgr') <= 0.17:
                spans.remove(span) # 注意：直接修改传入的 spans 列表
                continue
                # pass # 或者保留这个 span 但不进行 OCR

            # 准备进行 OCR (实际 OCR 调用被注释掉了，改为标记并收集图像)
            span['content'] = '' # 初始化 content
            span['score'] = 1 # 默认分数
            span['np_img'] = span_img # 将截图存储在 span 中，用于后续批量 OCR
            # # --- 以下是原先的单 span OCR 逻辑 (已注释) ---
            # ocr_res = ocr_model.ocr(span_img, det=False) # 只进行识别
            # # 如果 OCR 有结果
            # if ocr_res and len(ocr_res) > 0:
            #     # 如果结果列表的第一项不为空
            #     if len(ocr_res[0]) > 0:
            #         # 获取识别文本和分数
            #         ocr_text, ocr_score = ocr_res[0][0]
            #         # logger.info(f"ocr_text: {ocr_text}, ocr_score: {ocr_score}")
            #         # 如果分数大于 0.5 且文本非空
            #         if ocr_score > 0.5 and len(ocr_text) > 0:
            #             # 更新 span 的 content 和 score
            #             span['content'] = ocr_text
            #             span['score'] = float(round(ocr_score, 2))
            #         else:
            #             # 如果 OCR 结果不可信，则移除该 span
            #             spans.remove(span) # 注意：直接修改传入的 spans 列表
    # 返回处理后的 spans 列表
    return spans

# 定义模型初始化函数
def model_init(model_name: str):
    # 导入 LayoutLMv3 模型类
    from transformers import LayoutLMv3ForTokenClassification
    # 获取配置的设备名称 (如 "cuda:0", "cpu", "mps")
    device_name = get_device()
    # 检查设备是否支持 bfloat16
    bf_16_support = False
    if device_name.startswith("cuda"):
        bf_16_support = torch.cuda.is_bf16_supported()
    elif device_name.startswith("mps"): # Apple Silicon GPU
        bf_16_support = True # MPS 通常支持 bfloat16
    # 创建 torch 设备对象
    device = torch.device(device_name)

    # 根据模型名称加载模型
    if model_name == 'layoutreader':
        # 获取本地 layoutreader 模型的缓存目录
        layoutreader_model_dir = get_local_layoutreader_model_dir()
        # 如果本地缓存存在
        if os.path.exists(layoutreader_model_dir):
            # 从本地目录加载模型
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                layoutreader_model_dir
            )
        else:
            # 如果本地缓存不存在，记录警告并从 Hugging Face Hub 下载
            logger.warning(
                'local layoutreader model not exists, use online model from huggingface'
            )
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                'hantian/layoutreader' # 指定 Hugging Face 上的模型标识符
            )
        # 根据设备支持情况，将模型移到设备上，设置为评估模式 (eval)，并可能转换为 bfloat16
        if bf_16_support:
            model.to(device).eval().bfloat16()
        else:
            model.to(device).eval()
    else:
        # 如果模型名称不被支持，记录错误并退出程序
        logger.error('model name not allow')
        exit(1)
    # 返回初始化好的模型
    return model

# 定义模型单例类，用于管理和复用已加载的模型
class ModelSingleton:
    # 类变量，用于存储单例实例
    _instance = None
    # 类变量，用于存储已加载模型的字典 {model_name: model_object}
    _models = {}

    # 重写 __new__ 方法以实现单例模式
    def __new__(cls, *args, **kwargs): # 接受任意参数，但通常不使用
        # 如果实例尚未创建
        if cls._instance is None:
            # 创建实例
            cls._instance = super().__new__(cls)
        # 返回实例
        return cls._instance

    # 获取模型的方法
    def get_model(self, model_name: str):
        # 如果模型尚未加载
        if model_name not in self._models:
            # 调用 model_init 函数加载模型，并存储到字典中
            self._models[model_name] = model_init(model_name=model_name)
        # 返回已加载或新加载的模型
        return self._models[model_name]

# 定义执行模型预测的函数 (特指 LayoutReader 模型)
def do_predict(boxes: List[List[int]], model) -> List[int]:
    # 从 LayoutReader 的辅助模块导入所需函数
    from magic_pdf.model.sub_modules.reading_oreder.layoutreader.helpers import (
        boxes2inputs, parse_logits, prepare_inputs)
    # 忽略 transformers 库可能产生的 FutureWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        # 将输入的边界框列表转换为模型所需的输入格式
        inputs = boxes2inputs(boxes)
        # 进一步准备输入数据，适配模型
        inputs = prepare_inputs(inputs, model)
        # 使用模型进行推理，获取 logits (原始输出)，转移到 CPU 并移除批次维度
        logits = model(**inputs).logits.cpu().squeeze(0)
    # 解析 logits，得到排序后的索引列表
    return parse_logits(logits, len(boxes))

# 定义计算块 (block) 索引的函数
def cal_block_index(fix_blocks, sorted_bboxes):
    # 如果提供了基于模型排序的边界框列表 (sorted_bboxes)
    if sorted_bboxes is not None:
        # --- 使用 LayoutReader 排序结果 ---
        # 遍历修复后的块 (fix_blocks)
        for block in fix_blocks:
            # 初始化行索引列表
            line_index_list = []
            # 如果块没有行信息 (可能是纯图片或表格)
            if len(block['lines']) == 0:
                # 直接使用块自身的 bbox 在排序列表中的索引作为块索引
                block['index'] = sorted_bboxes.index(block['bbox'])
            else:
                # 如果块有行信息
                # 遍历块内的每一行
                for line in block['lines']:
                    # 获取行 bbox 在排序列表中的索引
                    line['index'] = sorted_bboxes.index(line['bbox'])
                    # 将行索引添加到列表中
                    line_index_list.append(line['index'])
                # 计算块内所有行索引的中位数
                median_value = statistics.median(line_index_list)
                # 使用行索引中位数作为块的索引
                block['index'] = median_value
            # 特殊处理：对于图、表、标题、行间公式的 Body 块
            if block['type'] in [BlockType.ImageBody, BlockType.TableBody, BlockType.Title, BlockType.InterlineEquation]:
                # 如果存在 'real_lines' (通常在 sort_lines_by_model 中添加)
                if 'real_lines' in block:
                    # 将当前的 'lines' (可能是虚拟生成的) 备份到 'virtual_lines'
                    block['virtual_lines'] = copy.deepcopy(block['lines'])
                    # 用 'real_lines' 覆盖 'lines'
                    block['lines'] = copy.deepcopy(block['real_lines'])
                    # 删除临时的 'real_lines'
                    del block['real_lines']
    else:
        # --- 如果没有提供模型排序结果，则使用 XY-Cut 排序 ---
        block_bboxes = []
        # 遍历修复后的块
        for block in fix_blocks:
            # 确保块的 bbox 坐标值不小于 0
            block['bbox'] = [max(0, x) for x in block['bbox']]
            # 将块的 bbox 添加到列表中
            block_bboxes.append(block['bbox'])
            # 特殊处理图表等 Body 块 (同上)
            if block['type'] in [BlockType.ImageBody, BlockType.TableBody, BlockType.Title, BlockType.InterlineEquation]:
                if 'real_lines' in block:
                    block['virtual_lines'] = copy.deepcopy(block['lines'])
                    block['lines'] = copy.deepcopy(block['real_lines'])
                    del block['real_lines']

        # 导入 XY-Cut 相关库
        import numpy as np
        from magic_pdf.model.sub_modules.reading_oreder.layoutreader.xycut import \
            recursive_xy_cut
        # 将块 bbox 列表转换为 NumPy 数组
        random_boxes = np.array(block_bboxes)
        # 打乱数组顺序 (XY-Cut 可能对输入顺序敏感或用于测试鲁棒性)
        np.random.shuffle(random_boxes)
        # 初始化结果列表
        res = []
        # 执行递归 XY-Cut 算法，res 存储排序后的索引
        recursive_xy_cut(np.asarray(random_boxes).astype(int), np.arange(len(block_bboxes)), res)
        # 断言确保 XY-Cut 返回的索引数量与块数量一致
        assert len(res) == len(block_bboxes)
        # 根据 XY-Cut 排序结果重新排列 bbox 列表
        sorted_boxes = random_boxes[np.array(res)].tolist()
        # 遍历修复后的块
        for i, block in enumerate(fix_blocks):
            # 将块 bbox 在 XY-Cut 排序结果中的索引赋给块的 'index'
            block['index'] = sorted_boxes.index(block['bbox'])

        # --- 生成行 (line) 索引 ---
        # 首先根据块索引对 fix_blocks 进行排序
        sorted_blocks = sorted(fix_blocks, key=lambda b: b['index'])
        # 初始化行索引计数器
        line_inedx = 1
        # 遍历排序后的块
        for block in sorted_blocks:
            # 遍历块内的每一行
            for line in block['lines']:
                # 分配递增的行索引
                line['index'] = line_inedx
                line_inedx += 1
    # 返回计算好索引的块列表
    return fix_blocks

# 定义一个函数，根据块的高度和页面尺寸，将一个块 (block) 的边界框在垂直方向上切分成多行
def insert_lines_into_block(block_bbox, line_height, page_w, page_h):
    # block_bbox 是一个元组 (x0, y0, x1, y1)，y0 是上边界，y1 是下边界 (与 PDF 坐标系可能不同，需确认)
    # 假设这里的坐标系是 y 向下增加，(x0, y0) 是左上角，(x1, y1) 是右下角
    x0, y0, x1, y1 = block_bbox
    # 计算块的高度
    block_height = y1 - y0
    # 计算块的宽度
    block_weight = x1 - x0

    # 如果块的高度大于等于 2 倍的估计行高
    if line_height * 2 < block_height:
        # 特殊情况：如果块很高 (超过页面高度 1/4) 且宽度在页面宽度的 1/4 到 1/2 之间 (可能是双列中的一列)
        if (
            block_height > page_h * 0.25 and page_w * 0.5 > block_weight > page_w * 0.25
        ):
            # 则按照估计行高来切分行数，切得细一点
            lines = int(block_height / line_height)
        else:
            # 其他情况：
            # 如果块很宽 (超过页面宽度 0.4)，可能包含复杂布局或横跨的图表，只切成 3 行，避免切太细
            if block_weight > page_w * 0.4:
                lines = 3
            # 如果块宽度在页面宽度 0.25 到 0.4 之间 (可能是三列结构)，也按行高切细点
            elif block_weight > page_w * 0.25:
                lines = int(block_height / line_height)
            else: # 块比较窄的情况
                # 判断长宽比
                if block_height / block_weight > 1.2: # 如果是细长的块，不切分或少切分 (这里返回原始 bbox，等于不切分)
                    return [[x0, y0, x1, y1]]
                else: # 如果不是细长的块，切分成 2 行
                    lines = 2
        # 确保至少切成 1 行
        lines = max(1, lines)
        # 计算切分后每行的高度
        line_height_new = (y1 - y0) / lines
        # 确定开始绘制线条的 y 位置
        current_y = y0
        # 用于存储切分后各行 bbox 的列表
        lines_positions = []
        # 循环生成每行的 bbox
        for i in range(lines):
            lines_positions.append([x0, current_y, x1, current_y + line_height_new])
            current_y += line_height_new
        # 返回切分后的行 bbox 列表
        return lines_positions
    else:
        # 如果块的高度小于 2 倍行高，则不切分，直接返回原始块的 bbox
        return [[x0, y0, x1, y1]]

# 定义一个函数，使用 LayoutReader 模型对页面中的所有行 (包括为图表等生成的虚拟行) 进行排序
def sort_lines_by_model(fix_blocks, page_w, page_h, line_height):
    # 初始化页面所有行的 bbox 列表
    page_line_list = []

    # 定义一个内部辅助函数，用于将块切分成行并添加到 page_line_list
    def add_lines_to_block(b):
        # 调用 insert_lines_into_block 将块 bbox 切分成多行 bbox
        line_bboxes = insert_lines_into_block(b['bbox'], line_height, page_w, page_h)
        # 初始化块的 'lines' 列表
        b['lines'] = []
        # 遍历切分后的行 bbox
        for line_bbox in line_bboxes:
            # 为每行创建一个字典，包含 bbox 和空的 spans 列表，并添加到块的 'lines' 中
            b['lines'].append({'bbox': line_bbox, 'spans': []})
        # 将切分后的行 bbox 添加到页面的总行列表 page_line_list 中
        page_line_list.extend(line_bboxes)

    # 遍历所有修复后的块
    for block in fix_blocks:
        # 如果是文本、标题、图注、表注等类型
        if block['type'] in [
            BlockType.Text, BlockType.Title,
            BlockType.ImageCaption, BlockType.ImageFootnote,
            BlockType.TableCaption, BlockType.TableFootnote
        ]:
            # 如果块没有行信息 (可能是空的或未提取到)
            if len(block['lines']) == 0:
                # 调用辅助函数，为这个块生成虚拟行
                add_lines_to_block(block)
            # 特殊情况：如果是标题块，且有多于 1 行，且块高度大于 2 倍行高 (可能需要重新切分排序)
            elif block['type'] in [BlockType.Title] and len(block['lines']) > 1 and (block['bbox'][3] - block['bbox'][1]) > line_height * 2:
                # 备份原始行信息到 'real_lines'
                block['real_lines'] = copy.deepcopy(block['lines'])
                # 调用辅助函数，重新生成虚拟行用于排序
                add_lines_to_block(block)
            else:
                # 如果是普通文本块或不需要重切的标题块，直接将现有行的 bbox 添加到总列表
                for line in block['lines']:
                    bbox = line['bbox']
                    page_line_list.append(bbox)
        # 如果是图片、表格、行间公式的主体部分
        elif block['type'] in [BlockType.ImageBody, BlockType.TableBody, BlockType.InterlineEquation]:
            # 备份原始行信息 (如果存在)
            block['real_lines'] = copy.deepcopy(block['lines'])
            # 调用辅助函数，为这个块生成虚拟行用于排序
            add_lines_to_block(block)

    # 如果页面总行数超过 200 (LayoutReader 模型支持上限通常是 512，这里设为 200 作为阈值)
    if len(page_line_list) > 200:
        # 返回 None，表示无法使用模型排序 (后续会退回 XY-Cut)
        return None

    # --- 使用 LayoutReader 模型进行排序 ---
    # 计算页面宽高到模型输入尺寸 (1000x1000) 的缩放比例
    x_scale = 1000.0 / page_w
    y_scale = 1000.0 / page_h
    # 初始化用于模型输入的 boxes 列表
    boxes = []
    # logger.info(f"Scale: {x_scale}, {y_scale}, Boxes len: {len(page_line_list)}") # 记录日志 (已注释)
    # 遍历页面所有行的 bbox
    for left, top, right, bottom in page_line_list:
        # 边界检查和修正，确保坐标在页面范围内
        if left < 0:
            logger.warning(
                f'left < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            left = 0
        if right > page_w:
            logger.warning(
                f'right > page_w, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            right = page_w
        if top < 0:
            logger.warning(
                f'top < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            top = 0
        if bottom > page_h:
            logger.warning(
                f'bottom > page_h, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            bottom = page_h
        # 应用缩放比例，并将坐标转换为整数
        left = round(left * x_scale)
        top = round(top * y_scale)
        right = round(right * x_scale)
        bottom = round(bottom * y_scale)
        # 断言检查缩放后的坐标是否在 [0, 1000] 范围内
        assert (
            1000 >= right >= left >= 0 and 1000 >= bottom >= top >= 0
        ), f'Invalid box. right: {right}, left: {left}, bottom: {bottom}, top: {top}'  # noqa: E126, E121
        # 将缩放后的坐标添加到 boxes 列表
        boxes.append([left, top, right, bottom])

    # 获取模型单例管理器
    model_manager = ModelSingleton()
    # 获取 LayoutReader 模型实例
    model = model_manager.get_model('layoutreader')
    # 在不计算梯度的模式下执行预测
    with torch.no_grad():
        # 调用 do_predict 函数获取排序后的索引列表
        orders = do_predict(boxes, model)
    # 根据排序索引重新排列原始的行 bbox 列表
    sorted_bboxes = [page_line_list[i] for i in orders]
    # 返回排序后的行 bbox 列表
    return sorted_bboxes

# 定义一个函数，计算页面中文本行高度的中位数
def get_line_height(blocks):
    # 初始化行高列表
    page_line_height_list = []
    # 遍历所有块
    for block in blocks:
        # 只考虑文本、标题、图注、表注等类型的块
        if block['type'] in [
            BlockType.Text, BlockType.Title,
            BlockType.ImageCaption, BlockType.ImageFootnote,
            BlockType.TableCaption, BlockType.TableFootnote
        ]:
            # 遍历块内的每一行
            for line in block['lines']:
                # 获取行 bbox
                bbox = line['bbox']
                # 计算行高并添加到列表
                page_line_height_list.append(int(bbox[3] - bbox[1]))
    # 如果列表不为空
    if len(page_line_height_list) > 0:
        # 返回行高的中位数
        return statistics.median(page_line_height_list)
    else:
        # 如果没有文本行，返回一个默认值 10
        return 10

# 定义一个函数，处理分组后的图/表元素（主体、标题、脚注）
def process_groups(groups, body_key, caption_key, footnote_key):
    # 初始化主体、标题、脚注的块列表
    body_blocks = []
    caption_blocks = []
    footnote_blocks = []
    # 遍历分组列表 (每个 group 包含一个主体和对应的标题/脚注列表)
    for i, group in enumerate(groups):
        # 为主体块添加 group_id
        group[body_key]['group_id'] = i
        body_blocks.append(group[body_key])
        # 遍历标题块列表
        for caption_block in group[caption_key]:
            # 添加 group_id
            caption_block['group_id'] = i
            caption_blocks.append(caption_block)
        # 遍历脚注块列表
        for footnote_block in group[footnote_key]:
            # 添加 group_id
            footnote_block['group_id'] = i
            footnote_blocks.append(footnote_block)
    # 返回三个包含 group_id 的块列表
    return body_blocks, caption_blocks, footnote_blocks

# 定义一个函数，将属于同一组 (group_id 相同) 的图/表相关块合并成一个逻辑块
def process_block_list(blocks, body_type, block_type):
    # 提取组内所有块的索引
    indices = [block['index'] for block in blocks]
    # 计算索引的中位数，作为合并后逻辑块的索引
    median_index = statistics.median(indices)
    # 找到组内主体块 (body_type) 的 bbox，作为合并后逻辑块的 bbox (如果找不到则为空列表)
    body_bbox = next((block['bbox'] for block in blocks if block.get('type') == body_type), [])
    # 返回合并后的逻辑块字典
    return {
        'type': block_type, # 逻辑块类型 (如 BlockType.Image 或 BlockType.Table)
        'bbox': body_bbox, # 使用主体部分的 bbox
        'blocks': blocks, # 包含组内所有原始块的列表
        'index': median_index, # 使用索引中位数
    }

# 定义一个函数，将之前按 group_id 分开处理的图/表块还原为合并后的逻辑块形式
def revert_group_blocks(blocks):
    # 初始化用于存储图像组和表格组的字典
    image_groups = {} # {group_id: [block1, block2, ...]}
    table_groups = {} # {group_id: [block1, block2, ...]}
    # 初始化存储非图/表块的新列表
    new_blocks = []
    # 遍历所有输入块
    for block in blocks:
        # 如果是图像相关块 (主体、标题、脚注)
        if block['type'] in [BlockType.ImageBody, BlockType.ImageCaption, BlockType.ImageFootnote]:
            # 获取 group_id
            group_id = block['group_id']
            # 如果该 group_id 首次出现，则在字典中创建新列表
            if group_id not in image_groups:
                image_groups[group_id] = []
            # 将块添加到对应 group_id 的列表中
            image_groups[group_id].append(block)
        # 如果是表格相关块 (主体、标题、脚注)
        elif block['type'] in [BlockType.TableBody, BlockType.TableCaption, BlockType.TableFootnote]:
            # 获取 group_id
            group_id = block['group_id']
            # 如果该 group_id 首次出现，则在字典中创建新列表
            if group_id not in table_groups:
                table_groups[group_id] = []
            # 将块添加到对应 group_id 的列表中
            table_groups[group_id].append(block)
        else:
            # 如果是非图/表块，直接添加到新列表中
            new_blocks.append(block)

    # 遍历收集到的图像组
    for group_id, group_blocks in image_groups.items():
        # 调用 process_block_list 将同一组的块合并成一个逻辑图像块
        new_blocks.append(process_block_list(group_blocks, BlockType.ImageBody, BlockType.Image))
    # 遍历收集到的表格组
    for group_id, group_blocks in table_groups.items():
        # 调用 process_block_list 将同一组的块合并成一个逻辑表格块
        new_blocks.append(process_block_list(group_blocks, BlockType.TableBody, BlockType.Table))
    # 返回包含合并后逻辑块和原始非图/表块的新列表
    return new_blocks

# 定义一个函数，移除那些不与任何有效块或丢弃块重叠的 span (过滤掉可能悬空的 span)
def remove_outside_spans(spans, all_bboxes, all_discarded_blocks):
    # 定义一个内部辅助函数，用于从块列表中提取指定类型的 bbox
    def get_block_bboxes(blocks, block_type_list):
        # 提取块列表 (all_bboxes 或 all_discarded_blocks) 中类型在 block_type_list 内的块的 bbox (前 4 个元素)
        return [block[0:4] for block in blocks if block[7] in block_type_list] # block[7] 存储块类型

    # 获取所有图像主体块的 bbox 列表
    image_bboxes = get_block_bboxes(all_bboxes, [BlockType.ImageBody])
    # 获取所有表格主体块的 bbox 列表
    table_bboxes = get_block_bboxes(all_bboxes, [BlockType.TableBody])
    # 定义其他块类型的列表 (除了图像和表格主体)
    other_block_type = []
    for block_type_value in BlockType.__dict__.values(): # 遍历 BlockType 枚举的所有值
        if not isinstance(block_type_value, str): # 跳过非字符串值 (如枚举成员本身)
            continue
        if block_type_value not in [BlockType.ImageBody, BlockType.TableBody]: # 如果不是图像或表格主体
            other_block_type.append(block_type_value) # 添加到其他类型列表
    # 获取所有其他类型有效块的 bbox 列表
    other_block_bboxes = get_block_bboxes(all_bboxes, other_block_type)
    # 获取所有丢弃块的 bbox 列表
    discarded_block_bboxes = get_block_bboxes(all_discarded_blocks, [BlockType.Discarded]) # 假设丢弃块类型为 Discarded

    # 初始化存储保留下来的 span 的新列表
    new_spans = []
    # 遍历输入的 spans 列表
    for span in spans:
        span_bbox = span['bbox'] # 获取 span 的 bbox
        span_type = span['type'] # 获取 span 的类型

        # 检查 span 是否与任何丢弃块有足够的重叠 (重叠面积占 span 面积 > 0.4)
        if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.4 for block_bbox in discarded_block_bboxes):
            # 如果与丢弃块重叠，则保留该 span (可能是有意丢弃区域内的文本)
            new_spans.append(span)
            continue # 处理下一个 span

        # 根据 span 类型检查与对应有效块的重叠
        if span_type == ContentType.Image: # 如果是图像 span
            # 检查是否与任何图像主体块有足够的重叠 (> 0.5)
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in image_bboxes):
                new_spans.append(span) # 保留
        elif span_type == ContentType.Table: # 如果是表格 span
            # 检查是否与任何表格主体块有足够的重叠 (> 0.5)
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in table_bboxes):
                new_spans.append(span) # 保留
        else: # 如果是其他类型的 span (如文本、公式等)
            # 检查是否与任何其他类型的有效块有足够的重叠 (> 0.5)
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in other_block_bboxes):
                new_spans.append(span) # 保留
    # 返回过滤后的 span 列表
    return new_spans

# 定义核心的单页处理函数
def parse_page_core(
    page_doc: PageableData, # 当前页面的数据对象
    magic_model, # MagicModel 实例，包含模型推理结果
    page_id, # 当前页面的 ID (通常是页码)
    pdf_bytes_md5, # 整个 PDF 文件的 MD5 哈希值
    imageWriter, # 图片写入器实例，用于保存裁剪的图片
    parse_mode, # 解析模式 (TXT 或 OCR)
    lang # 页面语言
):
    # 初始化页面是否需要丢弃的标志和原因列表
    need_drop = False
    drop_reason = []

    """从 magic_model 对象中获取当前页面的各种区块信息"""
    # 获取图像组信息 (主体、标题、脚注)
    img_groups = magic_model.get_imgs_v2(page_id)
    # 获取表格组信息 (主体、标题、脚注)
    table_groups = magic_model.get_tables_v2(page_id)

    """对 image 和 table 的区块分组，并添加 group_id"""
    img_body_blocks, img_caption_blocks, img_footnote_blocks = process_groups(
        img_groups, 'image_body', 'image_caption_list', 'image_footnote_list'
    )
    table_body_blocks, table_caption_blocks, table_footnote_blocks = process_groups(
        table_groups, 'table_body', 'table_caption_list', 'table_footnote_list'
    )

    # 获取丢弃区域块信息
    discarded_blocks = magic_model.get_discarded(page_id)
    # 获取文本块信息
    text_blocks = magic_model.get_text_blocks(page_id)
    # 获取标题块信息
    title_blocks = magic_model.get_title_blocks(page_id)
    # 获取公式信息 (行内公式 span，行间公式 span，行间公式 block)
    inline_equations, interline_equations, interline_equation_blocks = magic_model.get_equations(page_id)
    # 获取页面尺寸 (宽度和高度)
    page_w, page_h = magic_model.get_page_size(page_id)

    # 定义一个内部函数，用于合并相邻且 Y 轴重叠的标题块
    def merge_title_blocks(blocks, x_distance_threshold=0.1 * page_w): # x 轴距离阈值设为页面宽度的 10%
        # 定义合并两个 bbox 的函数
        def merge_two_bbox(b1, b2):
            # 取两个 bbox 的最小 x_min, y_min 和最大 x_max, y_max
            x_min = min(b1['bbox'][0], b2['bbox'][0])
            y_min = min(b1['bbox'][1], b2['bbox'][1])
            x_max = max(b1['bbox'][2], b2['bbox'][2])
            y_max = max(b1['bbox'][3], b2['bbox'][3])
            return x_min, y_min, x_max, y_max

        # 定义合并两个 block 的函数
        def merge_two_blocks(b1, b2):
            # 合并 bbox
            b1['bbox'] = merge_two_bbox(b1, b2)
            # 合并文本内容 (假设标题块都只有一行)
            line1 = b1['lines'][0]
            line2 = b2['lines'][0]
            # 合并行 bbox
            line1['bbox'] = merge_two_bbox(line1, line2)
            # 合并 spans
            line1['spans'].extend(line2['spans'])
            # 返回合并后的块 b1 和被合并的块 b2 (用于后续移除)
            return b1, b2

        # --- 按 Y 轴重叠度聚集标题块 ---
        y_overlapping_blocks = [] # 存储按行聚集的标题块 [[row1_block1, row1_block2], [row2_block1], ...]
        # 提取所有类型为 Title 的块
        title_bs = [b for b in blocks if b['type'] == BlockType.Title]
        # 当还有未处理的标题块时循环
        while title_bs:
            # 取出第一个标题块
            block1 = title_bs.pop(0)
            # 初始化当前行列表
            current_row = [block1]
            # 初始化需要从 title_bs 中移除的块列表
            to_remove = []
            # 遍历剩余的标题块
            for block2 in title_bs:
                # 检查 block2 是否与 block1 在 Y 轴上高度重叠 (阈值 0.9)
                # 并且两个块都只有一行 (避免合并多行标题)
                if (
                    __is_overlaps_y_exceeds_threshold(block1['bbox'], block2['bbox'], 0.9)
                    and len(block1['lines']) == 1
                    and len(block2['lines']) == 1
                ):
                    # 如果满足条件，将 block2 加入当前行，并标记为待移除
                    current_row.append(block2)
                    to_remove.append(block2)
            # 从 title_bs 中移除已加入当前行的块
            for b in to_remove:
                title_bs.remove(b)
            # 将当前行添加到 y_overlapping_blocks 列表中
            y_overlapping_blocks.append(current_row)

        # --- 按 X 轴坐标排序并合并同一行的标题块 ---
        to_remove_blocks = [] # 存储被合并掉的块，最后统一移除
        # 遍历按行聚集的标题块
        for row in y_overlapping_blocks:
            # 如果行内只有一个块，则跳过
            if len(row) == 1:
                continue
            # 按 X 轴坐标 (左边界 x0) 对行内块进行排序
            row.sort(key=lambda x: x['bbox'][0])
            # 初始化合并后的块为行内第一个块
            merged_block = row[0]
            # 遍历行内从第二个块开始的后续块
            for i in range(1, len(row)):
                left_block = merged_block # 当前已合并的左侧块
                right_block = row[i] # 待检查的右侧块
                # 计算左右块的高度
                left_height = left_block['bbox'][3] - left_block['bbox'][1]
                right_height = right_block['bbox'][3] - right_block['bbox'][1]
                # 检查是否需要合并：
                # 1. 右块的左边界与左块的右边界距离小于阈值
                # 2. 两个块的高度相近 (右块高度在左块高度的 0.95 到 1.05 倍之间)
                if (
                    right_block['bbox'][0] - left_block['bbox'][2] < x_distance_threshold
                    and left_height * 0.95 < right_height < left_height * 1.05
                ):
                    # 如果满足条件，则合并两个块
                    merged_block, to_remove_block = merge_two_blocks(merged_block, right_block)
                    # 将被合并的块标记为待移除
                    to_remove_blocks.append(to_remove_block)
                else:
                    # 如果不满足合并条件，则将当前右块作为新的 merged_block，继续向右比较
                    merged_block = right_block
        # 从原始 blocks 列表中移除所有被合并掉的块
        for b in to_remove_blocks:
            blocks.remove(b) # 注意：这里直接修改了传入的 blocks 列表

    """将所有需要参与布局分析的区块 bbox 整理到一起"""
    # interline_equation_blocks 参数目前不够准确，优先使用 interline_equations (span 列表)
    # (代码逻辑似乎有些混乱，先假设 interline_equation_blocks 为空)
    interline_equation_blocks = [] # 强制置空，使用下面的 else 分支
    if len(interline_equation_blocks) > 0: # 这个分支目前不会进入
        # 调用函数准备用于布局分割的 bbox 列表 (包含有效 bbox 和丢弃 bbox)
        all_bboxes, all_discarded_blocks = ocr_prepare_bboxes_for_layout_split_v2(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equation_blocks, # 使用 block 列表
            page_w,
            page_h,
        )
    else: # 使用行间公式 span 列表
        all_bboxes, all_discarded_blocks = ocr_prepare_bboxes_for_layout_split_v2(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equations, # 使用 span 列表
            page_w,
            page_h,
        )

    """获取当前页面所有的 spans 信息"""
    spans = magic_model.get_all_spans(page_id)

    """预处理 spans: 过滤掉无效或重叠的 span"""
    # 移除不与任何块 (有效或丢弃) 重叠的 span
    spans = remove_outside_spans(spans, all_bboxes, all_discarded_blocks)
    # 移除重叠 spans 中置信度较低的
    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    # 移除重叠 spans 中面积较小的
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)

    """根据 parse_mode (TXT 或 OCR) 进一步处理 spans，主要是填充文本内容"""
    if parse_mode == SupportedPdfParseMethod.TXT:
        # 使用基于 PyMuPDF 提取 + 辅助 OCR 的方案填充文本 span
        spans = txt_spans_extract_v2(page_doc, spans, all_bboxes, all_discarded_blocks, lang)
    elif parse_mode == SupportedPdfParseMethod.OCR:
        # 如果是纯 OCR 模式，这里假设 spans 已经包含了 OCR 结果，不做额外处理
        pass
    else:
        # 如果 parse_mode 无效，抛出异常
        raise Exception('parse_mode must be txt or ocr')

    """处理丢弃区域 (discarded_blocks)，将落在其内部的 spans 填充进去"""
    # 将与 discarded_blocks 重叠度 > 0.4 的 spans 填充进去
    discarded_block_with_spans, spans = fill_spans_in_blocks(
        all_discarded_blocks, spans, 0.4 # 使用较低的重叠阈值
    )
    # 对填充了 spans 的 discarded_blocks 进行修复/整理
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    """如果当前页面没有有效的 bbox (all_bboxes 为空)，则认为此页为空或无效，直接构造空页面信息并返回"""
    if len(all_bboxes) == 0:
        logger.warning(f'skip this page, not found useful bbox, page_id: {page_id}')
        # 调用页面构建函数，传入空列表
        return ocr_construct_page_component_v2(
            [], # sorted_blocks
            [], # paragraphs
            page_id, page_w, page_h,
            [], # images (旧格式)
            [], # tables (旧格式)
            [], # formulas (旧格式)
            interline_equations, # 行间公式 (新格式)
            fix_discarded_blocks, # 丢弃块
            True, # need_drop 设为 True
            ['not found useful bbox'], # drop_reason
        )

    """对需要截图的 span (图像和表格) 进行截图并保存"""
    spans = ocr_cut_image_and_table(
        spans, page_doc, page_id, pdf_bytes_md5, imageWriter
    )

    """将剩余的 spans 填充到有效的区块 (all_bboxes) 中"""
    block_with_spans, spans = fill_spans_in_blocks(all_bboxes, spans, 0.5) # 使用 0.5 的重叠阈值

    """对填充了 spans 的有效区块进行修复/整理"""
    fix_blocks = fix_block_spans_v2(block_with_spans)

    """合并同一行内可能被断开的标题块"""
    merge_title_blocks(fix_blocks)

    """计算页面中文本行的估计高度"""
    line_height = get_line_height(fix_blocks)

    """使用模型 (LayoutReader) 或 XY-Cut 对所有行 (包括虚拟行) 进行排序"""
    sorted_bboxes = sort_lines_by_model(fix_blocks, page_w, page_h, line_height)

    """根据行的排序结果计算块的索引"""
    fix_blocks = cal_block_index(fix_blocks, sorted_bboxes)

    """将按 group_id 分开处理的图/表块还原为合并后的逻辑块形式"""
    fix_blocks = revert_group_blocks(fix_blocks)

    """根据块索引对所有块进行最终排序"""
    sorted_blocks = sorted(fix_blocks, key=lambda b: b['index'])

    """对合并后的图/表逻辑块内部的子块 (标题、脚注) 进行排序"""
    for block in sorted_blocks:
        if block['type'] in [BlockType.Image, BlockType.Table]:
            # 对逻辑块内部的 'blocks' 列表按索引排序
            block['blocks'] = sorted(block['blocks'], key=lambda b: b['index'])

    """提取用于 QA (问答) 或特定格式输出所需的图像、表格、行间公式列表"""
    images, tables, interline_equations_qa = get_qa_need_list_v2(sorted_blocks) # 注意：这里的 interline_equations_qa 变量名与前面不同

    """构造最终的页面信息字典"""
    page_info = ocr_construct_page_component_v2(
        sorted_blocks, # 排序和处理后的块列表
        [], # 段落列表 (此时为空，将在后续步骤填充)
        page_id, page_w, page_h,
        [], # images (旧格式，为空)
        images, # 提取出的图像列表 (新格式)
        tables, # 提取出的表格列表 (新格式)
        interline_equations_qa, # 提取出的行间公式列表 (新格式)
        fix_discarded_blocks, # 处理后的丢弃块列表
        need_drop, # 是否需要丢弃此页的标志
        drop_reason, # 丢弃原因列表
    )
    # 返回构造好的页面信息字典
    return page_info

# 定义 PDF 解析的主函数 (联合 TXT 和 OCR)
def pdf_parse_union(
    model_list, # 模型推理结果列表 (通常是每页一个字典)
    dataset: Dataset, # 数据集对象，包含 PDF 页面数据
    imageWriter, # 图片写入器实例
    parse_mode, # 解析模式 (TXT 或 OCR)
    start_page_id=0, # 开始处理的页面 ID (默认为 0)
    end_page_id=None, # 结束处理的页面 ID (默认为最后一页)
    debug_mode=False, # 是否启用调试模式 (主要用于打印耗时)
    lang=None, # PDF 主要语言
):
    # 计算 PDF 文件内容的 MD5 哈希值，用于图片命名等
    pdf_bytes_md5 = compute_md5(dataset.data_bits())

    """初始化空的 pdf_info 字典，用于存储每页的处理结果"""
    pdf_info_dict = {} # { "page_0": page_info_0, "page_1": page_info_1, ... }

    """使用模型推理结果 (model_list) 和数据集对象 (dataset) 初始化 MagicModel"""
    magic_model = MagicModel(model_list, dataset)

    """确定实际处理的结束页面 ID"""
    end_page_id = (
        end_page_id
        if end_page_id is not None and end_page_id >= 0 # 如果提供了有效的 end_page_id
        else len(dataset) - 1 # 否则使用数据集的最后一页索引
    )
    # 再次检查 end_page_id 是否超出范围
    if end_page_id > len(dataset) - 1:
        logger.warning('end_page_id is out of range, use pdf_docs length')
        end_page_id = len(dataset) - 1

    # """初始化计时器 (已注释)"""
    # start_time = time.time()

    # --- 遍历数据集中的每一页 ---
    # 使用 tqdm 显示处理进度
    for page_id, page in tqdm(enumerate(dataset), total=len(dataset), desc="Processing pages"):
        # # --- 调试模式下打印每页耗时 (已注释) ---
        # if debug_mode:
        #     time_now = time.time()
        #     logger.info(
        #         f'page_id: {page_id}, last_page_cost_time: {round(time.time() - start_time, 2)}'
        #     )
        #     start_time = time_now

        # --- 处理指定范围内的页面 ---
        if start_page_id <= page_id <= end_page_id:
            # 调用核心处理函数 parse_page_core 处理当前页面
            page_info = parse_page_core(
                page, magic_model, page_id, pdf_bytes_md5, imageWriter, parse_mode, lang
            )
        else:
            # 如果页面不在处理范围内，则构造一个标记为跳过的空页面信息
            page_info_raw = page.get_page_info() # 获取原始页面信息 (主要是宽高)
            page_w = page_info_raw.w
            page_h = page_info_raw.h
            page_info = ocr_construct_page_component_v2(
                [], [], page_id, page_w, page_h, [], [], [], [], [], True, ['skip page'] # need_drop=True, reason='skip page'
            )
        # 将处理结果存储到 pdf_info_dict 中
        pdf_info_dict[f'page_{page_id}'] = page_info

    # --- 批量处理在 txt_spans_extract_v2 中标记需要 OCR 的 spans ---
    need_ocr_list = [] # 存储需要 OCR 的 span 对象
    img_crop_list = [] # 存储对应的截图 (numpy 数组)
    text_block_list = [] # 临时列表，用于收集所有可能包含待 OCR span 的块

    # 遍历所有页面的处理结果
    for pange_id, page_info in pdf_info_dict.items():
        # 收集页面中的有效块 (preproc_blocks)
        for block in page_info['preproc_blocks']:
            # 如果是逻辑图/表块，则收集其内部的标题/脚注块
            if block['type'] in ['table', 'image']:
                for sub_block in block['blocks']:
                    if sub_block['type'] in ['image_caption', 'image_footnote', 'table_caption', 'table_footnote']:
                        text_block_list.append(sub_block)
            # 如果是文本或标题块，直接收集
            elif block['type'] in ['text', 'title']:
                text_block_list.append(block)
        # 收集页面中的丢弃块 (discarded_blocks)，它们也可能包含需要 OCR 的 span
        for block in page_info['discarded_blocks']:
            text_block_list.append(block)

    # 遍历收集到的所有块
    for block in text_block_list:
        # 遍历块中的行
        for line in block['lines']:
            # 遍历行中的 span
            for span in line['spans']:
                # 如果 span 中包含 'np_img' 键 (在 txt_spans_extract_v2 中添加的标记)
                if 'np_img' in span:
                    # 将 span 对象和对应的截图添加到待处理列表
                    need_ocr_list.append(span)
                    img_crop_list.append(span['np_img'])
                    # 从 span 对象中移除截图数据，避免冗余存储
                    span.pop('np_img')

    # --- 如果存在需要 OCR 的截图 ---
    if len(img_crop_list) > 0:
        # 获取 OCR 模型实例
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            ocr_show_log=False, # 不显示 OCR 内部日志
            det_db_box_thresh=0.3, # 检测阈值 (虽然这里只做识别，但初始化可能需要)
            lang=lang # 指定语言
        )
        # rec_start = time.time() # 记录识别开始时间 (已注释)
        # 对所有截图进行批量 OCR 识别 (det=False 表示只识别)
        # tqdm_enable=True 会在 OCR 内部显示进度条
        ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0] # [0] 获取结果列表
        # rec_time = time.time() - rec_start # 计算识别耗时 (已注释)
        # logger.info(f'ocr-dynamic-rec time: {round(rec_time, 2)}, total images processed: {len(img_crop_list)}') # 记录日志 (已注释)

        # --- 验证 OCR 结果数量 ---
        # 断言确保 OCR 返回结果数量与输入截图数量一致
        assert len(ocr_res_list) == len(need_ocr_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}'

        # --- 处理 OCR 识别结果 ---
        # 遍历需要 OCR 的 span 对象列表
        for index, span in enumerate(need_ocr_list):
            # 从 OCR 结果列表中获取对应的文本和分数
            ocr_text, ocr_score = ocr_res_list[index]
            # 更新 span 的 content
            span['content'] = ocr_text
            # 更新 span 的 score (保留两位小数)
            span['score'] = float(round(ocr_score, 2))


    """执行段落切分逻辑"""
    para_split(pdf_info_dict) # 该函数会直接修改 pdf_info_dict 中的内容，划分段落

    """执行 LLM (大语言模型) 辅助优化"""
    # 获取 LLM 辅助配置
    llm_aided_config = get_llm_aided_config()
    # 如果配置存在
    if llm_aided_config is not None:
        # --- 公式优化 ---
        formula_aided_config = llm_aided_config.get('formula_aided', None) # 获取公式优化配置
        if formula_aided_config is not None:
            if formula_aided_config.get('enable', False): # 如果启用公式优化
                llm_aided_formula_start_time = time.time() # 记录开始时间
                # 调用公式优化函数
                llm_aided_formula(pdf_info_dict, formula_aided_config)
                logger.info(f'llm aided formula time: {round(time.time() - llm_aided_formula_start_time, 2)}') # 记录耗时
        # --- 文本优化 ---
        text_aided_config = llm_aided_config.get('text_aided', None) # 获取文本优化配置
        if text_aided_config is not None:
            if text_aided_config.get('enable', False): # 如果启用文本优化
                llm_aided_text_start_time = time.time() # 记录开始时间
                # 调用文本优化函数
                llm_aided_text(pdf_info_dict, text_aided_config)
                logger.info(f'llm aided text time: {round(time.time() - llm_aided_text_start_time, 2)}') # 记录耗时
        # --- 标题优化 ---
        title_aided_config = llm_aided_config.get('title_aided', None) # 获取标题优化配置
        if title_aided_config is not None:
            if title_aided_config.get('enable', False): # 如果启用标题优化
                llm_aided_title_start_time = time.time() # 记录开始时间
                # 调用标题优化函数
                llm_aided_title(pdf_info_dict, title_aided_config)
                logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}') # 记录耗时

    """将按页存储的字典转换为列表形式"""
    pdf_info_list = dict_to_list(pdf_info_dict) # 将 {"page_0": info0, ...} 转换为 [info0, info1, ...]

    """构建最终返回的字典结构"""
    new_pdf_info_dict = {
        'pdf_info': pdf_info_list, # 包含所有页面信息的列表
    }

    """清理 GPU 内存"""
    clean_memory(get_device()) # 调用内存清理函数

    # 返回最终的处理结果字典
    return new_pdf_info_dict

# 当脚本作为主程序运行时执行的代码块
if __name__ == '__main__':
    # 通常用于测试或示例代码，这里为空
    pass