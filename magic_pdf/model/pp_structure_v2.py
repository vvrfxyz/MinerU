# 导入 random 模块，用于生成随机数
import random

# 导入 loguru 模块，用于日志记录
from loguru import logger

try:
    # 尝试导入 paddleocr 的 PPStructure 类
    from paddleocr import PPStructure
except ImportError:
    # 如果导入失败，记录错误日志并退出程序
    logger.error('paddleocr not installed, please install by "pip install magic-pdf[lite]"')
    exit(1)


# 定义一个函数，将 paddleocr 返回的区域坐标转换为 [x0, y0, x1, y1] 格式的边界框
def region_to_bbox(region):
    # region 是一个包含四个点的列表，代表区域的四个角点
    # 例如: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    x0 = region[0][0] # 左上角 x 坐标
    y0 = region[0][1] # 左上角 y 坐标
    x1 = region[2][0] # 右下角 x 坐标
    y1 = region[2][1] # 右下角 y 坐标
    return [x0, y0, x1, y1] # 返回边界框坐标


# 定义一个自定义的 Paddle 模型类
class CustomPaddleModel:
    # 初始化方法
    def __init__(self,
                 ocr: bool = False, # 是否启用 OCR 识别文本内容，默认为 False
                 show_log: bool = False, # 是否显示 paddleocr 的日志，默认为 False
                 lang=None, # 指定 OCR 识别的语言，默认为 None (通常是中英文)
                 det_db_box_thresh=0.3, # DB 检测模型的边界框阈值
                 use_dilation=True, # 是否使用膨胀操作（通常用于表格）
                 det_db_unclip_ratio=1.8 # DB 检测模型 unclip ratio 参数
    ):
        # 如果指定了语言
        if lang is not None:
            # 使用指定语言初始化 PPStructure 模型
            # table=False 表示不进行表格结构识别（这里只做版面分析和OCR）
            # ocr=True 表示启用 OCR 功能（即使全局 ocr=False，这里也需要开启以获取文本行）
            self.model = PPStructure(table=False,
                                     ocr=True,
                                     show_log=show_log,
                                     lang=lang,
                                     det_db_box_thresh=det_db_box_thresh,
                                     use_dilation=use_dilation,
                                     det_db_unclip_ratio=det_db_unclip_ratio,
            )
        else:
            # 如果未指定语言，使用默认语言（通常是中英文混合）初始化 PPStructure 模型
            self.model = PPStructure(table=False,
                                     ocr=True,
                                     show_log=show_log,
                                     det_db_box_thresh=det_db_box_thresh,
                                     use_dilation=use_dilation,
                                     det_db_unclip_ratio=det_db_unclip_ratio,
            )

    # 定义类的调用方法，使其可以像函数一样被调用
    def __call__(self, img):
        try:
            # 尝试导入 OpenCV 库
            import cv2
        except ImportError:
            # 如果导入失败，记录错误日志并退出
            logger.error("opencv-python not installed, please install by pip.")
            exit(1)
        # 将输入的 RGB 图像转换为 BGR 格式，以适配 PaddleOCR 的输入要求
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 使用初始化的 PPStructure 模型处理图像
        result = self.model(img)
        # 初始化一个空列表，用于存储文本片段 (span) 信息
        spans = []
        # 遍历模型返回的每一行结果
        for line in result:
            # 移除结果中的图像数据 ("img")，因为它通常很大且后续不需要
            line.pop("img")
            """
            为 paddle 输出结果适配自定义的类别 ID (category_id)。
            不同的 "type" (版面元素类型) 映射到不同的数字 ID。
            title: 0 # 标题
            text: 1 # 普通文本
            header: 2 # 页眉 (映射到 2，可能后续会放弃或特殊处理)
            footer: 2 # 页脚 (映射到 2，可能后续会放弃或特殊处理)
            reference: 1 # 参考文献 (视为普通文本)
            equation: 8 # 行间公式 (块级)
            equation: 14 # 行间公式 (文本级，可能是 paddleocr 的某种内部表示，这里映射为 8)
            figure: 3 # 图片
            figure_caption: 4 # 图片标题
            table: 5 # 表格
            table_caption: 6 # 表格标题
            """
            # 根据 paddleocr 返回的 type 字段，设置 category_id
            if line["type"] == "title":
                line["category_id"] = 0
            elif line["type"] in ["text", "reference"]:
                line["category_id"] = 1
            elif line["type"] == "figure":
                line["category_id"] = 3
            elif line["type"] == "figure_caption":
                line["category_id"] = 4
            elif line["type"] == "table":
                line["category_id"] = 5
            elif line["type"] == "table_caption":
                line["category_id"] = 6
            elif line["type"] == "equation":
                # 注意：paddleocr 可能将公式块和公式内的文本都标记为 equation，这里统一映射为 8
                line["category_id"] = 8
            elif line["type"] in ["header", "footer"]:
                line["category_id"] = 2 # 页眉页脚暂时映射为 2
            else:
                # 如果遇到未知的类型，记录警告日志
                logger.warning(f"unknown type: {line['type']}")

            # 兼容不输出 score (置信度) 的 paddleocr 版本
            if line.get("score") is None:
                # 如果没有 score，则生成一个 0.5 到 1.0 之间的随机数作为得分
                line["score"] = 0.5 + random.random() * 0.5

            # 提取行内的文本识别结果 "res" (如果存在)
            # "res" 通常包含该行内识别出的每个文本片段的详细信息（坐标、文本、置信度）
            res = line.pop("res", None)
            # 如果 "res" 存在且不为空
            if res is not None and len(res) > 0:
                # 遍历 "res" 中的每个文本片段 (span)
                for span in res:
                    # 创建一个新的字典来存储格式化后的 span 信息
                    new_span = {
                        "category_id": 15, # 将识别出的文本片段标记为特定的 category_id (例如 15 代表 OCR 文本行)
                        "bbox": region_to_bbox(span["text_region"]), # 将 paddleocr 的区域坐标转换为 bbox 格式
                        "score": span["confidence"], # 获取文本识别的置信度
                        "text": span["text"], # 获取识别出的文本内容
                    }
                    # 将格式化后的 span 添加到 spans 列表中
                    spans.append(new_span)

        # 如果处理过程中产生了文本片段 (spans)
        if len(spans) > 0:
            # 将这些文本片段追加到主结果列表 `result` 中
            # 这意味着最终结果既包含版面元素（如标题、段落块），也包含这些块内的具体文本行
            result.extend(spans)

        # 返回处理后的结果列表
        return result [1]