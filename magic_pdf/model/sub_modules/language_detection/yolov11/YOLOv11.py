# Copyright (c) Opendatalab. All rights reserved. 
# 导入时间模块，用于计时
import time
# 从collections模块导入Counter类，用于统计元素出现次数
from collections import Counter
# 从uuid模块导入uuid4函数，用于生成唯一ID（虽然在此代码段未实际使用）
from uuid import uuid4
# 导入OpenCV库，用于图像处理
import cv2
# 导入NumPy库，用于数值计算和数组操作
import numpy as np
# 导入PyTorch库，用于深度学习模型
import torch
# 导入loguru库，用于日志记录
from loguru import logger
# 从ultralytics库导入YOLO类，用于目标检测/分类模型
from ultralytics import YOLO

# 定义一个字典，将语言缩写映射到中文全称
language_dict = {
    "ch": "中文简体",
    "en": "英语",
    "japan": "日语",
    "korean": "韩语",
    "fr": "法语",
    "german": "德语",
    "ar": "阿拉伯语",
    "ru": "俄语"
}

# 定义一个函数，用于拆分图像
def split_images(image, result_images=None):
    """
    对输入文件夹内的图片进行处理,若图片竖向(y方向)分辨率超过400,则进行拆分，
    每次平分图片,直至拆分出的图片竖向分辨率都满足400以下,将处理后的图片(拆分后的子图片)保存到输出文件夹。
    避免保存因裁剪区域超出图片范围导致出现的无效黑色图片部分。
    """
    # 如果结果列表未初始化，则创建一个空列表
    if result_images is None:
        result_images = []
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 获取宽度和高度中的较大值作为长边
    long_side = max(width, height)  # 获取较长边长度

    # 如果长边小于或等于400像素，则图像符合要求，直接添加到结果列表
    if long_side <= 400:
        result_images.append(image)
        return result_images

    # 计算新的长边长度（原长边的一半，向下取整）
    new_long_side = long_side // 2
    # 创建一个空列表，用于存储当前层级拆分出的子图像
    sub_images = []

    # 如果宽度大于或等于高度（即宽度是长边或图像为正方形）
    if width >= height:  # 如果宽度是较长边
        # 沿宽度方向，以new_long_side为步长进行切割
        for x in range(0, width, new_long_side):
            # 判断裁剪区域是否超出图片范围，如果超出则不进行裁剪保存操作
            if x + new_long_side > width:
                continue
            # 裁剪出子图像 [所有行, x到x+new_long_side列]
            sub_image = image[0:height, x:x + new_long_side]
            # 将子图像添加到临时列表
            sub_images.append(sub_image)
    # 如果高度大于宽度（即高度是长边）
    else:  # 如果高度是较长边
        # 沿高度方向，以new_long_side为步长进行切割
        for y in range(0, height, new_long_side):
            # 判断裁剪区域是否超出图片范围，如果超出则不进行裁剪保存操作
            if y + new_long_side > height:
                continue
            # 裁剪出子图像 [y到y+new_long_side行, 所有列]
            sub_image = image[y:y + new_long_side, 0:width]
            # 将子图像添加到临时列表
            sub_images.append(sub_image)

    # 遍历当前层级拆分出的所有子图像
    for sub_image in sub_images:
        # 递归调用split_images函数，对子图像进行进一步处理
        split_images(sub_image, result_images)
    # 返回包含所有最终（不再需要拆分）图像的列表
    return result_images

# 定义一个函数，将图像调整为224x224大小
def resize_images_to_224(image):
    """
    若分辨率小于224则用黑色背景补齐到224*224大小,若大于等于224则调整为224*224大小。
    Works directly with NumPy arrays. (直接处理NumPy数组)
    """
    try:
        # 获取原始图像的高度和宽度
        height, width = image.shape[:2]
        # 如果宽度或高度小于224
        if width < 224 or height < 224:
            # 创建一个224x224的黑色背景图像 (3通道，uint8类型)
            new_image = np.zeros((224, 224, 3), dtype=np.uint8)
            # 计算将原始图像粘贴到黑色背景中心的起始x坐标 (确保非负)
            paste_x = max(0, (224 - width) // 2)
            # 计算将原始图像粘贴到黑色背景中心的起始y坐标 (确保非负)
            paste_y = max(0, (224 - height) // 2)
            # 确定实际粘贴的宽度（防止原始图像比224还宽）
            paste_width = min(width, 224)
            # 确定实际粘贴的高度（防止原始图像比224还高）
            paste_height = min(height, 224)
            # 将原始图像（或其部分）粘贴到黑色背景上
            new_image[paste_y:paste_y + paste_height, paste_x:paste_x + paste_width] = image[:paste_height, :paste_width]
            # 更新image变量为处理后的新图像
            image = new_image
        # 如果宽度和高度都大于或等于224
        else:
            # 使用OpenCV的resize函数将图像调整为224x224，使用LANCZOS4插值法（一种高质量插值）
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        # 返回调整大小或填充后的224x224图像
        return image
    # 捕获处理过程中可能发生的任何异常
    except Exception as e:
        # 使用loguru记录异常信息
        logger.exception(f"Error in resize_images_to_224: {e}")
        # 如果发生错误，返回None
        return None

# 定义一个用于语言检测的YOLOv11模型类
class YOLOv11LangDetModel(object):
    # 类的初始化方法
    def __init__(self, langdetect_model_weight, device):
        # 使用提供的权重文件路径初始化YOLO模型
        self.model = YOLO(langdetect_model_weight)
        # 检查指定的设备是否为NPU (华为昇腾处理器)
        if str(device).startswith("npu"):
            # 如果是NPU，需要创建一个torch.device对象
            self.device = torch.device(device)
        else:
            # 对于其他设备（如 'cpu', 'cuda:0'），直接使用传入的字符串
            self.device = device

    # 定义执行语言检测的主要方法
    def do_detect(self, images: list):
        # 创建一个空列表，用于存储所有需要进行预测的图像 (经过预处理的)
        all_images = []
        # 遍历输入的原始图像列表
        for image in images:
            # 获取原始图像的高度和宽度
            height, width = image.shape[:2]
            # 如果图像的宽度和高度都小于100像素，则跳过该图像
            if width < 100 and height < 100:
                continue
            # 对当前图像进行拆分（如果需要的话）
            temp_images = split_images(image)
            # 遍历拆分后的子图像（如果未拆分，则列表只包含原始图像）
            for temp_image in temp_images:
                # 将每个子图像调整到224x224大小，并添加到待预测列表
                all_images.append(resize_images_to_224(temp_image))
        
        # # 记录语言检测开始时间 (代码被注释掉了)
        # langdetect_start = time.time()
        # 对所有处理后的图像进行批量预测
        images_lang_res = self.batch_predict(all_images, batch_size=256)
        # # 记录语言检测完成时间和图像数量 (代码被注释掉了)
        # logger.info(f"image number of langdetect: {len(images_lang_res)}, langdetect time: {round(time.time() - langdetect_start, 2)}")

        # 如果预测结果列表不为空
        if len(images_lang_res) > 0:
            # 使用Counter统计各种预测语言出现的次数
            count_dict = Counter(images_lang_res)
            # 找出出现次数最多的语言作为最终结果
            language = max(count_dict, key=count_dict.get)
        # 如果没有有效的预测结果
        else:
            # 将语言设置为None
            language = None
        # 返回检测到的最可能的语言代码
        return language

    # 定义对单个图像进行预测的方法
    def predict(self, image):
        # 使用YOLO模型对单个图像进行预测
        # verbose=False 表示不打印详细预测信息
        # device=self.device 指定在哪个设备上运行模型
        results = self.model.predict(image, verbose=False, device=self.device)
        # 获取预测结果中概率最高的类别的ID (转换为整数)
        predicted_class_id = int(results[0].probs.top1)
        # 从模型的名称映射中获取ID对应的类别名称（即语言名称）
        predicted_class_name = self.model.names[predicted_class_id]
        # 返回预测的语言名称
        return predicted_class_name

    # 定义批量预测方法，提高处理效率
    def batch_predict(self, images: list, batch_size: int) -> list:
        # 创建一个空列表，用于存储所有图像的预测结果
        images_lang_res = []
        # 以batch_size为步长，遍历图像列表的索引
        for index in range(0, len(images), batch_size):
            # 对当前批次的图像进行预测
            # 使用列表推导式处理批次结果，并将结果移到CPU（如果模型在GPU/NPU上运行）
            lang_res = [
                image_res.cpu() 
                for image_res in self.model.predict(
                    images[index: index + batch_size], # 取出当前批次的图像
                    verbose = False,                    # 不打印详细信息
                    device=self.device,                 # 指定运行设备
                )
            ]
            # 遍历当前批次的预测结果
            for res in lang_res:
                # 获取每个结果中概率最高的类别ID
                predicted_class_id = int(res.probs.top1)
                # 获取对应的语言名称
                predicted_class_name = self.model.names[predicted_class_id]
                # 将预测的语言名称添加到总结果列表中
                images_lang_res.append(predicted_class_name)
        # 返回包含所有图像预测语言名称的列表
        return images_lang_res