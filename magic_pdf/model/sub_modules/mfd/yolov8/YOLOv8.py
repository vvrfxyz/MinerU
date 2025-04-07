# 从 tqdm 库导入 tqdm 类，用于显示进度条。
from tqdm import tqdm
# 从 ultralytics 库导入 YOLO 类，这是YOLOv8等模型的官方实现库。
from ultralytics import YOLO

# 定义一个名为 YOLOv8MFDModel 的类，MFD 可能代表 Math Formula Detection (数学公式检测) 或类似含义。
class YOLOv8MFDModel(object):
    # 初始化方法，在创建类实例时调用。
    def __init__(self, weight, device="cpu"):
        # 使用给定的权重文件初始化 YOLO 模型。
        self.mfd_model = YOLO(weight)
        # 存储指定的运行设备，默认为 "cpu"。
        self.device = device

    # 定义预测方法，用于对单个图像进行检测。
    def predict(self, image):
        # 调用 YOLO 模型的 predict 方法进行目标检测。
        # image: 输入图像。
        # imgsz=1888: 指定推理时的图像大小，这个尺寸比较大，可能针对高分辨率或特定任务优化。
        # conf=0.25: 设置置信度阈值，过滤掉低置信度的检测结果。
        # iou=0.45: 设置非极大值抑制（NMS）的交并比（IoU）阈值。
        # verbose=False: 关闭详细的推理过程输出。
        # device=self.device: 指定运行推理的设备。
        # [0]: predict 方法对单张图像返回一个包含单个结果对象的列表，取第一个元素。
        mfd_res = self.mfd_model.predict(
            image, imgsz=1888, conf=0.25, iou=0.45, verbose=False, device=self.device
        )[0] # 对单张图片进行预测，获取结果列表的第一个元素
        # 返回原始的 YOLO 检测结果对象，包含了边界框、置信度、类别等信息。
        return mfd_res

    # 定义批量预测方法，用于同时处理多张图像。
    def batch_predict(self, images: list, batch_size: int) -> list:
        # 初始化一个空列表，用于存储所有图像的检测结果。
        images_mfd_res = []
        # 使用 tqdm 创建一个进度条，按指定的 batch_size 遍历图像列表。
        # range(0, len(images), batch_size) 生成批次的起始索引。
        # desc="MFD Predict": 设置进度条的描述文字。
        # for index in range(0, len(images), batch_size): # 原始的循环方式，没有进度条。
        for index in tqdm(range(0, len(images), batch_size), desc="MFD Predict"): # 使用tqdm显示批处理进度，描述为 "MFD Predict"
            # 调用 YOLO 模型的 predict 方法对当前批次的图像进行推理。
            # images[index : index + batch_size]: 获取当前批次的图像列表。
            # 其他参数与单张图像预测类似。
            # 使用列表推导式执行批量预测，并将每个图像的预测结果（可能在GPU上）转移到CPU。
            mfd_res = [
                image_res.cpu() # 将单个图像的推理结果转移到 CPU。
                for image_res in self.mfd_model.predict(
                    images[index : index + batch_size], # 输入当前批次的图像。
                    imgsz=1888, # 推理图像大小
                    conf=0.25, # 置信度阈值
                    iou=0.45, # IoU 阈值
                    verbose=False, # 关闭详细输出
                    device=self.device, # 指定运行设备
                )
            ]
            # 遍历当前批次中每张图像的推理结果（已经转移到CPU）。
            for image_res in mfd_res:
                # 将当前图像的检测结果对象直接添加到最终的结果列表 images_mfd_res 中。
                images_mfd_res.append(image_res)
        # 返回包含所有图像原始检测结果对象的列表。
        return images_mfd_res