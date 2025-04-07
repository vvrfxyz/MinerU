# 从 doclayout_yolo 模块导入 YOLOv10 类。
from doclayout_yolo import YOLOv10
# 从 tqdm 库导入 tqdm 类，用于显示进度条。
from tqdm import tqdm

# 定义一个名为 DocLayoutYOLOModel 的类，用于文档布局检测。
class DocLayoutYOLOModel(object):
    # 初始化方法，在创建类实例时调用。
    def __init__(self, weight, device):
        # 使用给定的权重文件初始化 YOLOv10 模型。
        self.model = YOLOv10(weight)
        # 存储指定的运行设备（如 'cpu' 或 'cuda:0'）。
        self.device = device

    # 定义预测方法，用于对单个图像进行布局检测。
    def predict(self, image):
        # 初始化一个空列表，用于存储布局检测结果。
        layout_res = []
        # 调用 YOLOv10 模型的 predict 方法进行目标检测。
        # image: 输入图像。
        # imgsz=1280: 指定推理时的图像大小。
        # conf=0.10: 设置置信度阈值，低于此阈值的检测框将被忽略。
        # iou=0.45: 设置非极大值抑制（NMS）的交并比（IoU）阈值。
        # verbose=False: 关闭详细输出。
        # device=self.device: 指定运行推理的设备。
        # [0]: YOLOv10 predict 返回一个列表，通常只包含一个结果对象（针对单张图），所以取第一个元素。
        doclayout_yolo_res = self.model.predict(
            image,
            imgsz=1280, # 推理图像大小设置为1280
            conf=0.10, # 置信度阈值设为0.10
            iou=0.45, # NMS的IoU阈值设为0.45
            verbose=False, # 不打印详细日志
            device=self.device # 指定运行设备
        )[0] # 获取第一个结果，因为是单张图片预测

        # 遍历检测结果中的边界框坐标、置信度和类别。
        # doclayout_yolo_res.boxes.xyxy.cpu(): 获取边界框坐标 (xmin, ymin, xmax, ymax) 并转移到 CPU。
        # doclayout_yolo_res.boxes.conf.cpu(): 获取置信度分数并转移到 CPU。
        # doclayout_yolo_res.boxes.cls.cpu(): 获取类别标签并转移到 CPU。
        for xyxy, conf, cla in zip(
            doclayout_yolo_res.boxes.xyxy.cpu(), # 获取检测框坐标并移至CPU
            doclayout_yolo_res.boxes.conf.cpu(), # 获取检测框置信度并移至CPU
            doclayout_yolo_res.boxes.cls.cpu(), # 获取检测框类别并移至CPU
        ):
            # 将边界框坐标从张量（tensor）转换为整数。
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy] # 提取并转换坐标为整数
            # 创建一个新的字典来存储格式化的检测结果。
            new_item = {
                # "category_id": 存储检测到的类别ID（整数）。
                "category_id": int(cla.item()), # 存储类别ID
                # "poly": 存储表示边界框的多边形坐标 [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]。
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax], # 存储矩形框的四个顶点坐标
                # "score": 存储检测结果的置信度分数，保留3位小数。
                "score": round(float(conf.item()), 3), # 存储置信度分数，四舍五入保留3位小数
            }
            # 将格式化后的结果添加到 layout_res 列表中。
            layout_res.append(new_item)
        # 返回包含所有检测结果的列表。
        return layout_res

    # 定义批量预测方法，用于同时处理多张图像。
    def batch_predict(self, images: list, batch_size: int) -> list:
        # 初始化一个空列表，用于存储所有图像的布局检测结果。
        images_layout_res = []
        # 使用 tqdm 创建一个进度条，按指定的 batch_size 遍历图像列表。
        # range(0, len(images), batch_size) 生成批次的起始索引。
        # desc="Layout Predict": 设置进度条的描述文字。
        # for index in range(0, len(images), batch_size): # 原始的循环方式，没有进度条。
        for index in tqdm(range(0, len(images), batch_size), desc="Layout Predict"): # 使用tqdm显示批处理进度
            # 调用 YOLOv10 模型的 predict 方法对当前批次的图像进行推理。
            # images[index : index + batch_size]: 获取当前批次的图像列表。
            # 其他参数与单张图像预测类似。
            # 使用列表推导式将每个图像的预测结果（在GPU上）转移到CPU。
            doclayout_yolo_res = [
                image_res.cpu() # 将单个图像的推理结果转移到 CPU。
                for image_res in self.model.predict(
                    images[index : index + batch_size], # 输入当前批次的图像。
                    imgsz=1280, # 推理图像大小
                    conf=0.10, # 置信度阈值
                    iou=0.45, # IoU 阈值
                    verbose=False, # 关闭详细输出
                    device=self.device, # 指定运行设备
                )
            ]
            # 遍历当前批次中每张图像的推理结果。
            for image_res in doclayout_yolo_res:
                # 为当前图像初始化一个空列表，用于存储其布局检测结果。
                layout_res = []
                # 遍历当前图像检测到的所有边界框、置信度和类别。
                # 注意：这里的 image_res 已经是转移到 CPU 上的结果。
                for xyxy, conf, cla in zip(
                    image_res.boxes.xyxy, # 获取边界框坐标。
                    image_res.boxes.conf, # 获取置信度。
                    image_res.boxes.cls, # 获取类别。
                ):
                    # 将边界框坐标从张量转换为整数。
                    xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy] # 提取并转换坐标为整数
                    # 创建格式化的结果字典。
                    new_item = {
                        "category_id": int(cla.item()), # 存储类别ID
                        "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax], # 存储矩形框的四个顶点坐标
                        "score": round(float(conf.item()), 3), # 存储置信度分数，保留3位小数
                    }
                    # 将当前检测结果添加到该图像的结果列表 layout_res 中。
                    layout_res.append(new_item)
                # 将当前图像的所有检测结果列表 layout_res 添加到最终的结果列表 images_layout_res 中。
                images_layout_res.append(layout_res)
        # 返回包含所有图像检测结果的列表（列表的列表）。
        return images_layout_res