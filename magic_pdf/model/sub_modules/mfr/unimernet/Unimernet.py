# 导入 torch 库，用于深度学习任务。
import torch
# 从 torch.utils.data 导入 DataLoader 和 Dataset 类，用于数据加载和处理。
from torch.utils.data import DataLoader, Dataset
# 从 tqdm 库导入 tqdm 类，用于显示进度条。
from tqdm import tqdm

# 定义一个名为 MathDataset 的类，继承自 torch.utils.data.Dataset，用于封装数学公式图像数据。
class MathDataset(Dataset):
    # 初始化方法。
    # image_paths: 包含图像数据（如numpy数组或PIL图像）的列表。
    # transform: 应用于图像的预处理转换。
    def __init__(self, image_paths, transform=None):
        # 存储图像数据列表。
        self.image_paths = image_paths
        # 存储图像预处理转换函数。
        self.transform = transform

    # 返回数据集中样本的数量。
    def __len__(self):
        return len(self.image_paths)

    # 根据索引 idx 获取数据集中的一个样本。
    def __getitem__(self, idx):
        # 获取原始图像数据。
        raw_image = self.image_paths[idx]
        # 如果定义了 transform，则应用它。
        if self.transform:
            # 对原始图像应用预处理转换。
            image = self.transform(raw_image)
            # 返回转换后的图像。
            return image
        # 如果没有定义transform，理论上应该返回原始图像，但当前实现没有处理这种情况。

# 定义一个名为 UnimernetModel 的类，用于执行数学公式识别 (MFR)。
class UnimernetModel(object):
    # 初始化方法。
    # weight_dir: 包含预训练模型权重和配置的目录。
    # cfg_path: 配置文件的路径（当前代码中未使用）。
    # _device_: 指定模型运行的设备（如 "cpu", "cuda:0", "mps"）。
    def __init__(self, weight_dir, cfg_path, _device_="cpu"):
        # 从本地 .unimernet_hf 模块导入 UnimernetModel 类（可能是Hugging Face Transformers风格的实现）。
        from .unimernet_hf import UnimernetModel
        # 检查设备是否为苹果 MPS (Metal Performance Shaders)。
        if _device_.startswith("mps"):
            # 如果是 MPS 设备，加载模型时指定 attn_implementation="eager"，可能为了兼容性或性能。
            self.model = UnimernetModel.from_pretrained(weight_dir, attn_implementation="eager")
        else:
            # 对于其他设备（CPU 或 CUDA），正常加载预训练模型。
            self.model = UnimernetModel.from_pretrained(weight_dir)
        # 存储设备信息。
        self.device = _device_
        # 将模型移动到指定的设备。
        self.model.to(_device_)
        # 如果设备不是 CPU。
        if not _device_.startswith("cpu"):
            # 将模型转换为半精度浮点数 (float16) 以加速推理并减少显存占用（在支持的GPU上）。
            self.model = self.model.to(dtype=torch.float16)
        # 将模型设置为评估模式，这会关闭 dropout 和 batch normalization 的更新。
        self.model.eval()

    # 定义预测方法，用于处理单个图像的数学公式检测 (MFD) 结果并进行识别。
    # mfd_res: 单张图像的 MFD 结果对象（来自如 YOLOv8MFDModel 的输出）。
    # image: 原始图像（numpy 数组）。
    def predict(self, mfd_res, image):
        # 初始化列表，存储格式化的公式信息。
        formula_list = []
        # 初始化列表，存储从原图中裁剪出的公式区域图像。
        mf_image_list = []
        # 遍历 MFD 检测结果中的边界框、置信度和类别。
        # 结果首先转移到 CPU 以便处理。
        for xyxy, conf, cla in zip(
            mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()
        ):
            # 将边界框坐标从张量转换为整数。
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            # 创建字典存储公式信息。类别ID加上13（原因不明，可能是为了区分或其他类别）。
            new_item = {
                "category_id": 13 + int(cla.item()), # 类别 ID 加上一个偏移量 13
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax], # 边界框多边形坐标
                "score": round(float(conf.item()), 2), # MFD 检测置信度，保留两位小数
                "latex": "", # 初始化 LaTeX 字符串为空
            }
            # 将格式化信息添加到列表。
            formula_list.append(new_item)
            # 从原图中裁剪出公式区域的图像。
            bbox_img = image[ymin:ymax, xmin:xmax]
            # 将裁剪出的公式图像添加到列表。
            mf_image_list.append(bbox_img)

        # 使用裁剪出的公式图像列表创建 MathDataset 实例，并应用模型自带的预处理转换。
        dataset = MathDataset(mf_image_list, transform=self.model.transform)
        # 创建 DataLoader 以便批量处理公式图像，batch_size设为32，num_workers=0表示在主进程中加载数据。
        dataloader = DataLoader(dataset, batch_size=32, num_workers=0)
        # 初始化列表，存储识别出的 LaTeX 结果。
        mfr_res = []
        # 遍历 DataLoader 提供的每个批次的公式图像。
        for mf_img in dataloader:
            # 将图像数据类型转换为模型所需的数据类型（可能是 float16）。
            mf_img = mf_img.to(dtype=self.model.dtype)
            # 将图像数据移动到指定的运行设备。
            mf_img = mf_img.to(self.device)
            # 使用 torch.no_grad() 上下文管理器，禁用梯度计算以节省内存并加速推理。
            with torch.no_grad():
                # 调用模型的 generate 方法进行公式识别，输入是包含图像张量的字典。
                output = self.model.generate({"image": mf_img})
            # 将当前批次识别出的 LaTeX 字符串（在 'fixed_str' 键中）添加到结果列表。
            mfr_res.extend(output["fixed_str"])

        # 将识别出的 LaTeX 结果填充回 formula_list 中对应的公式信息。
        for res, latex in zip(formula_list, mfr_res):
            res["latex"] = latex
        # 返回包含完整公式信息（包括识别出的 LaTeX）的列表。
        return formula_list

    # 定义批量预测方法，用于处理多张图像及其对应的 MFD 结果。
    # images_mfd_res: 包含多张图像 MFD 结果对象的列表。
    # images: 包含多张原始图像（numpy 数组）的列表。
    # batch_size: MFR 模型推理时的批处理大小，默认为 64。
    def batch_predict(self, images_mfd_res: list, images: list, batch_size: int = 64) -> list:
        # 初始化列表，用于存储每张原始图像对应的公式列表。
        images_formula_list = []
        # 初始化列表，存储所有图像中裁剪出的所有公式区域图像。
        mf_image_list = []
        # 初始化列表，按顺序存储所有公式的字典信息，用于后续回填 LaTeX 结果。
        backfill_list = []
        # 初始化列表，存储（面积，在 mf_image_list 中的索引，图像数据）元组，用于按面积排序。
        image_info = []  # Store (area, original_index, image) tuples

        # 收集所有图像中的公式区域图像及其元信息
        # 遍历每张输入图像及其 MFD 结果。
        for image_index in range(len(images_mfd_res)):
            # 获取当前图像的 MFD 结果。
            mfd_res = images_mfd_res[image_index]
            # 获取当前原始图像（numpy 数组）。
            np_array_image = images[image_index]
            # 初始化列表，存储当前图像的公式信息。
            formula_list = []
            # 遍历当前图像 MFD 结果中的每个检测框。
            for idx, (xyxy, conf, cla) in enumerate(zip(
                    mfd_res.boxes.xyxy, mfd_res.boxes.conf, mfd_res.boxes.cls
            )):
                # 提取坐标。
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                # 创建公式信息字典。
                new_item = {
                    "category_id": 13 + int(cla.item()), # 类别 ID 加偏移量
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax], # 边界框坐标
                    "score": round(float(conf.item()), 2), # MFD 置信度
                    "latex": "", # 初始化 LaTeX 字符串
                }
                # 添加到当前图像的公式列表。
                formula_list.append(new_item)
                # 裁剪公式图像。
                bbox_img = np_array_image[ymin:ymax, xmin:xmax]
                # 计算公式区域的面积。
                area = (xmax - xmin) * (ymax - ymin)
                # 获取当前公式图像在 mf_image_list 中的索引。
                curr_idx = len(mf_image_list)
                # 将 (面积, 索引, 图像数据) 元组添加到 image_info。
                image_info.append((area, curr_idx, bbox_img))
                # 将裁剪出的公式图像添加到 mf_image_list。
                mf_image_list.append(bbox_img)
            # 将当前图像的公式列表添加到最终结果列表。
            images_formula_list.append(formula_list)
            # 将当前图像的公式信息字典也添加到 backfill_list，保持原始顺序。
            backfill_list += formula_list

        # 按面积稳定排序
        # 根据面积对 image_info 进行升序排序（面积小的在前）。稳定排序保持相同面积元素的原始相对顺序。
        image_info.sort(key=lambda x: x[0])  # sort by area
        # 获取排序后的索引列表。
        sorted_indices = [x[1] for x in image_info]
        # 获取排序后的图像列表。
        sorted_images = [x[2] for x in image_info]

        # 创建结果映射关系
        # 创建一个字典，将排序后的新索引映射回未排序前的原始索引。
        index_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)}

        # 使用排序后的图像创建数据集
        # 使用按面积排序后的公式图像创建 MathDataset。按面积排序可能有助于提高批处理效率（相似大小的图像在同一批次）。
        dataset = MathDataset(sorted_images, transform=self.model.transform)
        # 创建 DataLoader，使用指定的 batch_size 进行批处理。
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        # 处理批次并存储结果
        # 初始化列表，存储按排序后顺序得到的 MFR 结果。
        mfr_res = []
        # 使用 tqdm 创建进度条，总数为排序后的图像数量。
        # for mf_img in dataloader: # 原始循环，没有进度条
        with tqdm(total=len(sorted_images), desc="MFR Predict") as pbar: # 使用tqdm显示MFR预测进度
            # 遍历 DataLoader 提供的按面积排序的批次。
            for index, mf_img in enumerate(dataloader):
                # 转换数据类型和设备。
                mf_img = mf_img.to(dtype=self.model.dtype)
                mf_img = mf_img.to(self.device)
                # 禁用梯度计算进行推理。
                with torch.no_grad():
                    # 模型生成 LaTeX 字符串。
                    output = self.model.generate({"image": mf_img})
                # 将批次结果添加到 mfr_res。
                mfr_res.extend(output["fixed_str"])
                # 更新进度条：计算当前批次的实际大小（最后一个批次可能不足batch_size）。
                current_batch_size = min(batch_size, len(sorted_images) - index * batch_size)
                # 更新进度条。
                pbar.update(current_batch_size)

        # 恢复原始顺序
        # 创建一个与 mfr_res 等长的列表，用于存储按原始顺序排列的 LaTeX 结果。
        unsorted_results = [""] * len(mfr_res)
        # 遍历排序后的结果 mfr_res。
        for new_idx, latex in enumerate(mfr_res):
            # 使用 index_mapping 找到当前结果对应的原始索引。
            original_idx = index_mapping[new_idx]
            # 将 LaTeX 结果存放到 unsorted_results 中对应的原始位置。
            unsorted_results[original_idx] = latex

        # 回填结果
        # 遍历 backfill_list（包含按原始顺序排列的所有公式字典）和 unsorted_results（包含按原始顺序排列的 LaTeX 结果）。
        for res, latex in zip(backfill_list, unsorted_results):
            # 将对应的 LaTeX 结果填充回公式字典的 "latex" 键。
            res["latex"] = latex

        # 返回包含所有图像公式列表（已填充 LaTeX）的列表。
        return images_formula_list