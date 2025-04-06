# dataset.py
import os # 导入操作系统接口模块
from abc import ABC, abstractmethod # 从 abc 模块导入 ABC (抽象基类) 和 abstractmethod (抽象方法装饰器)
from typing import Callable, Iterator # 导入类型提示相关的 Callable (可调用对象) 和 Iterator (迭代器)

import fitz # 导入 PyMuPDF 库
from loguru import logger # 导入日志库 loguru

from magic_pdf.config.enums import SupportedPdfParseMethod # 从项目中导入支持的 PDF 解析方法枚举
from magic_pdf.data.schemas import PageInfo # 从项目中导入页面信息模式类
from magic_pdf.data.utils import fitz_doc_to_image # 从项目中导入 PDF 页面转图片的工具函数
from magic_pdf.filter import classify # 从项目中导入文档分类函数


# 定义一个抽象基类 PageableData，代表可分页的数据（例如 PDF 的一页）
class PageableData(ABC):
    @abstractmethod
    def get_image(self) -> dict:
        """将数据转换为图像。"""
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def get_doc(self) -> fitz.Page:
        """获取 pymudoc 页面对象。"""
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def get_page_info(self) -> PageInfo:
        """获取页面的页面信息。

        返回:
            PageInfo: 此页面的页面信息对象
        """
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def draw_rect(self, rect_coords, color, fill, fill_opacity, width, overlay):
        """绘制矩形。

        参数:
            rect_coords (list[float]): 包含左上角和右下角坐标的四个元素的数组, [x0, y0, x1, y1]
            color (list[float] | None): 描述边框线条 RGB 的三元素元组，None 表示无边框线条
            fill (list[float] | None): 使用 RGB 填充边框，None 表示不填充颜色
            fill_opacity (float): 填充的不透明度，范围从 [0, 1]
            width (float): 边框的宽度
            overlay (bool): 在前景或背景中填充颜色。True 表示在背景中填充。
        """
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def insert_text(self, coord, content, fontsize, color):
        """插入文本。

        参数:
            coord (list[float]): 包含左上角和右下角坐标的四个元素的数组, [x0, y0, x1, y1]（通常指插入点或文本框）
            content (str): 文本内容
            fontsize (int): 文本的字体大小
            color (list[float] | None): 描述文本 RGB 的三元素元组，None 将使用默认字体颜色！
        """
        pass # 抽象方法，子类必须实现


# 定义一个抽象基类 Dataset，代表一个数据集（例如一个完整的 PDF 文档或一组图片）
class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """数据集的长度（例如 PDF 的页数）。"""
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def __iter__(self) -> Iterator[PageableData]:
        """产生页面数据（迭代器）。"""
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def supported_methods(self) -> list[SupportedPdfParseMethod]:
        """此数据集支持的方法。

        返回:
            list[SupportedPdfParseMethod]: 支持的方法列表，有效方法为：OCR, TXT
        """
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def data_bits(self) -> bytes:
        """用于创建此数据集的原始数据（字节）。"""
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def get_page(self, page_id: int) -> PageableData:
        """获取由 page_id 索引的页面。

        参数:
            page_id (int): 页面的索引

        返回:
            PageableData: 页面文档对象
        """
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def dump_to_file(self, file_path: str):
        """将文件转储到指定路径。

        参数:
            file_path (str): 文件路径
        """
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def apply(self, proc: Callable, *args, **kwargs):
        """应用一个可调用方法。

        参数:
            proc (Callable): 按如下方式调用 proc：
                proc(self, *args, **kwargs)

        返回:
            Any: 返回由 proc 生成的结果
        """
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def classify(self) -> SupportedPdfParseMethod:
        """对数据集进行分类（判断是扫描型还是文本型等）。

        返回:
            SupportedPdfParseMethod: 分类结果枚举值
        """
        pass # 抽象方法，子类必须实现

    @abstractmethod
    def clone(self):
        """克隆此数据集。"""
        pass # 抽象方法，子类必须实现


# PymuDocDataset 类，继承自 Dataset，用于处理 PDF 文件
class PymuDocDataset(Dataset):
    def __init__(self, bits: bytes, lang=None):
        """初始化数据集，该数据集包装了 pymudoc 文档。

        参数:
            bits (bytes): PDF 文件的字节内容
            lang (str or None, optional): 指定语言或自动检测 ('zh', 'en', 'auto', None)
        """
        # 使用 fitz 从字节流打开 PDF 文档
        self._raw_fitz = fitz.open('pdf', bits)
        # 为 PDF 的每一页创建一个 Doc 对象，并存储在 _records 列表中
        self._records = [Doc(v) for v in self._raw_fitz]
        # 存储用于创建此数据集的原始 PDF 字节
        self._data_bits = bits
        # 存储原始输入数据（用于克隆）
        self._raw_data = bits
        # 初始化分类结果为 None
        self._classify_result = None

        # 处理语言参数
        if lang == '':
            # 如果 lang 为空字符串，则设为 None
            self._lang = None
        elif lang == 'auto':
            # 如果 lang 为 'auto'，则导入并使用自动语言检测工具
            from magic_pdf.model.sub_modules.language_detection.utils import \
                auto_detect_lang
            self._lang = auto_detect_lang(bits)
            # 记录检测到的语言
            logger.info(f'lang: {lang}, detect_lang: {self._lang}')
        else:
            # 否则，直接使用传入的 lang
            self._lang = lang
            logger.info(f'lang: {lang}')

    def __len__(self) -> int:
        """返回 PDF 的页数。"""
        return len(self._records)

    def __iter__(self) -> Iterator[PageableData]:
        """返回页面 Doc 对象的迭代器。"""
        return iter(self._records)

    def supported_methods(self) -> list[SupportedPdfParseMethod]:
        """此数据集支持的方法（PDF 可以是文本型或扫描型）。

        返回:
            list[SupportedPdfParseMethod]: 支持的方法列表 [OCR, TXT]
        """
        return [SupportedPdfParseMethod.OCR, SupportedPdfParseMethod.TXT]

    def data_bits(self) -> bytes:
        """返回用于创建此数据集的 PDF 字节。"""
        return self._data_bits

    def get_page(self, page_id: int) -> PageableData:
        """获取指定索引的页面 Doc 对象。

        参数:
            page_id (int): 页面索引

        返回:
            PageableData: 页面 Doc 对象
        """
        return self._records[page_id]

    def dump_to_file(self, file_path: str):
        """将内部的 fitz 文档保存到文件。

        参数:
            file_path (str): 要保存的文件路径
        """
        # 获取文件路径的目录名
        dir_name = os.path.dirname(file_path)
        # 如果目录名不是空、当前目录或上级目录，则创建目录（如果不存在）
        if dir_name not in ('', '.', '..'):
            os.makedirs(dir_name, exist_ok=True)
        # 保存 fitz 文档到指定路径
        self._raw_fitz.save(file_path)

    def apply(self, proc: Callable, *args, **kwargs):
        """应用一个可调用方法到数据集自身。

        参数:
            proc (Callable): 按如下方式调用 proc：
                proc(dataset, *args, **kwargs)
            *args, **kwargs: 传递给 proc 的额外参数

        返回:
            Any: 返回由 proc 生成的结果
        """
        # 如果调用时提供了 lang 参数且数据集有确定的语言，则用数据集的语言覆盖传入的 lang
        if 'lang' in kwargs and self._lang is not None:
            kwargs['lang'] = self._lang
        # 调用 proc，并将数据集实例 (self) 作为第一个参数传递
        return proc(self, *args, **kwargs)

    def classify(self) -> SupportedPdfParseMethod:
        """对数据集进行分类（如果尚未分类）。

        返回:
            SupportedPdfParseMethod: 分类结果 (OCR 或 TXT)
        """
        # 如果尚未进行分类
        if self._classify_result is None:
            # 调用外部的 classify 函数进行分类，并存储结果
            self._classify_result = classify(self._data_bits)
        # 返回存储的分类结果
        return self._classify_result

    def clone(self):
        """克隆此数据集。"""
        # 使用原始数据字节创建一个新的 PymuDocDataset 实例
        return PymuDocDataset(self._raw_data)

    def set_images(self, images):
        """为数据集中的每个页面设置预渲染的图像。

        参数:
            images (list): 包含每个页面图像数据的列表
        """
        # 遍历所有记录（页面）
        for i in range(len(self._records)):
            # 为每个页面 Doc 对象设置对应的图像
            self._records[i].set_image(images[i])

# ImageDataset 类，继承自 Dataset，用于处理图像文件
class ImageDataset(Dataset):
    def __init__(self, bits: bytes):
        """初始化数据集，该数据集包装了 pymudoc 文档。

        参数:
            bits (bytes): 将首先转换为 PDF，然后转换为 pymudoc 的照片字节。
        """
        # 使用 fitz 将输入的图像字节流转换为 PDF 字节流
        pdf_bytes = fitz.open(stream=bits).convert_to_pdf()
        # 使用转换后的 PDF 字节流打开 fitz 文档
        self._raw_fitz = fitz.open('pdf', pdf_bytes)
        # 为（现在是 PDF 的）每一页创建 Doc 对象
        self._records = [Doc(v) for v in self._raw_fitz]
        # 存储原始输入的图像字节（用于克隆）
        self._raw_data = bits
        # 存储转换后的 PDF 字节
        self._data_bits = pdf_bytes

    def __len__(self) -> int:
        """数据集的长度（通常为 1，因为输入是单个图像）。"""
        return len(self._records)

    def __iter__(self) -> Iterator[PageableData]:
        """产生页面 Doc 对象的迭代器。"""
        return iter(self._records)

    def supported_methods(self):
        """此数据集支持的方法（图像只能通过 OCR 处理）。

        返回:
            list[SupportedPdfParseMethod]: 支持的方法列表 [OCR]
        """
        return [SupportedPdfParseMethod.OCR]

    def data_bits(self) -> bytes:
        """返回用于创建此数据集的 PDF 字节（转换后的）。"""
        return self._data_bits

    def get_page(self, page_id: int) -> PageableData:
        """获取指定索引的页面 Doc 对象。

        参数:
            page_id (int): 页面索引

        返回:
            PageableData: 页面 Doc 对象
        """
        return self._records[page_id]

    def dump_to_file(self, file_path: str):
        """将内部的 fitz 文档（转换后的 PDF）保存到文件。

        参数:
            file_path (str): 要保存的文件路径
        """
        # 获取文件路径的目录名
        dir_name = os.path.dirname(file_path)
        # 如果目录名不是空、当前目录或上级目录，则创建目录（如果不存在）
        if dir_name not in ('', '.', '..'):
            os.makedirs(dir_name, exist_ok=True)
        # 保存 fitz 文档到指定路径
        self._raw_fitz.save(file_path)

    def apply(self, proc: Callable, *args, **kwargs):
        """应用一个可调用方法到数据集自身。

        参数:
            proc (Callable): 按如下方式调用 proc：
                proc(dataset, *args, **kwargs)
            *args, **kwargs: 传递给 proc 的额外参数

        返回:
            Any: 返回由 proc 生成的结果
        """
        # 调用 proc，并将数据集实例 (self) 作为第一个参数传递
        return proc(self, *args, **kwargs)

    def classify(self) -> SupportedPdfParseMethod:
        """对数据集进行分类（图像数据集总是 OCR）。

        返回:
            SupportedPdfParseMethod: 分类结果 (OCR)
        """
        return SupportedPdfParseMethod.OCR

    def clone(self):
        """克隆此数据集。"""
        # 使用原始图像数据字节创建一个新的 ImageDataset 实例
        return ImageDataset(self._raw_data)

    def set_images(self, images):
        """为数据集中的每个页面设置预渲染的图像。

        参数:
            images (list): 包含每个页面图像数据的列表
        """
        # 遍历所有记录（页面）
        for i in range(len(self._records)):
            # 为每个页面 Doc 对象设置对应的图像
            self._records[i].set_image(images[i])

# Doc 类，继承自 PageableData，代表 PDF 中的单个页面
class Doc(PageableData):
    """使用 pymudoc 页面对象初始化。"""

    def __init__(self, doc: fitz.Page):
        # 存储传入的 fitz.Page 对象
        self._doc = doc
        # 初始化图像缓存为 None
        self._img = None

    def get_image(self):
        """返回图像信息。

        返回:
            dict: {
                img: np.ndarray, # 图像的 numpy 数组
                width: int,      # 图像宽度
                height: int      # 图像高度
            }
        """
        # 如果图像尚未生成或设置
        if self._img is None:
            # 调用工具函数将 fitz.Page 转换为图像字典
            self._img = fitz_doc_to_image(self._doc)
        # 返回缓存的图像字典
        return self._img

    def set_image(self, img):
        """设置页面的图像。
        Args:
            img (dict): 包含图像信息的字典，结构同 get_image 返回值
                      {'img': np.ndarray, 'width': int, 'height': int}
        """
        # 仅在当前没有缓存图像时设置
        if self._img is None:
            self._img = img

    def get_doc(self) -> fitz.Page:
        """获取 pymudoc 页面对象。

        返回:
            fitz.Page: pymudoc 页面对象
        """
        return self._doc

    def get_page_info(self) -> PageInfo:
        """获取页面的页面信息。

        返回:
            PageInfo: 此页面的页面信息对象
        """
        # 获取页面的宽度和高度
        page_w = self._doc.rect.width
        page_h = self._doc.rect.height
        # 创建并返回 PageInfo 对象
        return PageInfo(w=page_w, h=page_h)

    def __getattr__(self, name):
        """使 Doc 对象能够直接访问其底层 _doc (fitz.Page) 对象的属性和方法。"""
        # 如果底层 _doc 对象有该属性/方法
        if hasattr(self._doc, name):
            # 返回该属性/方法
            return getattr(self._doc, name)
        # 否则，按正常方式抛出 AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    def draw_rect(self, rect_coords, color, fill, fill_opacity, width, overlay):
        """在页面上绘制矩形。

        参数:
            rect_coords (list[float]): 包含左上角和右下角坐标的四个元素的数组, [x0, y0, x1, y1]
            color (list[float] | None): 描述边框线条 RGB 的三元素元组，None 表示无边框线条
            fill (list[float] | None): 使用 RGB 填充边框，None 表示不填充颜色
            fill_opacity (float): 填充的不透明度，范围从 [0, 1]
            width (float): 边框的宽度
            overlay (bool): 在前景或背景中填充颜色。True 表示在背景中填充。
        """
        # 调用底层 fitz.Page 对象的 draw_rect 方法
        self._doc.draw_rect(
            rect_coords,
            color=color,
            fill=fill,
            fill_opacity=fill_opacity,
            width=width,
            overlay=overlay,
        )

    def insert_text(self, coord, content, fontsize, color):
        """在页面上插入文本。

        参数:
            coord (list[float]): 包含左上角坐标的二元素数组或插入点的 Point 对象, [x0, y0] 或 fitz.Point(x, y)
            content (str): 要插入的文本内容
            fontsize (int): 文本的字体大小
            color (list[float] | None): 描述文本 RGB 的三元素元组，None 将使用默认字体颜色！
        """
        # 调用底层 fitz.Page 对象的 insert_text 方法
        # 注意：fitz.Page.insert_text 的 'coord' 参数通常是插入点 (Point 或 [x, y])，而不是矩形。
        # 如果传入的是矩形坐标，可能需要调整为插入点。这里假设传入的是兼容的坐标。
        self._doc.insert_text(coord, content, fontsize=fontsize, color=color)
