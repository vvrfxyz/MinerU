# 导入 copy 模块，用于深拷贝对象
import copy
# 导入 json 模块，用于处理 JSON 数据
import json
# 导入 os 模块，用于与操作系统交互，如处理文件路径
import os
# 从 typing 模块导入 Callable 类型提示，用于表示可调用对象
from typing import Callable
# 从配置文件中导入 DropMode 和 MakeMode 枚举，用于控制内容生成模式
from magic_pdf.config.make_content_config import DropMode, MakeMode
# 导入数据读写器类
from magic_pdf.data.data_reader_writer import DataWriter
# 导入数据集类
from magic_pdf.data.dataset import Dataset
# 导入 Markdown 内容生成函数
from magic_pdf.dict2md.ocr_mkcontent import union_make
# 导入用于绘制边界框的函数 (布局框，行排序框，Span框)
from magic_pdf.libs.draw_bbox import (draw_layout_bbox, draw_line_sort_bbox,
                                      draw_span_bbox)
# 导入 JSON 压缩器类
from magic_pdf.libs.json_compressor import JsonCompressor

# 定义管道处理结果类，封装处理结果并提供常用操作方法
class PipeResult:
    # 初始化方法
    def __init__(self, pipe_res, dataset: Dataset):
        """初始化 PipeResult 对象。

        Args:
            pipe_res (list[dict]): 模型推理结果经过整个处理管道后的最终结果。
            dataset (Dataset): 与 pipe_res 相关联的数据集对象，可能包含原始图像等信息。
        """
        # 存储管道处理结果，通常是一个包含页面信息的列表或字典
        self._pipe_res = pipe_res
        # 存储关联的数据集对象
        self._dataset = dataset

    # 获取 Markdown 格式内容的方法
    def get_markdown(
        self,
        img_dir_or_bucket_prefix: str, # 指定图片存储的本地目录或S3存储桶前缀
        drop_mode=DropMode.NONE, # 页面丢弃模式，默认不丢弃任何页面
        md_make_mode=MakeMode.MM_MD, # Markdown 生成模式，默认为标准 Markdown 格式
    ) -> str:
        """根据管道处理结果生成 Markdown 格式的内容。

        Args:
            img_dir_or_bucket_prefix (str): 用于存储从PDF中提取的图片的 S3 存储桶前缀或本地文件目录路径。Markdown中图片的引用路径会基于此。
            drop_mode (str, optional): 当遇到损坏或不合适的页面时的处理策略（如跳过）。默认为 DropMode.NONE (不丢弃)。
            md_make_mode (str, optional): 指定要生成的 Markdown 内容的类型或格式。默认为 MakeMode.MM_MD (标准 Markdown)。

        Returns:
            str: 返回生成的完整 Markdown 内容字符串。
        """
        # 从管道结果中提取核心的PDF信息列表
        pdf_info_list = self._pipe_res['pdf_info']
        # 调用 union_make 函数，传入页面信息、生成模式、丢弃策略和图片路径，生成 Markdown
        md_content = union_make(
            pdf_info_list, md_make_mode, drop_mode, img_dir_or_bucket_prefix
        )
        # 返回生成的 Markdown 字符串
        return md_content

    # 将生成的 Markdown 内容写入文件的方法
    def dump_md(
        self,
        writer: DataWriter, # 数据写入器实例，用于写入文件
        file_path: str, # 要保存 Markdown 文件的完整路径
        img_dir_or_bucket_prefix: str, # 图片存储路径，传递给 get_markdown
        drop_mode=DropMode.NONE, # 页面丢弃模式，传递给 get_markdown
        md_make_mode=MakeMode.MM_MD, # Markdown 生成模式，传递给 get_markdown
    ):
        """将生成的 Markdown 内容转储（写入）到指定文件。

        Args:
            writer (DataWriter): 用于执行文件写入操作的数据写入器对象。
            file_path (str): Markdown 文件要保存的目标路径。
            img_dir_or_bucket_prefix (str): 用于存储图片的 S3 存储桶前缀或本地文件目录。
            drop_mode (str, optional): 页面丢弃策略。默认为 DropMode.NONE。
            md_make_mode (str, optional): Markdown 内容类型。默认为 MakeMode.MM_MD。
        """
        # 首先调用 get_markdown 生成 Markdown 内容
        md_content = self.get_markdown(
            img_dir_or_bucket_prefix, drop_mode=drop_mode, md_make_mode=md_make_mode
        )
        # 使用传入的 writer 对象将 Markdown 内容字符串写入指定的文件路径
        writer.write_string(file_path, md_content)

    # 获取结构化内容列表的方法 (通常是JSON格式)
    def get_content_list(
        self,
        image_dir_or_bucket_prefix: str, # 图片存储路径
        drop_mode=DropMode.NONE, # 页面丢弃模式
    ) -> str: # 注意：虽然方法名是 get_content_list，但返回值类型注解是 str，实际可能返回JSON字符串
        """获取结构化的内容列表（通常以某种标准格式，如JSON）。

        Args:
            image_dir_or_bucket_prefix (str): 用于存储图片的 S3 存储桶前缀或本地文件目录。
            drop_mode (str, optional): 页面丢弃策略。默认为 DropMode.NONE。

        Returns:
            str: 返回结构化内容列表（通常是序列化后的JSON字符串）。
        """
        # 从管道结果中提取核心的PDF信息列表
        pdf_info_list = self._pipe_res['pdf_info']
        # 调用 union_make 函数，但使用 MakeMode.STANDARD_FORMAT 指定生成标准格式的内容列表
        content_list = union_make(
            pdf_info_list,
            MakeMode.STANDARD_FORMAT, # 指定生成模式为标准格式
            drop_mode,
            image_dir_or_bucket_prefix,
        )
        # 返回生成的内容列表 (可能是Python列表/字典，也可能是序列化后的字符串)
        return content_list

    # 将内容列表写入文件的方法
    def dump_content_list(
        self,
        writer: DataWriter, # 数据写入器实例
        file_path: str, # 要保存内容列表文件的路径
        image_dir_or_bucket_prefix: str, # 图片存储路径，传递给 get_content_list
        drop_mode=DropMode.NONE, # 页面丢弃模式，传递给 get_content_list
    ):
        """将结构化内容列表转储（写入）到指定文件（通常存为JSON）。

        Args:
            writer (DataWriter): 用于执行文件写入操作的数据写入器对象。
            file_path (str): 内容列表文件要保存的目标路径。
            image_dir_or_bucket_prefix (str): 用于存储图片的 S3 存储桶前缀或本地文件目录。
            drop_mode (str, optional): 页面丢弃策略。默认为 DropMode.NONE。
        """
        # 首先调用 get_content_list 获取内容列表
        content_list = self.get_content_list(
            image_dir_or_bucket_prefix, drop_mode=drop_mode,
        )
        # 使用 json.dumps 将内容列表序列化为格式化的 JSON 字符串（带缩进，确保UTF-8编码）
        # 然后使用 writer 对象将 JSON 字符串写入指定的文件路径
        writer.write_string(
            file_path, json.dumps(content_list, ensure_ascii=False, indent=4)
        )

    # 获取中间处理结果的 JSON 字符串表示形式的方法
    def get_middle_json(self) -> str:
        """获取整个管道处理结果的 JSON 字符串表示。

        Returns:
            str: 包含管道处理结果的、格式化的 JSON 字符串。
        """
        # 使用 json.dumps 将内部存储的管道结果 (_pipe_res) 序列化为格式化的 JSON 字符串
        # ensure_ascii=False 确保中文字符等能正确显示，indent=4 用于美化输出（4个空格缩进）
        return json.dumps(self._pipe_res, ensure_ascii=False, indent=4)

    # 将中间 JSON 结果写入文件的方法
    def dump_middle_json(self, writer: DataWriter, file_path: str):
        """将管道处理的中间结果（JSON格式）转储到文件。

        Args:
            writer (DataWriter): 文件写入器句柄。
            file_path (str): 中间 JSON 文件要保存的目标路径。
        """
        # 调用 get_middle_json 获取 JSON 字符串
        middle_json = self.get_middle_json()
        # 使用 writer 对象将 JSON 字符串写入指定的文件路径
        writer.write_string(file_path, middle_json)

    # 绘制布局边界框并保存为图片文件的方法
    def draw_layout(self, file_path: str) -> None:
        """在原始页面图像上绘制布局分析结果（边界框）并保存。

        Args:
            file_path (str): 保存绘制结果图像的文件路径（通常不带扩展名，函数内部会处理）。
        """
        # 获取指定路径的目录部分
        dir_name = os.path.dirname(file_path)
        # 获取指定路径的文件名部分
        base_name = os.path.basename(file_path)
        # 检查目录是否存在，如果不存在则创建（包括可能的多级父目录）
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        # 从管道结果中提取 PDF 页面信息
        pdf_info = self._pipe_res['pdf_info']
        # 调用 draw_layout_bbox 函数，传入页面信息、原始图像数据（从dataset获取）、输出目录和基本文件名来绘制和保存图像
        draw_layout_bbox(pdf_info, self._dataset.data_bits(), dir_name, base_name)

    # 绘制 Span 边界框并保存为图片文件的方法
    def draw_span(self, file_path: str):
        """在原始页面图像上绘制 Span（例如文本片段）的边界框并保存。

        Args:
            file_path (str): 保存绘制结果图像的文件路径。
        """
        # 获取目录名
        dir_name = os.path.dirname(file_path)
        # 获取基本文件名
        base_name = os.path.basename(file_path)
        # 确保目录存在
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        # 获取 PDF 页面信息
        pdf_info = self._pipe_res['pdf_info']
        # 调用 draw_span_bbox 函数进行绘制和保存
        draw_span_bbox(pdf_info, self._dataset.data_bits(), dir_name, base_name)

    # 绘制行排序结果并保存为图片文件的方法
    def draw_line_sort(self, file_path: str):
        """在原始页面图像上绘制文本行排序结果（边界框和顺序）并保存。

        Args:
            file_path (str): 保存绘制结果图像的文件路径。
        """
        # 获取目录名
        dir_name = os.path.dirname(file_path)
        # 获取基本文件名
        base_name = os.path.basename(file_path)
        # 确保目录存在
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        # 获取 PDF 页面信息
        pdf_info = self._pipe_res['pdf_info']
        # 调用 draw_line_sort_bbox 函数进行绘制和保存
        draw_line_sort_bbox(pdf_info, self._dataset.data_bits(), dir_name, base_name)

    # 获取压缩后的管道处理结果（JSON）的方法
    def get_compress_pdf_mid_data(self):
        """压缩管道处理的中间结果（JSON）。

        Returns:
            str: 返回压缩后的 JSON 数据字符串。
        """
        # 使用 JsonCompressor 类提供的 compress_json 方法压缩内部存储的管道结果
        return JsonCompressor.compress_json(self._pipe_res)

    # 应用一个自定义处理函数到管道结果上的方法
    def apply(self, proc: Callable, *args, **kwargs): # 使用 *args 和 **kwargs 接收可变参数
        """应用一个外部定义的可调用处理函数 `proc` 到当前的管道结果上。

        Args:
            proc (Callable): 一个可调用对象（如函数），它将接收管道结果作为第一个参数，后面可以跟任意位置参数和关键字参数。
                             调用形式为：proc(pipeline_result_copy, *args, **kwargs)
            *args: 传递给 proc 函数的位置参数。
            **kwargs: 传递给 proc 函数的关键字参数。


        Returns:
            Any: 返回由 `proc` 函数处理后返回的结果。
        """
        # 使用 copy.deepcopy 创建管道结果的一个深拷贝，以防止 proc 修改原始结果
        # 调用传入的 proc 函数，将深拷贝的结果以及接收到的 *args 和 **kwargs 传递给它
        return proc(copy.deepcopy(self._pipe_res), *args, **kwargs)