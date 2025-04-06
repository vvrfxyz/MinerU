# read_api.py
import json # 导入 JSON 处理库
import os # 导入操作系统接口模块
import tempfile # 导入临时文件/目录创建模块
import shutil # 导入高级文件操作模块（如删除目录树）
from pathlib import Path # 导入面向对象的文件系统路径模块

from magic_pdf.config.exceptions import EmptyData, InvalidParams # 从项目中导入自定义异常类
from magic_pdf.data.data_reader_writer import (FileBasedDataReader, # 导入基于文件的数据读取器
                                               MultiBucketS3DataReader) # 导入支持多存储桶的 S3 数据读取器
from magic_pdf.data.dataset import ImageDataset, PymuDocDataset # 导入数据集类
from magic_pdf.utils.office_to_pdf import convert_file_to_pdf, ConvertToPdfError # 导入 Office 转 PDF 的工具函数和错误类

def read_jsonl(
    s3_path_or_local: str, s3_client: MultiBucketS3DataReader | None = None
) -> list[PymuDocDataset]:
    """读取 jsonl 文件并返回 PymuDocDataset 列表。
       JSONL 文件每行是一个 JSON 对象，包含 PDF 文件的位置信息。

    参数:
        s3_path_or_local (str): 本地文件路径或 S3 路径 (例如 's3://bucket/path/to/file.jsonl')
        s3_client (MultiBucketS3DataReader | None, optional): 支持多存储桶的 S3 客户端。
                                                               如果 s3_path_or_local 是 S3 路径，则必须提供此参数。
                                                               默认为 None.

    异常:
        InvalidParams: 如果 s3_path_or_local 是 S3 路径但未提供 s3_client。
        EmptyData: 如果 jsonl 文件某行中未提供 PDF 文件位置。
        InvalidParams: 如果 PDF 文件位置是 S3 路径但未提供 s3_client。

    返回:
        list[PymuDocDataset]: jsonl 文件中的每一行（对应的PDF）将被转换为一个 PymuDocDataset 对象。
    """
    # 用于存储读取到的 PDF 文件字节内容的列表
    bits_arr = []
    # 检查输入路径是否是 S3 路径
    if s3_path_or_local.startswith('s3://'):
        # 如果是 S3 路径但未提供 S3 客户端，则抛出异常
        if s3_client is None:
            raise InvalidParams('s3_client is required when s3_path is provided')
        # 使用 S3 客户端读取 JSONL 文件内容
        jsonl_bits = s3_client.read(s3_path_or_local)
    else:
        # 如果是本地路径，使用文件读取器读取 JSONL 文件内容
        jsonl_bits = FileBasedDataReader('').read(s3_path_or_local)

    # 解析 JSONL 文件内容
    # 1. 解码字节内容为字符串
    # 2. 按换行符分割成行
    # 3. 过滤掉空行
    # 4. 将每行解析为 JSON 对象
    jsonl_d = [
        json.loads(line) for line in jsonl_bits.decode().split('\n') if line.strip()
    ]

    # 遍历解析后的 JSON 对象列表
    for d in jsonl_d:
        # 获取 PDF 文件路径，尝试 'file_location' 和 'path' 两个键
        pdf_path = d.get('file_location', '') or d.get('path', '')
        # 如果 PDF 路径为空，则抛出异常
        if len(pdf_path) == 0:
            raise EmptyData('pdf file location is empty')
        # 检查 PDF 路径是否是 S3 路径
        if pdf_path.startswith('s3://'):
            # 如果是 S3 路径但未提供 S3 客户端，则抛出异常
            if s3_client is None:
                raise InvalidParams('s3_client is required when s3_path is provided')
            # 使用 S3 客户端读取 PDF 文件字节内容并添加到列表
            bits_arr.append(s3_client.read(pdf_path))
        else:
            # 如果是本地路径，使用文件读取器读取 PDF 文件字节内容并添加到列表
            bits_arr.append(FileBasedDataReader('').read(pdf_path))
    # 将每个 PDF 文件的字节内容转换为 PymuDocDataset 对象，并返回列表
    return [PymuDocDataset(bits) for bits in bits_arr]


def read_local_pdfs(path: str) -> list[PymuDocDataset]:
    """从本地路径（文件或目录）读取 PDF 文件。

    参数:
        path (str): PDF 文件路径或包含 PDF 文件的目录路径

    返回:
        list[PymuDocDataset]: 每个 PDF 文件将被转换为一个 PymuDocDataset 对象。
    """
    # 初始化文件读取器
    reader = FileBasedDataReader()
    # 初始化结果列表
    ret = []
    # 检查路径是否是一个目录
    if os.path.isdir(path):
        # 遍历目录及其子目录
        for root, _, files in os.walk(path):
            # 遍历当前目录下的文件
            for file in files:
                # 获取文件后缀名
                suffix = file.split('.')
                # 如果后缀是 'pdf'
                if suffix[-1].lower() == 'pdf': # 转小写以兼容 .PDF
                    # 读取 PDF 文件字节内容并创建 PymuDocDataset 对象，添加到结果列表
                    ret.append( PymuDocDataset(reader.read(os.path.join(root, file))))
    else:
        # 如果路径是单个文件
        # 读取文件字节内容
        bits = reader.read(path)
        # 创建 PymuDocDataset 对象并添加到结果列表（只有一个元素）
        ret = [PymuDocDataset(bits)]
    # 返回结果列表
    return ret

def read_local_office(path: str) -> list[PymuDocDataset]:
    """从本地路径（文件或目录）读取 MS Office 文件 (ppt, pptx, doc, docx)。
       这些文件将被转换为 PDF 后再处理。

    参数:
        path (str): MS Office 文件路径或包含这些文件的目录路径

    返回:
        list[PymuDocDataset]: 每个 MS Office 文件将被转换为一个 PymuDocDataset 对象。

    异常:
        ConvertToPdfError: 通过 libreoffice 将 MS Office 文件转换为 PDF 失败。
        FileNotFoundError: 文件未找到。
        Exception: 引发未知异常。
    """
    # 定义支持的 Office 文件后缀
    suffixes = ['.ppt', '.pptx', '.doc', '.docx']
    # 初始化存储找到的 Office 文件路径的列表
    fns = []
    # 初始化结果列表
    ret = []
    # 检查路径是否是一个目录
    if os.path.isdir(path):
        # 遍历目录及其子目录
        for root, _, files in os.walk(path):
            # 遍历当前目录下的文件
            for file in files:
                # 获取文件后缀名（使用 pathlib）
                suffix = Path(file).suffix.lower() # 转小写
                # 如果文件后缀在支持的列表中
                if suffix in suffixes:
                    # 将完整文件路径添加到 fns 列表
                    fns.append((os.path.join(root, file)))
    else:
        # 如果路径是单个文件，直接添加到 fns 列表
        fns.append(path)

    # 初始化文件读取器
    reader = FileBasedDataReader()
    # 创建一个临时目录用于存放转换后的 PDF 文件
    temp_dir = tempfile.mkdtemp()
    try: # 使用 try...finally 确保临时目录会被清理
        # 遍历找到的所有 Office 文件路径
        for fn in fns:
            try:
                # 调用工具函数将 Office 文件转换为 PDF，输出到临时目录
                convert_file_to_pdf(fn, temp_dir)
            except ConvertToPdfError as e:
                # 捕获并重新抛出转换错误
                logger.error(f"转换 Office 文件失败: {fn}, 错误: {e}")
                raise e
            except FileNotFoundError as e:
                # 捕获并重新抛出文件未找到错误
                logger.error(f"Office 文件未找到: {fn}, 错误: {e}")
                raise e
            except Exception as e:
                # 捕获并重新抛出其他未知异常
                logger.error(f"处理 Office 文件时发生未知错误: {fn}, 错误: {e}")
                raise e
            # 构建转换后的 PDF 文件路径
            fn_path = Path(fn)
            pdf_fn = f"{temp_dir}/{fn_path.stem}.pdf"
            # 读取转换后的 PDF 文件字节内容，创建 PymuDocDataset 对象，并添加到结果列表
            ret.append(PymuDocDataset(reader.read(pdf_fn)))
    finally:
        # 无论是否发生异常，都删除临时目录及其内容
        shutil.rmtree(temp_dir)
    # 返回结果列表
    return ret

def read_local_images(path: str, suffixes: list[str]=['.png', '.jpg']) -> list[ImageDataset]:
    """从本地路径（文件或目录）读取图像文件。

    参数:
        path (str): 图像文件路径或包含图像文件的目录路径
        suffixes (list[str], optional): 用于过滤文件的图像文件后缀列表。
                                         示例: ['.jpg', '.png']。默认为 ['.png', '.jpg']。

    返回:
        list[ImageDataset]: 每个图像文件将被转换为一个 ImageDataset 对象。
    """
    # 初始化存储图像文件字节内容的列表
    imgs_bits = []
    # 将后缀列表转换为集合以便快速查找，并转为小写
    s_suffixes = set(s.lower() for s in suffixes)
    # 初始化文件读取器
    reader = FileBasedDataReader()
    # 检查路径是否是一个目录
    if os.path.isdir(path):
        # 遍历目录及其子目录
        for root, _, files in os.walk(path):
            # 遍历当前目录下的文件
            for file in files:
                # 获取文件后缀名（使用 pathlib），并转小写
                suffix = Path(file).suffix.lower()
                # 如果文件后缀在支持的后缀集合中
                if suffix in s_suffixes:
                    # 读取图像文件字节内容并添加到列表
                    imgs_bits.append(reader.read(os.path.join(root, file)))
    else:
        # 如果路径是单个文件
        # 获取文件后缀并检查是否支持 (虽然对于单个文件可能不太必要，但保持一致性)
        suffix = Path(path).suffix.lower()
        if suffix in s_suffixes:
            # 读取文件字节内容
            bits = reader.read(path)
            # 添加到列表
            imgs_bits.append(bits)
        else:
             logger.warning(f"文件 {path} 的后缀 {suffix} 不在支持列表 {suffixes} 中，将被忽略。")


    # 将每个图像文件的字节内容转换为 ImageDataset 对象，并返回列表
    return [ImageDataset(bits) for bits in imgs_bits]

