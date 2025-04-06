# batch_build_dataset.py
import concurrent.futures # 导入并发库，用于并行处理
import fitz # 导入 PyMuPDF 库，用于处理 PDF 文件

from magic_pdf.data.dataset import PymuDocDataset # 从项目中导入 PymuDocDataset 类
from magic_pdf.data.utils import fitz_doc_to_image  # 从项目中导入 PDF 页面转图片的工具函数 (基于 PyMuPDF)


def partition_array_greedy(arr, k):
    """使用简单的贪心方法将数组划分为 k 个部分。

    参数:
    -----------
    arr : list
        输入的整数数组（实际在此代码中是包含页面数的元组列表）
    k : int
        要创建的分区数量

    返回:
    --------
    partitions : list of lists
        数组的 k 个分区，每个分区包含原始数组元素的索引
    """
    # 处理边界情况
    if k <= 0:
        # k 必须是正整数，否则抛出值错误
        raise ValueError('k must be a positive integer')
    if k > len(arr):
        # 如果 k 大于数组长度，则将 k 调整为数组长度（每个元素自成一个分区）
        k = len(arr)
    if k == 1:
        # 如果 k 为 1，则所有元素都在一个分区中
        return [list(range(len(arr)))]
    if k == len(arr):
        # 如果 k 等于数组长度，则每个元素自成一个分区
        return [[i] for i in range(len(arr))]

    # 按降序对数组进行排序（根据元组的第二个元素，即页面数）
    # sorted_indices 存储的是排序后的原始索引
    sorted_indices = sorted(range(len(arr)), key=lambda i: arr[i][1], reverse=True)

    # 初始化 k 个空分区
    partitions = [[] for _ in range(k)]
    # 初始化 k 个分区的当前总和（页面数总和）
    partition_sums = [0] * k

    # 将每个元素分配给当前总和最小的分区
    for idx in sorted_indices:
        # 找到总和最小的分区的索引
        min_sum_idx = partition_sums.index(min(partition_sums))

        # 将元素的原始索引添加到这个分区
        partitions[min_sum_idx].append(idx)
        # 更新该分区的总和
        partition_sums[min_sum_idx] += arr[idx][1]

    # 返回划分好的分区列表
    return partitions


def process_pdf_batch(pdf_jobs, idx):
    """使用多线程处理一批 PDF 页面。(注意：实际此函数内没有使用多线程，而是串行处理批次内的PDF)

    参数:
    -----------
    pdf_jobs : list of tuples
        (pdf_path, page_num) 元组的列表 (page_num 在此函数中未被使用)
    idx : int
        批次的索引，用于结果返回时标识

    返回:
    --------
    tuple : (int, list)
        返回一个元组，包含批次索引和处理后的图像列表
        图像列表是一个嵌套列表，外层列表对应批次中的PDF，内层列表对应每个PDF的所有页面图像
    """
    # 初始化用于存储该批次所有图像的列表
    images = []

    # 遍历批次中的每个 PDF 任务
    for pdf_path, _ in pdf_jobs:
        # 使用 fitz 打开 PDF 文件
        doc = fitz.open(pdf_path)
        # 初始化用于存储当前 PDF 所有页面图像的临时列表
        tmp = []
        # 遍历当前 PDF 的所有页面
        for page_num in range(len(doc)):
            # 获取当前页面对象
            page = doc[page_num]
            # 将页面转换为图像并添加到临时列表
            tmp.append(fitz_doc_to_image(page))
        # 将当前 PDF 的所有页面图像列表添加到总图像列表中
        images.append(tmp)
    # 返回批次索引和该批次的图像数据
    return (idx, images)


def batch_build_dataset(pdf_paths, k, lang=None):
    """通过将多个 PDF 文件划分为 k 个均衡的部分，并并行处理每个部分来处理它们。

    参数:
    -----------
    pdf_paths : list
        PDF 文件路径的列表
    k : int
        要创建的分区数量（也是并行处理的工作进程数）
    lang : str or None, optional
        语言参数，用于 PymuDocDataset 初始化 (例如 'zh', 'en', 'auto', None)
        默认为 None.

    返回:
    --------
    results : list
        所有处理后的 PymuDocDataset 对象的列表，顺序与输入的 pdf_paths 对应
    """
    # 获取每个 PDF 的页面数信息
    pdf_info = []
    # 初始化总页面数
    total_pages = 0

    # 遍历输入的 PDF 路径列表
    for pdf_path in pdf_paths:
        try:
            # 打开 PDF 文件
            doc = fitz.open(pdf_path)
            # 获取 PDF 的页面数
            num_pages = len(doc)
            # 将 (PDF路径, 页面数) 元组添加到 pdf_info 列表中
            pdf_info.append((pdf_path, num_pages))
            # 累加总页面数
            total_pages += num_pages
            # 关闭 PDF 文档
            doc.close()
        except Exception as e:
            # 如果打开文件出错，打印错误信息
            print(f'Error opening {pdf_path}: {e}')

    # 根据页面数对任务进行分区 (每个任务是一个 PDF 文件)
    partitions = partition_array_greedy(pdf_info, k)

    # 初始化一个字典来存储按分区索引的图像结果
    all_images_h = {}

    # 使用进程池执行器并行处理每个分区
    # max_workers=k 表示最多同时运行 k 个进程
    with concurrent.futures.ProcessPoolExecutor(max_workers=k) as executor:
        # 初始化用于存储 future 对象的列表
        futures = []
        # 遍历每个分区及其索引 (sn)
        for sn, partition in enumerate(partitions):
            # 获取当前分区对应的任务（(pdf_path, page_count) 元组列表）
            partition_jobs = [pdf_info[idx] for idx in partition]

            # 提交任务到进程池执行 process_pdf_batch 函数
            future = executor.submit(
                process_pdf_batch, # 要执行的函数
                partition_jobs,    # 传递给函数的第一个参数 (该分区的任务)
                sn                 # 传递给函数的第二个参数 (分区索引)
            )
            # 将 future 对象添加到列表中
            futures.append(future)

        # 当任务完成时，处理结果
        # as_completed 会在任何 future 完成时立即返回它
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                # 获取任务的结果 (idx, images)
                idx, images = future.result()
                # 将结果存储在字典中，以分区索引为键
                all_images_h[idx] = images
            except Exception as e:
                # 如果处理分区时出错，打印错误信息
                print(f'Error processing partition: {e}')

    # 初始化最终结果列表，长度与输入 PDF 数量相同，并用 None 填充
    results = [None] * len(pdf_paths)
    # 遍历分区及其索引
    for i in range(len(partitions)):
        # 获取当前分区（包含原始 pdf_info 的索引）
        partition = partitions[i]
        # 遍历当前分区中的每个原始索引 j
        for j in range(len(partition)):
            # 获取该索引对应的 PDF 文件路径
            pdf_path = pdf_info[partition[j]][0]
            # 以二进制读取模式打开 PDF 文件
            with open(pdf_path, 'rb') as f:
                # 读取 PDF 文件的字节内容
                pdf_bytes = f.read()
            # 使用 PDF 字节内容和指定的语言创建 PymuDocDataset 对象
            dataset = PymuDocDataset(pdf_bytes, lang=lang)
            # 设置该 Dataset 对象的图像数据 (从 all_images_h 中获取预处理好的图像)
            # all_images_h[i] 是第 i 个分区的结果 (一个列表，包含该分区所有PDF的图像)
            # all_images_h[i][j] 是该分区中第 j 个 PDF 的图像列表
            dataset.set_images(all_images_h[i][j])
            # 将创建好的 Dataset 对象放入最终结果列表的正确位置 (根据原始索引 partition[j])
            results[partition[j]] = dataset
    # 返回包含所有 PymuDocDataset 对象的列表
    return results
