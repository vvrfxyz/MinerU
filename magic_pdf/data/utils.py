# utils.py
import multiprocessing as mp # 导入多进程模块，并重命名为 mp
import threading # 导入线程模块 (虽然未直接使用 threading.Thread)
from concurrent.futures import (ProcessPoolExecutor, # 从并发期货模块导入进程池执行器
                                ThreadPoolExecutor, # 导入线程池执行器
                                as_completed) # 导入 as_completed 用于获取已完成的 future

import fitz # 导入 PyMuPDF 库
import numpy as np # 导入 NumPy 库，用于处理图像数组
from loguru import logger # 导入日志库


def fitz_doc_to_image(doc, dpi=200) -> dict:
    """将 fitz.Page (PyMuPDF 页面对象) 转换为图像，然后将图像转换为 numpy 数组。

    参数:
        doc (fitz.Page): PyMuPDF 页面对象
        dpi (int, optional): 重置图像的分辨率 (每英寸点数)。默认为 200。

    返回:
        dict: {'img': numpy 数组, 'width': 宽度, 'height': 高度}
    """
    # 创建一个缩放矩阵，用于控制渲染的 DPI
    # fitz 默认 DPI 是 72，所以 scaling = dpi / 72
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    # 使用指定的矩阵获取页面的像素图 (Pixmap)
    # alpha=False 表示不包含透明通道
    pm = doc.get_pixmap(matrix=mat, alpha=False)

    # 如果缩放后的宽度或高度超过 4500 像素，则不进行缩放，使用原始分辨率渲染
    # 这是为了防止生成过大的图像导致内存问题
    if pm.width > 4500 or pm.height > 4500:
        logger.warning(f"页面 {doc.number} 缩放后尺寸过大 ({pm.width}x{pm.height})，将使用原始分辨率渲染。")
        # 使用单位矩阵获取原始分辨率的像素图
        pm = doc.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

    # 将像素图的样本数据 (samples) 直接转换为 numpy 数组
    # pm.samples 是一个 bytes 对象，包含 RGB 或 RGBA 数据
    # dtype=np.uint8 指定数据类型为无符号 8 位整数
    # reshape 根据像素图的高度、宽度和颜色通道数 (3 for RGB) 重塑数组
    img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, 3)

    # 创建包含图像数组、宽度和高度的字典
    img_dict = {'img': img, 'width': pm.width, 'height': pm.height}

    # 返回图像字典
    return img_dict

def load_images_from_pdf(pdf_bytes: bytes, dpi=200, start_page_id=0, end_page_id=None) -> list:
    """从 PDF 字节流加载图像。

    参数:
        pdf_bytes (bytes): PDF 文件的字节内容。
        dpi (int, optional): 图像渲染的 DPI。默认为 200。
        start_page_id (int, optional): 开始加载的页面索引（从 0 开始）。默认为 0。
        end_page_id (int | None, optional): 结束加载的页面索引（包含此页）。
                                           如果为 None 或负数，则加载到最后一页。默认为 None。

    返回:
        list: 包含每个页面图像信息字典的列表。
              即使页面不在指定范围内，列表长度也等于 PDF 总页数，
              范围外的页面对应的字典值为 {'img': [], 'width': 0, 'height': 0}。
    """
    # 初始化存储图像信息的列表
    images = []
    # 使用 fitz 从字节流打开 PDF 文档
    with fitz.open('pdf', pdf_bytes) as doc:
        # 获取 PDF 的总页数
        pdf_page_num = doc.page_count
        # 确定结束页面索引
        # 如果 end_page_id 有效 (非 None 且 >= 0)，则使用它
        # 否则，使用最后一页的索引 (pdf_page_num - 1)
        end_page_id = (
            end_page_id
            if end_page_id is not None and end_page_id >= 0
            else pdf_page_num - 1
        )
        # 如果指定的 end_page_id 超出范围，则修正为最后一页的索引
        if end_page_id > pdf_page_num - 1:
            logger.warning(f'指定的 end_page_id ({end_page_id}) 超出范围，将使用最后一页索引 ({pdf_page_num - 1})')
            end_page_id = pdf_page_num - 1

        # 遍历 PDF 的所有页面索引
        for index in range(0, doc.page_count):
            # 检查当前页面索引是否在指定的 [start_page_id, end_page_id] 范围内
            if start_page_id <= index <= end_page_id:
                # 获取当前页面对象
                page = doc[index]
                # --- 这部分逻辑与 fitz_doc_to_image 函数重复 ---
                # 创建缩放矩阵
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                # 获取像素图
                pm = page.get_pixmap(matrix=mat, alpha=False)
                # 检查尺寸是否过大
                if pm.width > 4500 or pm.height > 4500:
                    logger.warning(f"页面 {index} 缩放后尺寸过大 ({pm.width}x{pm.height})，将使用原始分辨率渲染。")
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                # 转换为 numpy 数组
                img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, 3)
                # 创建图像字典
                img_dict = {'img': img, 'width': pm.width, 'height': pm.height}
                # --- 重复逻辑结束 ---
            else:
                # 如果页面不在指定范围内，创建一个空的图像字典
                img_dict = {'img': [], 'width': 0, 'height': 0}

            # 将当前页面的图像字典添加到列表中
            images.append(img_dict)
    # 返回包含所有页面（或指定范围页面）图像信息的列表
    return images


def convert_page(bytes_page):
    """转换单个 PDF 页面（以字节形式传入）为图像。
       设计用于多进程处理，避免传递复杂的 fitz 对象。

    参数:
        bytes_page (bytes): 单个 PDF 页面的字节内容。

    返回:
        dict: 包含页面图像信息的字典，格式同 fitz_doc_to_image 返回值。
    """
    # 从字节流打开这个单页 PDF
    pdfs = fitz.open('pdf', bytes_page)
    # 获取第一页（也是唯一一页）
    page = pdfs[0]
    # 调用 fitz_doc_to_image 函数进行转换
    result = fitz_doc_to_image(page)
    # 关闭单页 PDF 文档
    pdfs.close()
    # 返回转换结果
    return result

def parallel_process_pdf_safe(pages, num_workers=None, **kwargs):
    """使用序列化安全的方法并行处理 PDF 页面。
       假设 'pages' 是一个包含每个页面字节内容的列表。

    参数:
        pages (list[bytes]): 每个元素是单个 PDF 页面的字节内容。
        num_workers (int | None, optional): 使用的工作进程数。如果为 None，则使用 CPU 核心数。默认为 None。
        **kwargs: 传递给 convert_page (最终是 fitz_doc_to_image) 的额外参数 (例如 dpi)。

    返回:
        list: 处理后的图像信息字典列表，顺序与输入 pages 列表对应。
    """
    # 如果未指定工作进程数，则获取 CPU 核心数
    if num_workers is None:
        num_workers = mp.cpu_count()
    # 限制最大工作进程数，例如可以设置为 mp.cpu_count() 或其他合理值
    num_workers = min(num_workers, mp.cpu_count()) # 确保不超过 CPU 核心数

    # 使用进程池执行器
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 使用 executor.map 并行处理页面字节列表
        # map 会将 pages 列表中的每个元素作为参数传递给 convert_page 函数
        # 它会保持结果的顺序与输入一致
        # 注意：kwargs 不能直接传递给 map 的目标函数，需要使用 functools.partial 或 lambda 包装
        # 但此处的 kwargs 并没有被实际使用到 convert_page 里，如果需要传递 dpi 等参数，需要修改
        results = list(
            executor.map(convert_page, pages)
        )

    # 返回结果列表
    return results


def threaded_process_pdf(pdf_path, num_threads=4, **kwargs):
    """使用多个线程处理单个 PDF 文件的所有页面。

    参数:
    -----------
    pdf_path : str
        PDF 文件的路径
    num_threads : int
        使用的线程数。默认为 4。
    **kwargs :
        传递给 fitz_doc_to_image 的额外参数 (例如 dpi)。

    返回:
    --------
    images : list
        按页面顺序排列的处理后的图像信息字典列表。
        如果某页处理出错，对应位置为 None。
    """
    # 打开 PDF 文件
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"打开 PDF 文件失败: {pdf_path}, 错误: {e}")
        return [] # 返回空列表表示失败

    # 获取 PDF 的总页数
    num_pages = len(doc)

    # 创建一个列表来按顺序存储结果，用 None 初始化
    results = [None] * num_pages

    # 创建线程池执行器
    # max_workers 指定了线程池中的最大线程数
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 使用字典来存储 future 对象和对应的页面编号
        futures = {}
        # 遍历所有页面
        for page_num in range(num_pages):
            # 获取当前页面对象
            page = doc[page_num]
            # 提交任务到线程池：执行 fitz_doc_to_image 函数
            # page 是传递给函数的位置参数
            # **kwargs 是传递给函数的关键字参数 (例如 dpi)
            future = executor.submit(fitz_doc_to_image, page, **kwargs)
            # 将 future 对象和页面编号存储在字典中
            futures[future] = page_num

        # 使用 as_completed 迭代已完成的 future 对象
        # 这允许在任务完成时立即处理结果，而不是等待所有任务完成
        for future in as_completed(futures):
            # 获取完成的 future 对应的页面编号
            page_num = futures[future]
            try:
                # 获取 future 的结果（即 fitz_doc_to_image 的返回值）
                result = future.result()
                # 将结果存储在结果列表的正确位置
                results[page_num] = result
            except Exception as e:
                # 如果处理页面时发生异常
                logger.error(f'处理 PDF "{os.path.basename(pdf_path)}" 的页面 {page_num} 时出错: {e}')
                # 将结果列表中的对应位置保持为 None (或可以设置一个错误标记)
                results[page_num] = None # 保持 None

    # 关闭 PDF 文档
    doc.close()

    # 返回包含所有页面结果的列表
    return results

# 当脚本作为主程序运行时执行以下代码块
if __name__ == '__main__':
    # 示例：打开一个 PDF 文件（路径需要替换为实际存在的 PDF）
    try:
        pdf = fitz.open('/tmp/[MS-DOC].pdf') # 请替换为实际 PDF 路径

        # --- 准备用于 parallel_process_pdf_safe 的数据 ---
        # 创建一个空 fitz 文档列表，数量等于原 PDF 页数
        pdf_page = [fitz.open() for i in range(pdf.page_count)]
        # 将原 PDF 的每一页分别插入到对应的空文档中，制作成单页 PDF 列表
        [pdf_page[i].insert_pdf(pdf, from_page=i, to_page=i) for i in range(pdf.page_count)]

        # 将每个单页 fitz 文档转换为字节流
        pdf_page_bytes = [v.tobytes() for v in pdf_page]
        # 关闭这些临时的单页文档
        [p.close() for p in pdf_page]

        # --- 调用多进程处理函数 ---
        # 使用 16 个工作进程并行处理这些单页 PDF 字节流
        print(f"开始使用 parallel_process_pdf_safe (多进程) 处理 {len(pdf_page_bytes)} 页...")
        import time
        start_time = time.time()
        results_mp = parallel_process_pdf_safe(pdf_page_bytes, num_workers=16)
        end_time = time.time()
        print(f"多进程处理完成，耗时: {end_time - start_time:.3f} 秒")
        # print(f"结果数量: {len(results_mp)}")

        # --- 调用多线程处理函数 (作为对比) ---
        print(f"\n开始使用 threaded_process_pdf (多线程) 处理 {pdf.page_count} 页...")
        start_time = time.time()
        # 使用 16 个线程处理同一个 PDF 文件
        results_mt = threaded_process_pdf('/tmp/[MS-DOC].pdf', num_threads=16) # 请替换为实际 PDF 路径
        end_time = time.time()
        print(f"多线程处理完成，耗时: {end_time - start_time:.3f} 秒")
        # print(f"结果数量: {len(results_mt)}")

        # 关闭原始 PDF 文档
        pdf.close()

    except Exception as e:
        print(f"执行示例时出错: {e}")
        print("请确保 /tmp/[MS-DOC].pdf 文件存在或替换为有效的 PDF 文件路径。")


    """ 多线程处理 (fitz 页面转图像) 的基准测试结果示例 (来自原注释)
    总页数: 578
    线程数,    耗时
    1           7.351 秒
    2           6.334 秒
    4           5.968 秒  <- 似乎是最佳点
    8           6.728 秒
    16          8.085 秒
    * 注意: GIL (全局解释器锁) 可能限制了 CPU 密集型任务在纯 Python 线程中的并行效果。
    * I/O 操作（如此处没有的磁盘读写）或调用释放 GIL 的 C 扩展（如某些 numpy 操作或 fitz 底层）可能从多线程中受益更多。
    """

    """ 多进程处理 (fitz 页面转图像) 的基准测试结果示例 (来自原注释)
    总页数: 578
    进程数,    耗时
    1           17.170 秒 (比单线程慢，因为有进程创建和通信开销)
    2           10.170 秒
    4           7.841 秒  <- 接近最佳点
    8           7.900 秒
    16          7.984 秒
    * 注意: 多进程避免了 GIL 问题，对于 CPU 密集任务通常扩展性更好，但有更高的启动和通信成本。z
    * 此处多进程看起来比多线程稍慢，可能是因为页面转换操作本身可能部分释放了 GIL，或者进程间数据传输 (序列化/反序列化页面字节) 的开销较大。
    """
