
# 导入枚举库，用于定义枚举类型
import enum
# 从项目配置中导入模型块类型枚举
from magic_pdf.config.model_block_type import ModelBlockTypeEnum
# 从项目配置中导入OCR内容类别ID和内容类型
from magic_pdf.config.ocr_content_type import CategoryId, ContentType
# 从项目数据模块导入数据集类
from magic_pdf.data.dataset import Dataset
# 从项目库中导入边界框（bounding box）相关工具函数
from magic_pdf.libs.boxbase import (_is_in, bbox_distance, bbox_relative_pos,
                                    calculate_iou)
# 从项目库中导入坐标转换工具函数
from magic_pdf.libs.coordinate_transform import get_scale_ratio
# 从项目预处理模块导入移除边界框重叠的函数
from magic_pdf.pre_proc.remove_bbox_overlap import _remove_overlap_between_bbox

# 定义标题与相关内容重叠区域的比率阈值，用于判断是否关联
CAPATION_OVERLAP_AREA_RATIO = 0.6
# 定义合并框时允许的重叠区域比率阈值（大于1表示允许一定程度的包含）
MERGE_BOX_OVERLAP_AREA_RATIO = 1.1


# 定义空间相对位置关系的枚举
class PosRelationEnum(enum.Enum):
    LEFT = 'left'  # 左侧
    RIGHT = 'right'  # 右侧
    UP = 'up'  # 上方
    BOTTOM = 'bottom'  # 下方
    ALL = 'all'  # 所有方向（或任意方向）


# 用于处理和解析模型输出数据的核心类
class MagicModel:
    """每个函数没有得到元素的时候返回空list."""  # 注意：此类中的许多 'get_' 方法在没有找到对应元素时会返回空列表。

    # 修正坐标轴：处理坐标缩放，将多边形(poly)转换为边界框(bbox)，并移除无效框（宽度或高度<=0）
    def __fix_axis(self):
        # 遍历模型输出列表中的每一页信息
        for model_page_info in self.__model_list:
            # 初始化需要移除的元素列表
            need_remove_list = []
            # 获取当前页码
            page_no = model_page_info['page_info']['page_no']
            # 获取当前页模型坐标到文档坐标的水平和垂直缩放比例
            horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(
                model_page_info, self.__docs.get_page(page_no)
            )
            # 获取当前页的布局检测结果列表
            layout_dets = model_page_info['layout_dets']
            # 遍历当前页的每一个布局检测结果
            for layout_det in layout_dets:
                # 检查是否存在 'bbox' 键
                if layout_det.get('bbox') is not None:
                    # 兼容直接输出bbox的模型数据,如paddle
                    # 直接解包边界框坐标
                    x0, y0, x1, y1 = layout_det['bbox']
                else:
                    # 兼容直接输出poly的模型数据，如xxx
                    # 从多边形坐标中提取左上角和右下角坐标来构建边界框
                    x0, y0, _, _, x1, y1, _, _ = layout_det['poly']

                # 根据缩放比例将模型坐标转换为文档坐标，并取整
                bbox = [
                    int(x0 / horizontal_scale_ratio),
                    int(y0 / vertical_scale_ratio),
                    int(x1 / horizontal_scale_ratio),
                    int(y1 / vertical_scale_ratio),
                ]
                # 将计算得到的文档坐标bbox存回布局检测结果中
                layout_det['bbox'] = bbox

                # 检查计算后的bbox宽度或高度是否小于等于0
                if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                    # 如果是，则将此检测结果加入待移除列表
                    need_remove_list.append(layout_det)

            # 遍历待移除列表，从布局检测结果中删除无效的元素
            for need_remove in need_remove_list:
                layout_dets.remove(need_remove)

    # 移除低置信度的检测结果
    def __fix_by_remove_low_confidence(self):
        # 遍历模型输出列表中的每一页信息
        for model_page_info in self.__model_list:
            # 初始化需要移除的元素列表
            need_remove_list = []
            # 获取当前页的布局检测结果列表
            layout_dets = model_page_info['layout_dets']
            # 遍历当前页的每一个布局检测结果
            for layout_det in layout_dets:
                # 检查置信度分数是否小于等于0.05
                if layout_det['score'] <= 0.05:
                    # 如果是，则加入待移除列表
                    need_remove_list.append(layout_det)
                else:
                    # 否则，继续检查下一个
                    continue
            # 遍历待移除列表，从布局检测结果中删除低置信度的元素
            for need_remove in need_remove_list:
                layout_dets.remove(need_remove)

    # 移除高IOU（交并比）重叠且置信度较低的检测结果
    def __fix_by_remove_high_iou_and_low_confidence(self):
        # 遍历模型输出列表中的每一页信息
        for model_page_info in self.__model_list:
            # 初始化需要移除的元素列表
            need_remove_list = []
            # 获取当前页的布局检测结果列表
            layout_dets = model_page_info['layout_dets']
            # 使用双重循环比较页面内所有检测结果对
            for layout_det1 in layout_dets:
                for layout_det2 in layout_dets:
                    # 跳过与自身的比较
                    if layout_det1 == layout_det2:
                        continue

                    # 检查两个检测结果的类别ID是否都在指定的范围内（通常是文本或段落相关的类别）
                    if layout_det1['category_id'] in [
                        0,  # 文本
                        1,  # 标题
                        2,  # 列表
                        3,  # 图形 (Figure) - 根据后续用法，这里可能包含了需要处理的类型
                        4,  # 图标题 (Figure caption)
                        5,  # 表格 (Table)
                        6,  # 表标题 (Table caption)
                        7,  # 页眉 (Header) / 页脚 (Footer) / 脚注 (Footnote) - 根据后续用法调整
                        8,  # 引用 (Reference)
                        9,  # 公式 (Equation) - 这里可能包括行内和行间
                    ] and layout_det2['category_id'] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                        # 计算两个检测结果边界框的IOU
                        if (
                                calculate_iou(layout_det1['bbox'], layout_det2['bbox'])
                                > 0.9  # 如果IOU大于0.9，表示高度重叠
                        ):
                            # 比较两者的置信度分数
                            if layout_det1['score'] < layout_det2['score']:
                                # layout_det1分数较低，标记为待移除
                                layout_det_need_remove = layout_det1
                            else:
                                # layout_det2分数较低（或相等，优先移除后者），标记为待移除
                                layout_det_need_remove = layout_det2

                            # 如果待移除项尚未加入列表，则添加
                            if layout_det_need_remove not in need_remove_list:
                                need_remove_list.append(layout_det_need_remove)
                        else:
                            # IOU不满足条件，继续比较下一对
                            continue
                    else:
                        # 类别ID不满足条件，继续比较下一对
                        continue

            # 遍历待移除列表，从布局检测结果中删除需要移除的元素
            for need_remove in need_remove_list:
                layout_dets.remove(need_remove)

    # 初始化MagicModel实例
    def __init__(self, model_list: list, docs: Dataset):
        # 存储传入的模型输出列表
        self.__model_list = model_list
        # 存储传入的文档数据集对象
        self.__docs = docs

        # 第一步：修正坐标轴和无效框
        """为所有模型数据添加bbox信息(缩放，poly->bbox)"""
        self.__fix_axis()

        # 第二步：移除置信度极低的检测结果
        """删除置信度特别低的模型数据(<0.05),提高质量"""
        self.__fix_by_remove_low_confidence()

        # 第三步：处理高重叠区域，保留置信度更高的结果
        """删除高iou(>0.9)数据中置信度较低的那个"""
        self.__fix_by_remove_high_iou_and_low_confidence()

        # 第四步：修正脚注类别
        self.__fix_footnote()

    # 计算两个边界框之间的自定义距离（可能考虑相对位置和尺寸）
    def _bbox_distance(self, bbox1, bbox2):
        # 获取bbox2相对于bbox1的相对位置（左、右、下、上）
        left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)
        # 将布尔标志转换为列表
        flags = [left, right, bottom, top]
        # 计算bbox2相对于bbox1处于多少个基本方位（期望只有一个）
        count = sum([1 if v else 0 for v in flags])
        # 如果相对位置不明确（例如，对角线方向，处于多个方位），则返回无穷大距离，表示无效或不考虑
        if count > 1:
            return float('inf')

        # 如果是左右相邻关系
        if left or right:
            # 计算两个bbox的高度
            l1 = bbox1[3] - bbox1[1]
            l2 = bbox2[3] - bbox2[1]
        # 如果是上下相邻关系
        else:
            # 计算两个bbox的宽度
            l1 = bbox1[2] - bbox1[0]
            l2 = bbox2[2] - bbox2[0]

        # 如果bbox2的尺寸（高度或宽度，取决于相对位置）比bbox1大很多（超过30%），也返回无穷大距离
        # 这可能用于避免将一个大的元素关联到一个小元素上（例如，一个跨栏表格关联到一个单栏文本）
        if l2 > l1 and (l2 - l1) / l1 > 0.3:
            return float('inf')

        # 如果以上条件都满足，则计算并返回标准的边界框间距离
        return bbox_distance(bbox1, bbox2)

    # 修正脚注类别：根据脚注与图、表的距离，将靠近图的脚注重新分类为图片脚注
    def __fix_footnote(self):
        # 类别ID定义: 3: figure (图片), 5: table (表格), 7: footnote (脚注) - 这里假设7是通用脚注
        # 遍历模型输出列表中的每一页信息
        for model_page_info in self.__model_list:
            # 初始化列表，用于存储当前页的脚注、图片和表格对象
            footnotes = []
            figures = []
            tables = []
            # 遍历当前页的所有布局检测结果
            for obj in model_page_info['layout_dets']:
                # 根据类别ID将对象分类存储
                if obj['category_id'] == 7:  # 假设 7 是通用脚注 ID
                    footnotes.append(obj)
                elif obj['category_id'] == 3:  # 图片 ID
                    figures.append(obj)
                elif obj['category_id'] == 5:  # 表格 ID
                    tables.append(obj)

            # 如果当前页没有脚注，或者同时没有图片和表格，则无需处理，跳到下一页
            if len(footnotes) == 0 or (len(figures) == 0 and len(tables) == 0):
                continue

            # 初始化字典，存储每个脚注到最近图片和最近表格的距离
            dis_figure_footnote = {}  # key: footnote索引, value: 到最近figure的距离
            dis_table_footnote = {}  # key: footnote索引, value: 到最近table的距离

            # 计算每个脚注到所有图片的距离，并记录最小值
            for i in range(len(footnotes)):
                for j in range(len(figures)):
                    # 检查脚注和图片的相对位置标志数量，如果大于1（表示非直接相邻），则跳过
                    pos_flag_count = sum(
                        list(
                            map(
                                lambda x: 1 if x else 0,
                                bbox_relative_pos(
                                    footnotes[i]['bbox'], figures[j]['bbox']
                                ),
                            )
                        )
                    )
                    if pos_flag_count > 1:
                        continue
                    # 计算自定义距离，并更新该脚注到最近图片的距离
                    dis_figure_footnote[i] = min(
                        self._bbox_distance(figures[j]['bbox'], footnotes[i]['bbox']),
                        dis_figure_footnote.get(i, float('inf')),  # 使用get获取当前最小值，默认为无穷大
                    )

            # 计算每个脚注到所有表格的距离，并记录最小值
            for i in range(len(footnotes)):
                for j in range(len(tables)):
                    # 检查脚注和表格的相对位置标志数量，如果大于1，则跳过
                    pos_flag_count = sum(
                        list(
                            map(
                                lambda x: 1 if x else 0,
                                bbox_relative_pos(
                                    footnotes[i]['bbox'], tables[j]['bbox']
                                ),
                            )
                        )
                    )
                    if pos_flag_count > 1:
                        continue
                    # 计算自定义距离，并更新该脚注到最近表格的距离
                    dis_table_footnote[i] = min(
                        self._bbox_distance(tables[j]['bbox'], footnotes[i]['bbox']),
                        dis_table_footnote.get(i, float('inf')),  # 使用get获取当前最小值，默认为无穷大
                    )

            # 重新分类脚注
            for i in range(len(footnotes)):
                # 如果脚注i没有找到任何有效距离的图片（可能所有图片都不相邻或距离计算为inf），则跳过
                if i not in dis_figure_footnote:
                    continue
                # 比较该脚注到最近图片和最近表格的距离
                # 如果到图片的距离 小于 到表格的距离 (如果没找到表格，则表格距离为inf)
                if dis_table_footnote.get(i, float('inf')) > dis_figure_footnote[i]:
                    # 将该脚注的类别ID修改为图片脚注 (ImageFootnote)
                    footnotes[i]['category_id'] = CategoryId.ImageFootnote

    # 移除完全包含在其他边界框内的边界框 (只保留外层框)
    def __reduct_overlap(self, bboxes):
        # 获取边界框列表的长度
        N = len(bboxes)
        # 初始化一个布尔列表，标记每个边界框是否应该保留，默认为True
        keep = [True] * N
        # 使用双重循环比较所有边界框对
        for i in range(N):
            for j in range(N):
                # 跳过与自身的比较
                if i == j:
                    continue
                # 检查bbox i 是否完全包含在 bbox j 内
                if _is_in(bboxes[i]['bbox'], bboxes[j]['bbox']):
                    # 如果是，则标记 bbox i 不保留 (因为j包含了i)
                    keep[i] = False
                    # 注意：这里没有 break，如果i被多个框包含，会被多次标记为False，结果不变
                    # 考虑是否需要优化：如果i被j包含，是否需要检查j是否被其他框包含？目前逻辑是只移除内层框。
        # 返回一个新列表，只包含标记为保留的边界框
        return [bboxes[i] for i in range(N) if keep[i]]

    # (版本2) 根据距离和位置关系将主体类别(subject)和客体类别(object)进行关联绑定
    def __tie_up_category_by_distance_v2(
            self,
            page_no: int,  # 页码
            subject_category_id: int,  # 主体元素的类别ID (例如：图片、表格)
            object_category_id: int,  # 客体元素的类别ID (例如：标题、脚注)
            priority_pos: PosRelationEnum,  # 优先考虑的相对位置 (上/下/左/右/所有)
    ):
        """
        根据距离和指定的优先方向，将页面中的主体元素和客体元素进行匹配关联。
        例如，将图片(subject)与其最近的下方标题(object)关联。

        Args:
            page_no (int): 要处理的页码。
            subject_category_id (int): 主体元素的类别ID。
            object_category_id (int): 客体元素的类别ID。
            priority_pos (PosRelationEnum): 优先匹配的方向枚举值。

        Returns:
            list: 返回一个列表，每个元素代表一个主体及其关联的所有客体。
                  格式: [{'sub_bbox': {'bbox': [...], 'score': ...}, 'obj_bboxes': [{'bbox': [...], 'score': ...}, ...], 'sub_idx': ...}]
        """
        # 定义轴向距离差异倍数阈值，用于判断距离是否相近
        AXIS_MULPLICITY = 0.5

        # 1. 提取并预处理主体(subject)元素
        # a. 过滤出指定页码和类别ID的主体元素
        # b. 提取 bbox 和 score 信息
        # c. 使用 __reduct_overlap 移除内部重叠的主体元素 (保留外层)
        subjects = self.__reduct_overlap(
            list(
                map(
                    lambda x: {'bbox': x['bbox'], 'score': x['score']},
                    filter(
                        lambda x: x['category_id'] == subject_category_id,
                        self.__model_list[page_no]['layout_dets'],
                    ),
                )
            )
        )

        # 2. 提取并预处理客体(object)元素 (同主体处理方式)
        objects = self.__reduct_overlap(
            list(
                map(
                    lambda x: {'bbox': x['bbox'], 'score': x['score']},
                    filter(
                        lambda x: x['category_id'] == object_category_id,
                        self.__model_list[page_no]['layout_dets'],
                    ),
                )
            )
        )

        # 获取客体元素的数量
        M = len(objects)
        # 如果没有主体或客体，直接返回空列表
        if len(subjects) == 0 or M == 0:
            return []

        # 3. 对主体和客体按坐标排序 (左上角优先)，便于处理
        subjects.sort(key=lambda x: x['bbox'][0] ** 2 + x['bbox'][1] ** 2)  # 按左上角欧氏距离排序
        objects.sort(key=lambda x: x['bbox'][0] ** 2 + x['bbox'][1] ** 2)  # 按左上角欧氏距离排序

        # 4. 初始化主体到客体的映射关系字典，key为主体索引，value为关联的客体索引列表
        sub_obj_map_h = {i: [] for i in range(len(subjects))}

        # 5. 初始化字典，存储每个客体在四个方向上最近的主体及其距离
        # 格式: {'top': [[主体索引, 距离], [主体索引, 距离], ...], 'bottom': ..., 'left': ..., 'right': ...}
        # 初始值设为 [-1, 无穷大] 表示尚未找到
        dis_by_directions = {
            'top': [[-1, float('inf')]] * M,  # 上方最近的主体
            'bottom': [[-1, float('inf')]] * M,  # 下方最近的主体
            'left': [[-1, float('inf')]] * M,  # 左侧最近的主体
            'right': [[-1, float('inf')]] * M,  # 右侧最近的主体
        }

        # 6. 遍历每个客体(object)，计算其与所有主体(subject)的距离和相对位置
        for i, obj in enumerate(objects):
            # 获取客体bbox的宽度和高度
            l_x_axis, l_y_axis = (
                obj['bbox'][2] - obj['bbox'][0],
                obj['bbox'][3] - obj['bbox'][1],
            )
            # 计算一个基准单位长度，取宽高中的较小值，用于后续距离比较
            axis_unit = min(l_x_axis, l_y_axis) if min(l_x_axis, l_y_axis) > 0 else 1  # 避免除零

            # 遍历所有主体
            for j, sub in enumerate(subjects):
                # 尝试移除主体和客体间的重叠区域，得到可能不重叠的bbox (用于更精确判断相对位置)
                bbox1, bbox2, _ = _remove_overlap_between_bbox(
                    objects[i]['bbox'], subjects[j]['bbox']
                )
                # 获取调整后bbox2(主体)相对于bbox1(客体)的相对位置
                left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)
                # 将布尔标志转换为列表
                flags = [left, right, bottom, top]
                # 如果主体相对客体的位置不明确（处于多个方位），跳过此主体
                if sum([1 if v else 0 for v in flags]) > 1:
                    continue

                # 计算主体和客体原始bbox之间的距离
                current_distance = bbox_distance(obj['bbox'], sub['bbox'])

                # 根据相对位置，更新该客体在对应方向上的最近主体记录
                if left:  # 主体在客体左侧
                    if dis_by_directions['left'][i][1] > current_distance:
                        dis_by_directions['left'][i] = [j, current_distance]
                if right:  # 主体在客体右侧
                    if dis_by_directions['right'][i][1] > current_distance:
                        dis_by_directions['right'][i] = [j, current_distance]
                if bottom:  # 主体在客体下方
                    if dis_by_directions['bottom'][i][1] > current_distance:
                        dis_by_directions['bottom'][i] = [j, current_distance]
                if top:  # 主体在客体上方
                    if dis_by_directions['top'][i][1] > current_distance:
                        dis_by_directions['top'][i] = [j, current_distance]

            # 7. 确定客体i的最佳关联主体 (基于距离和优先方向)

            # a. 特殊处理：如果上下方向都有候选主体，且距离相近，并且优先方向是上或下
            if (
                    dis_by_directions['top'][i][1] != float('inf')  # 上方有主体
                    and dis_by_directions['bottom'][i][1] != float('inf')  # 下方有主体
                    and priority_pos in (PosRelationEnum.BOTTOM, PosRelationEnum.UP)  # 优先考虑上下
            ):
                RATIO = 3  # 定义上下距离差异的倍数阈值
                # 如果上下距离差小于 RATIO * 基准单位长度 (认为距离接近)
                if (
                        abs(
                            dis_by_directions['top'][i][1]
                            - dis_by_directions['bottom'][i][1]
                        )
                        < RATIO * axis_unit
                ):
                    # 根据优先方向直接选择关联主体
                    if priority_pos == PosRelationEnum.BOTTOM:
                        sub_idx_to_add = dis_by_directions['bottom'][i][0]
                        sub_obj_map_h[sub_idx_to_add].append(i)  # 将客体i关联到下方主体
                    else:  # priority_pos == PosRelationEnum.UP
                        sub_idx_to_add = dis_by_directions['top'][i][0]
                        sub_obj_map_h[sub_idx_to_add].append(i)  # 将客体i关联到上方主体
                    continue  # 处理完此客体，进行下一次循环

            # b. 确定左右方向的最佳候选主体 (left_or_right)
            # 初始化左右最佳候选为无效
            left_or_right = [-1, float('inf')]
            # 如果左右都有主体
            if dis_by_directions['left'][i][1] != float('inf') or dis_by_directions['right'][i][1] != float('inf'):
                # 如果左右两侧都有主体
                if dis_by_directions['left'][i][1] != float('inf') and dis_by_directions['right'][i][1] != float('inf'):
                    # 如果左右距离差小于阈值 (距离相近)
                    if AXIS_MULPLICITY * axis_unit >= abs(
                            dis_by_directions['left'][i][1]
                            - dis_by_directions['right'][i][1]
                    ):
                        # 进一步比较：考虑主体和客体的高度差异以及距离
                        # 选择 (主体高度-客体高度)绝对值 + 距离 更小的一方
                        left_sub_bbox = subjects[dis_by_directions['left'][i][0]]['bbox']
                        right_sub_bbox = subjects[dis_by_directions['right'][i][0]]['bbox']
                        left_sub_bbox_y_axis = left_sub_bbox[3] - left_sub_bbox[1]
                        right_sub_bbox_y_axis = right_sub_bbox[3] - right_sub_bbox[1]
                        # 比较惩罚分数（高度差+距离），选择分数低者
                        if (
                                abs(left_sub_bbox_y_axis - l_y_axis)
                                + dis_by_directions['left'][i][1]  # 原代码这里是[0], 疑似笔误，应为距离[1]
                                > abs(right_sub_bbox_y_axis - l_y_axis)
                                + dis_by_directions['right'][i][1]  # 原代码这里是[0], 疑似笔误，应为距离[1]
                        ):
                            left_or_right = dis_by_directions['right'][i]  # 右侧更优
                        else:
                            left_or_right = dis_by_directions['left'][i]  # 左侧更优
                    else:  # 如果左右距离差异较大
                        # 直接选择距离更近的一方
                        left_or_right = dis_by_directions['left'][i]
                        if left_or_right[1] > dis_by_directions['right'][i][1]:
                            left_or_right = dis_by_directions['right'][i]
                else:  # 如果只有一侧有主体 (左或右)
                    # 选择有主体的那一侧
                    left_or_right = dis_by_directions['left'][i]
                    if left_or_right[1] == float('inf'):  # 如果左侧是inf，说明只有右侧有
                        left_or_right = dis_by_directions['right'][i]

            # c. 确定上下方向的最佳候选主体 (top_or_bottom) - 逻辑同左右方向类似
            # 初始化上下最佳候选为无效
            top_or_bottom = [-1, float('inf')]
            # 如果上下有主体
            if dis_by_directions['top'][i][1] != float('inf') or dis_by_directions['bottom'][i][1] != float('inf'):
                # 如果上下两侧都有主体
                if dis_by_directions['top'][i][1] != float('inf') and dis_by_directions['bottom'][i][1] != float('inf'):
                    # 如果上下距离差小于阈值 (距离相近)
                    if AXIS_MULPLICITY * axis_unit >= abs(
                            dis_by_directions['top'][i][1]
                            - dis_by_directions['bottom'][i][1]
                    ):
                        # 进一步比较：考虑主体和客体的宽度差异以及距离
                        top_sub_bbox = subjects[dis_by_directions['top'][i][0]]['bbox']  # 上方主体bbox
                        bottom_sub_bbox = subjects[dis_by_directions['bottom'][i][0]]['bbox']  # 下方主体bbox
                        top_sub_bbox_x_axis = top_sub_bbox[2] - top_sub_bbox[0]
                        bottom_sub_bbox_x_axis = bottom_sub_bbox[2] - bottom_sub_bbox[0]
                        # 比较惩罚分数（宽度差+距离），选择分数低者
                        if (
                                abs(bottom_sub_bbox_x_axis - l_x_axis)  # 下方主体宽度与客体宽度差
                                + dis_by_directions['bottom'][i][1]
                                > abs(top_sub_bbox_x_axis - l_x_axis)  # 上方主体宽度与客体宽度差
                                + dis_by_directions['top'][i][1]
                        ):
                            top_or_bottom = dis_by_directions['top'][i]  # 上方更优
                        else:
                            top_or_bottom = dis_by_directions['bottom'][i]  # 下方更优
                    else:  # 如果上下距离差异较大
                        # 直接选择距离更近的一方
                        top_or_bottom = dis_by_directions['top'][i]
                        if top_or_bottom[1] > dis_by_directions['bottom'][i][1]:
                            top_or_bottom = dis_by_directions['bottom'][i]
                else:  # 如果只有一侧有主体 (上或下)
                    # 选择有主体的那一侧
                    top_or_bottom = dis_by_directions['top'][i]
                    if top_or_bottom[1] == float('inf'):  # 如果上方是inf，说明只有下方有
                        top_or_bottom = dis_by_directions['bottom'][i]

            # d. 最终决策：比较左右最佳候选和上下最佳候选
            # 如果左右和上下都有有效的候选主体
            if left_or_right[1] != float('inf') or top_or_bottom[1] != float('inf'):
                # 如果左右和上下均有有效候选
                if left_or_right[1] != float('inf') and top_or_bottom[1] != float('inf'):
                    # 如果左右最佳距离和上下最佳距离相近
                    if AXIS_MULPLICITY * axis_unit >= abs(
                            left_or_right[1] - top_or_bottom[1]
                    ):
                        # 进一步比较：基于尺寸相似度决定最终关联
                        # 选择 主体尺寸/客体尺寸 比率 更接近1 的方向
                        y_axis_bbox = subjects[left_or_right[0]]['bbox']  # 左右方向最佳主体bbox
                        x_axis_bbox = subjects[top_or_bottom[0]]['bbox']  # 上下方向最佳主体bbox
                        # 比较 x轴(宽度)相对差异 和 y轴(高度)相对差异
                        # 选择相对差异更小的一方进行关联
                        if (
                                abs((x_axis_bbox[2] - x_axis_bbox[0]) - l_x_axis) / (
                        l_x_axis if l_x_axis else 1)  # 上下方向宽度相对差异
                                > abs((y_axis_bbox[3] - y_axis_bbox[1]) - l_y_axis) / (l_y_axis if l_y_axis else 1)
                        # 左右方向高度相对差异
                        ):
                            # 左右方向的高度差异更小，关联到左右方向的最佳主体
                            sub_obj_map_h[left_or_right[0]].append(i)
                        else:
                            # 上下方向的宽度差异更小（或相等），关联到上下方向的最佳主体
                            sub_obj_map_h[top_or_bottom[0]].append(i)
                    else:  # 如果左右和上下距离差异较大
                        # 直接选择距离更近的方向进行关联
                        if left_or_right[1] > top_or_bottom[1]:
                            sub_obj_map_h[top_or_bottom[0]].append(i)  # 上下更近
                        else:
                            sub_obj_map_h[left_or_right[0]].append(i)  # 左右更近 (或相等)
                else:  # 如果只有左右或上下其中一个方向有有效候选
                    if left_or_right[1] != float('inf'):  # 只有左右有候选
                        sub_obj_map_h[left_or_right[0]].append(i)
                    else:  # 只有上下有候选
                        sub_obj_map_h[top_or_bottom[0]].append(i)
            # else: # 如果所有方向都没有找到有效的主体，则此客体不关联任何主体 (隐含逻辑)

        # 8. 格式化输出结果
        ret = []
        # 遍历主体到客体的映射字典
        for i in sub_obj_map_h.keys():
            # 构建每个主体的结果字典
            ret.append(
                {
                    'sub_bbox': {  # 主体信息
                        'bbox': subjects[i]['bbox'],
                        'score': subjects[i]['score'],
                    },
                    'obj_bboxes': [  # 关联的客体列表
                        {'score': objects[j]['score'], 'bbox': objects[j]['bbox']}
                        for j in sub_obj_map_h[i]  # 获取关联的客体索引列表
                    ],
                    'sub_idx': i,  # 主体在原始排序列表中的索引
                }
            )
        # 返回最终的关联结果列表
        return ret

    # (版本3) 使用不同的策略根据距离将主体类别和客体类别进行关联绑定 (基于全局最近邻思想)
    def __tie_up_category_by_distance_v3(
            self,
            page_no: int,  # 页码
            subject_category_id: int,  # 主体元素的类别ID
            object_category_id: int,  # 客体元素的类别ID
            priority_pos: PosRelationEnum,  # 优先考虑的相对位置 (注意：此版本似乎未使用此参数)
    ):
        """
        (版本3) 使用一种基于全局最近邻和迭代匹配的策略来关联主体和客体。

        Args:
            page_no (int): 要处理的页码。
            subject_category_id (int): 主体元素的类别ID。
            object_category_id (int): 客体元素的类别ID。
            priority_pos (PosRelationEnum): 优先匹配的方向枚举值 (在此版本中似乎未使用)。

        Returns:
            list: 返回一个列表，每个元素代表一个主体及其关联的所有客体。
                  格式: [{'sub_bbox': {'bbox': [...], 'score': ...}, 'obj_bboxes': [{'bbox': [...], 'score': ...}, ...], 'sub_idx': ...}]
        """
        # 1. 提取并预处理主体(subject)元素 (同v2)
        subjects = self.__reduct_overlap(
            list(
                map(
                    lambda x: {'bbox': x['bbox'], 'score': x['score']},
                    filter(
                        lambda x: x['category_id'] == subject_category_id,
                        self.__model_list[page_no]['layout_dets'],
                    ),
                )
            )
        )
        # 2. 提取并预处理客体(object)元素 (同v2)
        objects = self.__reduct_overlap(
            list(
                map(
                    lambda x: {'bbox': x['bbox'], 'score': x['score']},
                    filter(
                        lambda x: x['category_id'] == object_category_id,
                        self.__model_list[page_no]['layout_dets'],
                    ),
                )
            )
        )

        # 初始化结果列表
        ret = []
        # 获取主体和客体的数量
        N, M = len(subjects), len(objects)
        # 如果没有主体，直接返回空列表（因为是以主体为中心构建结果）
        if N == 0:
            return []

        # 对主体和客体按坐标排序 (同v2)
        subjects.sort(key=lambda x: x['bbox'][0] ** 2 + x['bbox'][1] ** 2)
        objects.sort(key=lambda x: x['bbox'][0] ** 2 + x['bbox'][1] ** 2)

        # 3. 创建包含所有主体和客体信息的列表，并标记类型和原始索引
        OBJ_IDX_OFFSET = 10000  # 定义一个偏移量，用于区分客体索引和主体索引
        SUB_BIT_KIND, OBJ_BIT_KIND = 0, 1  # 定义主体和客体的类型标记
        # 格式: [(原始索引(带偏移), 类型标记, x0, y0), ...]
        all_boxes_with_idx = [(i, SUB_BIT_KIND, sub['bbox'][0], sub['bbox'][1]) for i, sub in enumerate(subjects)] + \
                             [(i + OBJ_IDX_OFFSET, OBJ_BIT_KIND, obj['bbox'][0], obj['bbox'][1]) for i, obj in
                              enumerate(objects)]

        # 4. 迭代匹配主体和客体
        seen_idx = set()  # 记录已匹配或处理过的元素的 (带偏移的) 索引
        seen_sub_idx = set()  # 记录已匹配或处理过的主体的 原始索引

        # 当还有未处理的主体时，继续迭代
        while N > len(seen_sub_idx) and M > 0:  # 确保有主体且有客体可匹配
            # a. 筛选出当前所有未处理的元素作为候选
            candidates = []
            for idx, kind, x0, y0 in all_boxes_with_idx:
                if idx in seen_idx:  # 跳过已处理的
                    continue
                candidates.append((idx, kind, x0, y0))

            # 如果没有候选元素了，退出循环
            if len(candidates) == 0:
                break

            # b. 找到全局左上角的元素作为起始点
            # 计算候选中的最小x和最小y
            left_x = min([v[2] for v in candidates]) if candidates else 0
            top_y = min([v[3] for v in candidates]) if candidates else 0
            # 按到左上角(left_x, top_y)的距离排序候选者
            candidates.sort(key=lambda x: (x[2] - left_x) ** 2 + (x[3] - top_y) ** 2)
            # 取出最近的元素作为第一个元素 (fst)
            fst_idx, fst_kind, fst_x0, fst_y0 = candidates[0]

            # c. 在剩余候选者中，找到与fst类型不同且距离fst最近的元素 (nxt)
            # 重新按到fst元素的距离排序剩余候选者
            candidates.sort(key=lambda x: (x[2] - fst_x0) ** 2 + (x[3] - fst_y0) ** 2)
            nxt = None  # 初始化下一个匹配项
            for i in range(1, len(candidates)):  # 从第二个元素开始查找
                # 检查类型是否与fst不同 (异或结果为1)
                if candidates[i][1] ^ fst_kind == 1:
                    nxt = candidates[i]  # 找到第一个类型不同的最近元素
                    break

            # 如果找不到类型不同的元素（例如只剩下同类元素），退出循环
            if nxt is None:
                break

            # d. 确定配对的主体索引 (sub_idx) 和客体索引 (obj_idx)
            if fst_kind == SUB_BIT_KIND:  # 如果第一个是主体
                sub_idx, obj_idx = fst_idx, nxt[0] - OBJ_IDX_OFFSET
            else:  # 如果第一个是客体
                sub_idx, obj_idx = nxt[0], fst_idx - OBJ_IDX_OFFSET

            # e. 检查配对质量：防止客体被错误关联到非最近的主体
            # 计算当前配对的主体和客体之间的距离
            pair_dis = bbox_distance(subjects[sub_idx]['bbox'], objects[obj_idx]['bbox'])
            # 查找该客体到其他所有 *未处理* 主体的最短距离
            nearest_dis = float('inf')
            for i in range(N):
                # 跳过当前配对的主体 和 已处理的主体
                if i == sub_idx or i in seen_sub_idx: continue
                nearest_dis = min(nearest_dis, bbox_distance(subjects[i]['bbox'], objects[obj_idx]['bbox']))

            # 如果当前配对距离 >= 3倍 的到其他未处理主体的最近距离，认为这个配对可能不好
            # （说明客体离另一个未处理的主体近得多）
            if pair_dis >= 3 * nearest_dis and nearest_dis != float('inf'):
                # 仅将当前配对的主体标记为已处理（不再参与后续的fst选择），但不进行配对
                seen_idx.add(sub_idx)
                seen_sub_idx.add(sub_idx)  # 标记主体已处理
                continue  # 继续下一次迭代寻找新的配对

            # f. 确认配对，标记主体和客体为已处理
            seen_idx.add(sub_idx)  # 标记主体索引 (无偏移)
            seen_idx.add(obj_idx + OBJ_IDX_OFFSET)  # 标记客体索引 (带偏移)
            seen_sub_idx.add(sub_idx)  # 标记主体原始索引

            # g. 将配对结果添加到结果列表中
            # 注意：这里的结构是先找到一个配对就添加，后续如果发现其他客体也应关联到此主体，会在步骤5处理
            ret.append(
                {
                    'sub_bbox': {  # 主体信息
                        'bbox': subjects[sub_idx]['bbox'],
                        'score': subjects[sub_idx]['score'],
                    },
                    'obj_bboxes': [  # 当前配对的客体信息
                        {'score': objects[obj_idx]['score'], 'bbox': objects[obj_idx]['bbox']}
                    ],
                    'sub_idx': sub_idx,  # 主体原始索引
                }
            )

        # 5. 处理剩余未匹配的客体
        for i in range(M):  # 遍历所有原始客体
            j = i + OBJ_IDX_OFFSET  # 获取带偏移的客体索引
            if j in seen_idx:  # 如果客体已被匹配，跳过
                continue

            # 将未匹配的客体标记为已处理
            seen_idx.add(j)

            # 为这个未匹配的客体找到距离最近的主体 (无论该主体是否已处理)
            nearest_dis, nearest_sub_idx = float('inf'), -1
            for k in range(N):  # 遍历所有主体
                dis = bbox_distance(objects[i]['bbox'], subjects[k]['bbox'])
                if dis < nearest_dis:
                    nearest_dis = dis
                    nearest_sub_idx = k

            # 如果找到了最近的主体 (nearest_sub_idx != -1)
            if nearest_sub_idx != -1:
                k = nearest_sub_idx  # 最近主体的索引
                # 检查这个最近的主体是否已经在之前的配对中处理过
                if k in seen_sub_idx:
                    # 如果处理过，找到结果列表中对应主体的记录，并将当前客体追加进去
                    for kk in range(len(ret)):
                        if ret[kk]['sub_idx'] == k:
                            ret[kk]['obj_bboxes'].append({'score': objects[i]['score'], 'bbox': objects[i]['bbox']})
                            break
                else:
                    # 如果这个最近的主体是全新的 (之前未被匹配)
                    # 创建一个新的记录，包含此主体和当前客体
                    ret.append(
                        {
                            'sub_bbox': {
                                'bbox': subjects[k]['bbox'],
                                'score': subjects[k]['score'],
                            },
                            'obj_bboxes': [
                                {'score': objects[i]['score'], 'bbox': objects[i]['bbox']}
                            ],
                            'sub_idx': k,
                        }
                    )
                    # 标记此主体为已处理
                    seen_sub_idx.add(k)
                    seen_idx.add(k)  # 标记主体索引 (无偏移)

        # 6. 处理剩余未匹配的主体 (这些主体没有任何客体关联到它们)
        for i in range(N):  # 遍历所有主体
            if i in seen_sub_idx:  # 如果主体已被处理 (即有关联的客体)，跳过
                continue
            # 为未关联任何客体的主体创建记录，客体列表为空
            ret.append(
                {
                    'sub_bbox': {
                        'bbox': subjects[i]['bbox'],
                        'score': subjects[i]['score'],
                    },
                    'obj_bboxes': [],  # 客体列表为空
                    'sub_idx': i,
                }
            )
            # (标记为已处理，虽然循环即将结束，保持逻辑完整性)
            seen_sub_idx.add(i)
            seen_idx.add(i)

        # 7. 返回最终的关联结果列表
        return ret

    # 获取页面中的图片及其关联的标题和脚注 (使用v3关联方法)
    def get_imgs_v2(self, page_no: int):
        # 使用 v3 方法关联图片(主体ID=3)和图片标题(客体ID=4)，优先方向设为下方(BOTTOM) - v3未使用此参数
        with_captions = self.__tie_up_category_by_distance_v3(
            page_no, CategoryId.ImageBody, CategoryId.ImageCaption, PosRelationEnum.BOTTOM
        )
        # 使用 v3 方法关联图片(主体ID=3)和图片脚注(客体ID=ImageFootnote)，优先方向设为所有(ALL) - v3未使用此参数
        with_footnotes = self.__tie_up_category_by_distance_v3(
            page_no, CategoryId.ImageBody, CategoryId.ImageFootnote, PosRelationEnum.ALL
        )

        # 合并标题和脚注结果
        ret = []
        # 遍历带标题的图片结果
        for v in with_captions:
            # 初始化图片记录，包含图片主体和标题列表
            record = {
                'image_body': v['sub_bbox'],
                'image_caption_list': v['obj_bboxes'],  # 这是图片标题
            }
            # 获取当前图片的索引
            filter_idx = v['sub_idx']
            # 在带脚注的结果中查找相同索引的图片记录
            # 使用 next 和 filter 找到对应项，如果找不到会抛出 StopIteration 错误 (假设总能找到)
            try:
                d = next(filter(lambda x: x['sub_idx'] == filter_idx, with_footnotes))
                # 将找到的脚注列表添加到记录中
                record['image_footnote_list'] = d['obj_bboxes']  # 这是图片脚注
            except StopIteration:
                # 如果在脚注结果中找不到对应的图片，则添加空列表
                record['image_footnote_list'] = []
            # 将完整的图片记录添加到最终结果列表
            ret.append(record)

        # 处理只有脚注没有标题的图片 (如果存在的话)
        caption_sub_indices = {item['sub_idx'] for item in with_captions}  # 获取所有有标题的图片索引
        for fn_item in with_footnotes:
            if fn_item['sub_idx'] not in caption_sub_indices:  # 如果此图片只有脚注没有标题
                record = {
                    'image_body': fn_item['sub_bbox'],
                    'image_caption_list': [],  # 标题列表为空
                    'image_footnote_list': fn_item['obj_bboxes'],  # 添加脚注列表
                }
                ret.append(record)

        # 返回包含图片主体、标题列表、脚注列表的记录列表
        return ret

    # 获取页面中的表格及其关联的标题和脚注 (使用v3关联方法)
    def get_tables_v2(self, page_no: int) -> list:
        # 使用 v3 方法关联表格(主体ID=5)和表格标题(客体ID=6)，优先方向设为上方(UP) - v3未使用此参数
        with_captions = self.__tie_up_category_by_distance_v3(
            page_no, CategoryId.TableBody, CategoryId.TableCaption, PosRelationEnum.UP
        )
        # 使用 v3 方法关联表格(主体ID=5)和表格脚注(客体ID=7)，优先方向设为所有(ALL) - v3未使用此参数
        with_footnotes = self.__tie_up_category_by_distance_v3(
            page_no, CategoryId.TableBody, CategoryId.TableFootnote, PosRelationEnum.ALL
        )

        # 合并标题和脚注结果 (逻辑同 get_imgs_v2)
        ret = []
        # 遍历带标题的表格结果
        for v in with_captions:
            # 初始化表格记录，包含表格主体和标题列表
            record = {
                'table_body': v['sub_bbox'],
                'table_caption_list': v['obj_bboxes'],  # 这是表格标题
            }
            # 获取当前表格的索引
            filter_idx = v['sub_idx']
            # 在带脚注的结果中查找相同索引的表格记录
            try:
                d = next(filter(lambda x: x['sub_idx'] == filter_idx, with_footnotes))
                # 将找到的脚注列表添加到记录中
                record['table_footnote_list'] = d['obj_bboxes']  # 这是表格脚注
            except StopIteration:
                # 如果在脚注结果中找不到对应的表格，则添加空列表
                record['table_footnote_list'] = []
            # 将完整的表格记录添加到最终结果列表
            ret.append(record)

        # 处理只有脚注没有标题的表格
        caption_sub_indices = {item['sub_idx'] for item in with_captions}  # 获取所有有标题的表格索引
        for fn_item in with_footnotes:
            if fn_item['sub_idx'] not in caption_sub_indices:  # 如果此表格只有脚注没有标题
                record = {
                    'table_body': fn_item['sub_bbox'],
                    'table_caption_list': [],  # 标题列表为空
                    'table_footnote_list': fn_item['obj_bboxes'],  # 添加脚注列表
                }
                ret.append(record)

        # 返回包含表格主体、标题列表、脚注列表的记录列表
        return ret

    # 获取页面中的图片及其关联信息 (公开接口，调用 v2 版本)
    def get_imgs(self, page_no: int):
        return self.get_imgs_v2(page_no)

    # 获取页面中的表格及其关联信息 (公开接口，调用 v2 版本)
    def get_tables(
            self, page_no: int
    ) -> list:  # 返回列表，每个元素包含: table主体坐标, caption坐标列表, table-note坐标列表
        return self.get_tables_v2(page_no)

    # 获取页面中的公式 (行内、行间)
    def get_equations(self, page_no: int) -> list:  # 返回包含坐标和可能的latex文本
        # 获取行内公式 (类型: EMBEDDING, 需要额外列: 'latex')
        inline_equations = self.__get_blocks_by_type(
            ModelBlockTypeEnum.EMBEDDING.value, page_no, ['latex']  # 假设 EMBEDDING 代表行内公式
        )
        # 获取行间公式 (类型: ISOLATED, 需要额外列: 'latex') - 这可能是通过OCR识别的行间公式
        interline_equations = self.__get_blocks_by_type(
            ModelBlockTypeEnum.ISOLATED.value, page_no, ['latex']  # 假设 ISOLATED 代表识别出的独立公式文本
        )
        # 获取行间公式块 (类型: ISOLATE_FORMULA) - 这可能是布局模型检测出的独立公式区域
        interline_equations_blocks = self.__get_blocks_by_type(
            ModelBlockTypeEnum.ISOLATE_FORMULA.value, page_no  # 假设 ISOLATE_FORMULA 代表公式块区域
        )
        # 返回三种类型的公式列表
        return inline_equations, interline_equations, interline_equations_blocks

    # 获取标记为废弃的块 (例如页眉页脚等)
    def get_discarded(self, page_no: int) -> list:  # 假设来自特定模型，只有坐标
        # 获取类型为 ABANDON 的块
        blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.ABANDON.value, page_no)
        return blocks

    # 获取纯文本块 (可能由特定模型生成，只有坐标)
    def get_text_blocks(self, page_no: int) -> list:  # 假设来自特定模型，只有坐标，没有文本内容
        # 获取类型为 PLAIN_TEXT 的块
        blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.PLAIN_TEXT.value, page_no)
        return blocks

    # 获取标题块 (可能由特定模型生成，只有坐标)
    def get_title_blocks(self, page_no: int) -> list:  # 假设来自特定模型，只有坐标，没文本内容
        # 获取类型为 TITLE 的块
        blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.TITLE.value, page_no)
        return blocks

    # 获取OCR识别的文本内容 (例如由PaddleOCR生成，包含文本和坐标)
    def get_ocr_text(self, page_no: int) -> list:  # 来自OCR引擎，有文本内容和坐标
        # 初始化文本片段列表
        text_spans = []
        # 获取指定页面的模型信息
        model_page_info = self.__model_list[page_no]
        # 获取布局检测结果
        layout_dets = model_page_info['layout_dets']
        # 遍历所有检测结果
        for layout_det in layout_dets:
            # 检查类别ID是否为 '15' (假设 '15' 代表OCR文本)
            # 注意：之前的类别ID是整数，这里是字符串 '15'，需要确认数据一致性或处理逻辑
            if str(layout_det.get('category_id', -1)) == '15':  # 转为字符串比较
                # 构建文本片段字典，包含bbox和文本内容
                span = {
                    'bbox': layout_det['bbox'],
                    'content': layout_det.get('text', ''),  # 获取文本，若无则为空字符串
                }
                # 添加到列表
                text_spans.append(span)
        # 返回OCR文本片段列表
        return text_spans

    # 获取页面上所有可识别的基本单元（spans），包括文本、图片、表格、公式等
    def get_all_spans(self, page_no: int) -> list:
        # 内部函数：移除列表中重复的span (基于字典完全相等判断)
        def remove_duplicate_spans(spans):
            new_spans = []
            for span in spans:
                # 检查当前span是否已存在于new_spans中
                if not any(span == existing_span for existing_span in new_spans):
                    new_spans.append(span)
            return new_spans

        # 初始化所有span的列表
        all_spans = []
        # 获取指定页面的模型信息
        model_page_info = self.__model_list[page_no]
        # 获取布局检测结果
        layout_dets = model_page_info['layout_dets']

        # 定义允许被视为span的类别ID列表
        allow_category_id_list = [
            CategoryId.ImageBody,  # 3: 图片
            CategoryId.TableBody,  # 5: 表格
            CategoryId.InlineEquation,  # 13: 行内公式
            CategoryId.InterlineEquation_YOLO,  # 14: 行间公式
            CategoryId.Text,  # 15: OCR识别文本 (注意: 这里用整数15，前面get_ocr_text用字符串'15')
        ]
        """当成span拼接的类别ID"""
        #  3: 'image', # 图片
        #  5: 'table',       # 表格
        #  13: 'inline_equation',     # 行内公式
        #  14: 'interline_equation',      # 行间公式
        #  15: 'text',      # ocr识别文本 (这里又是整数)

        # 遍历所有布局检测结果
        for layout_det in layout_dets:
            # 获取类别ID
            category_id = layout_det['category_id']
            # 检查类别ID是否在允许列表中
            if category_id in allow_category_id_list:
                # 构建基础span字典，包含bbox和score
                span = {'bbox': layout_det['bbox'], 'score': layout_det['score']}
                # 根据不同的类别ID，添加额外信息和类型
                if category_id == CategoryId.ImageBody:  # 图片
                    span['type'] = ContentType.Image
                elif category_id == CategoryId.TableBody:  # 表格
                    # 尝试获取表格的latex或html表示
                    latex = layout_det.get('latex', None)
                    html = layout_det.get('html', None)
                    if latex:
                        span['latex'] = latex
                    elif html:
                        span['html'] = html
                    span['type'] = ContentType.Table
                elif category_id == CategoryId.InlineEquation:  # 行内公式
                    span['content'] = layout_det.get('latex', '')  # 获取latex内容
                    span['type'] = ContentType.InlineEquation
                elif category_id == CategoryId.InterlineEquation_YOLO:  # 行间公式
                    span['content'] = layout_det.get('latex', '')  # 获取latex内容
                    span['type'] = ContentType.InterlineEquation
                elif category_id == CategoryId.Text:  # OCR文本
                    span['content'] = layout_det.get('text', '')  # 获取文本内容
                    span['type'] = ContentType.Text
                # 将构建好的span添加到总列表
                all_spans.append(span)

        # 移除重复的span并返回结果
        return remove_duplicate_spans(all_spans)

    # 获取指定页面的宽度和高度
    def get_page_size(self, page_no: int):
        # 获取当前页的页面对象信息
        page = self.__docs.get_page(page_no).get_page_info()
        # 获取当前页的宽度
        page_w = page.w
        # 获取当前页的高度
        page_h = page.h
        # 返回宽度和高度
        return page_w, page_h

    # 根据指定的类型ID (category_id) 从指定页面检索块信息
    def __get_blocks_by_type(
            self, type: int, page_no: int, extra_col: list[str] = []
    ) -> list:
        """
        内部辅助函数，用于根据类型ID筛选指定页面的布局检测结果。

        Args:
            type (int): 要筛选的类别ID (模型块类型或内容类型ID)。
            page_no (int): 要筛选的页码。
            extra_col (list[str], optional): 需要额外从原始检测结果中提取的字段名列表。 Defaults to [].

        Returns:
            list: 包含符合条件的块信息的列表，每个块是一个字典。
        """
        # 初始化结果列表
        blocks = []
        # 遍历模型列表中的每一页字典 (虽然可以通过page_no直接索引，但这里选择遍历)
        for page_dict in self.__model_list:
            # 获取当前页的布局检测结果，默认为空列表
            layout_dets = page_dict.get('layout_dets', [])
            # 获取当前页的页面信息，默认为空字典
            page_info = page_dict.get('page_info', {})
            # 获取当前页码，默认为-1
            page_number = page_info.get('page_no', -1)
            # 如果当前页码与目标页码不符，跳过当前页
            if page_no != page_number:
                continue

            # 如果页码匹配，则遍历该页的布局检测结果
            for item in layout_dets:
                # 获取当前项的类别ID，默认为-1
                category_id = item.get('category_id', -1)
                # 获取当前项的边界框，默认为None
                bbox = item.get('bbox', None)
                # 如果当前项的类别ID与目标类型匹配
                if category_id == type:
                    # 构建块字典，包含bbox和score
                    block = {
                        'bbox': bbox,
                        'score': item.get('score'),
                    }
                    # 遍历需要额外提取的字段名列表
                    for col in extra_col:
                        # 从原始检测结果中获取对应字段的值，若无则为None，并添加到块字典中
                        block[col] = item.get(col, None)
                    # 将构建好的块字典添加到结果列表中
                    blocks.append(block)
            # 找到目标页面后即可退出外层循环 (优化点)
            break

            # 返回筛选结果列表
        return blocks

    # 获取指定页面的原始模型数据列表 (未经此类处理的原始layout_dets)
    def get_model_list(self, page_no):
        # 直接通过页码索引返回对应的模型页面信息字典
        # 假设 self.__model_list 是按页码顺序存储的列表
        # 注意：这里没有错误处理，如果page_no超出范围会引发IndexError
        return self.__model_list[page_no]


