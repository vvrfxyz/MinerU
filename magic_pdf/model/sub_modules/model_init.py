# 导入torch库，用于深度学习任务
import torch
# 导入loguru库，用于日志记录
from loguru import logger
# 从常量配置中导入模型名称常量
from magic_pdf.config.constants import MODEL_NAME
# 从模型列表中导入原子模型枚举
from magic_pdf.model.model_list import AtomicModel
# 导入YOLOv11语言检测模型
from magic_pdf.model.sub_modules.language_detection.yolov11.YOLOv11 import YOLOv11LangDetModel
# 导入DocLayoutYOLO文档布局模型
from magic_pdf.model.sub_modules.layout.doclayout_yolo.DocLayoutYOLO import DocLayoutYOLOModel
# 导入YOLOv8数学公式检测（MFD）模型
from magic_pdf.model.sub_modules.mfd.yolov8.YOLOv8 import YOLOv8MFDModel
# 导入Unimernet数学公式识别（MFR）模型
from magic_pdf.model.sub_modules.mfr.unimernet.Unimernet import UnimernetModel
# 导入基于Pytorch的PaddleOCR模型
from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.pytorch_paddle import PytorchPaddleOCR
# 导入RapidTable表格识别模型
from magic_pdf.model.sub_modules.table.rapidtable.rapid_table import RapidTableModel
# # 尝试导入昇腾（Ascend）插件相关的库和模型
# try:
#     # 导入许可证验证相关的类和函数
#     from magic_pdf_ascend_plugin.libs.license_verifier import (
#         LicenseExpiredError, LicenseFormatError, LicenseSignatureError,
#         load_license)
#     # 导入昇腾NPU版本的PaddleOCR模型
#     from magic_pdf_ascend_plugin.model_plugin.ocr.paddleocr.ppocr_273_npu import ModifiedPaddleOCR
#     # 导入昇腾NPU版本的RapidTable模型
#     from magic_pdf_ascend_plugin.model_plugin.table.rapidtable.rapid_table_npu import RapidTableModel
#     # 加载许可证
#     license_key = load_license()
#     # 记录成功加载昇腾插件和许可证信息的日志
#     logger.info(f'Using Ascend Plugin Success, License id is {license_key["payload"]["id"]},'
#                 f' License expired at {license_key["payload"]["date"]["end_date"]}')
# # 捕获导入或加载许可证过程中可能出现的异常
# except Exception as e:
#     # 如果是导入错误，则忽略（可能未安装昇腾插件）
#     if isinstance(e, ImportError):
#         pass
#     # 如果是许可证格式错误
#     elif isinstance(e, LicenseFormatError):
#         logger.error('Ascend Plugin: Invalid license format. Please check the license file.')
#     # 如果是许可证签名错误
#     elif isinstance(e, LicenseSignatureError):
#         logger.error('Ascend Plugin: Invalid signature. The license may be tampered with.')
#     # 如果是许可证过期错误
#     elif isinstance(e, LicenseExpiredError):
#         logger.error('Ascend Plugin: License has expired. Please renew your license.')
#     # 如果是许可证文件未找到错误
#     elif isinstance(e, FileNotFoundError):
#         logger.error('Ascend Plugin: Not found License file.')
#     # 其他未知错误
#     else:
#         logger.error(f'Ascend Plugin: {e}')
#     # 如果加载昇腾插件失败，则导入非NPU版本的模型
#     from magic_pdf.model.sub_modules.ocr.paddleocr.ppocr_273_mod import ModifiedPaddleOCR
#     # from magic_pdf.model.sub_modules.ocr.paddleocr.ppocr_291_mod import ModifiedPaddleOCR # 备选的OCR模型
#     from magic_pdf.model.sub_modules.table.rapidtable.rapid_table import RapidTableModel

# 定义表格模型初始化函数
def table_model_init(table_model_type, model_path, max_time, _device_='cpu', lang=None, table_sub_model_name=None):
    # 如果表格模型类型是StructTableModel
    if table_model_type == MODEL_NAME.STRUCT_EQTABLE:
        # 导入StructTableModel
        from magic_pdf.model.sub_modules.table.structeqtable.struct_eqtable import StructTableModel
        # 初始化StructTableModel实例
        table_model = StructTableModel(model_path, max_new_tokens=2048, max_time=max_time)
    # 如果表格模型类型是TableMaster
    elif table_model_type == MODEL_NAME.TABLE_MASTER:
        # 导入TableMasterPaddleModel
        from magic_pdf.model.sub_modules.table.tablemaster.tablemaster_paddle import TableMasterPaddleModel
        # 配置模型参数
        config = {
            'model_dir': model_path, # 模型目录
            'device': _device_      # 运行设备
        }
        # 初始化TableMasterPaddleModel实例
        table_model = TableMasterPaddleModel(config)
    # 如果表格模型类型是RapidTable
    elif table_model_type == MODEL_NAME.RAPID_TABLE:
        # 获取原子模型管理器单例
        atom_model_manager = AtomModelSingleton()
        # 获取OCR引擎实例
        ocr_engine = atom_model_manager.get_atom_model(
            atom_model_name='ocr',       # 原子模型名称为OCR
            ocr_show_log=False,          # 不显示OCR日志
            det_db_box_thresh=0.5,       # DB检测框阈值
            det_db_unclip_ratio=1.6,     # DB unclip比例
            lang=lang                    # OCR语言
        )
        # 初始化RapidTableModel实例，传入OCR引擎和子模型名称
        table_model = RapidTableModel(ocr_engine, table_sub_model_name)
    # 如果模型类型不被允许
    else:
        # 记录错误日志
        logger.error('table model type not allow')
        # 退出程序
        exit(1)
    # 返回初始化后的表格模型实例
    return table_model

# 定义数学公式检测（MFD）模型初始化函数
def mfd_model_init(weight, device='cpu'):
    # 如果设备是NPU，转换为torch.device对象
    if str(device).startswith('npu'):
        device = torch.device(device)
    # 初始化YOLOv8MFDModel实例
    mfd_model = YOLOv8MFDModel(weight, device)
    # 返回初始化后的MFD模型实例
    return mfd_model

# 定义数学公式识别（MFR）模型初始化函数
def mfr_model_init(weight_dir, cfg_path, device='cpu'):
    # 初始化UnimernetModel实例
    mfr_model = UnimernetModel(weight_dir, cfg_path, device)
    # 返回初始化后的MFR模型实例
    return mfr_model

# 定义LayoutLMv3布局模型初始化函数
def layout_model_init(weight, config_file, device):
    # 导入Layoutlmv3预测器
    from magic_pdf.model.sub_modules.layout.layoutlmv3.model_init import Layoutlmv3_Predictor
    # 初始化Layoutlmv3_Predictor实例
    model = Layoutlmv3_Predictor(weight, config_file, device)
    # 返回初始化后的布局模型实例
    return model

# 定义DocLayoutYOLO文档布局模型初始化函数
def doclayout_yolo_model_init(weight, device='cpu'):
    # 如果设备是NPU，转换为torch.device对象
    if str(device).startswith('npu'):
        device = torch.device(device)
    # 初始化DocLayoutYOLOModel实例
    model = DocLayoutYOLOModel(weight, device)
    # 返回初始化后的DocLayoutYOLO模型实例
    return model

# 定义语言检测模型初始化函数
def langdetect_model_init(langdetect_model_weight, device='cpu'):
    # 如果设备是NPU，转换为torch.device对象
    if str(device).startswith('npu'):
        device = torch.device(device)
    # 初始化YOLOv11LangDetModel实例
    model = YOLOv11LangDetModel(langdetect_model_weight, device)
    # 返回初始化后的语言检测模型实例
    return model

# 定义OCR模型初始化函数
def ocr_model_init(show_log: bool = False,      # 是否显示日志，默认为False
                   det_db_box_thresh=0.3,       # DB检测框阈值，默认为0.3
                   lang=None,                   # 识别语言，默认为None (通常表示中英文)
                   use_dilation=True,           # 是否使用膨胀处理，默认为True
                   det_db_unclip_ratio=1.8,     # DB unclip比例，默认为1.8
                   ):
    # 如果指定了语言且不为空字符串
    if lang is not None and lang != '':
        # 初始化PytorchPaddleOCR实例，并传入指定语言及其他参数
        # model = ModifiedPaddleOCR( # 备选的PaddleOCR实现
        model = PytorchPaddleOCR(
            show_log=show_log,                  # 是否显示日志
            det_db_box_thresh=det_db_box_thresh,# DB检测框阈值
            lang=lang,                          # 指定语言
            use_dilation=use_dilation,          # 是否使用膨胀
            det_db_unclip_ratio=det_db_unclip_ratio, # DB unclip比例
        )
    # 如果未指定语言或语言为空
    else:
        # 初始化PytorchPaddleOCR实例，不指定语言（通常默认处理中英文）
        # model = ModifiedPaddleOCR( # 备选的PaddleOCR实现
        model = PytorchPaddleOCR(
            show_log=show_log,                  # 是否显示日志
            det_db_box_thresh=det_db_box_thresh,# DB检测框阈值
            use_dilation=use_dilation,          # 是否使用膨胀
            det_db_unclip_ratio=det_db_unclip_ratio, # DB unclip比例
        )
    # 返回初始化后的OCR模型实例
    return model

# 定义原子模型管理器的单例类
class AtomModelSingleton:
    # 类变量，存储唯一的实例
    _instance = None
    # 类变量，用于缓存已初始化的模型
    _models = {}
    # 重写__new__方法以实现单例模式
    def __new__(cls, *args, **kwargs):
        # 如果实例不存在
        if cls._instance is None:
            # 创建实例
            cls._instance = super().__new__(cls)
        # 返回实例
        return cls._instance
    # 获取原子模型的方法
    def get_atom_model(self, atom_model_name: str, **kwargs):
        # 从关键字参数中获取语言信息，默认为None
        lang = kwargs.get('lang', None)
        # 从关键字参数中获取布局模型名称，默认为None
        layout_model_name = kwargs.get('layout_model_name', None)
        # 从关键字参数中获取表格模型名称，默认为None
        table_model_name = kwargs.get('table_model_name', None)
        # 根据原子模型名称和特定参数生成缓存键 (key)
        # OCR模型使用 (模型名, 语言) 作为键
        if atom_model_name in [AtomicModel.OCR]:
            key = (atom_model_name, lang)
        # 布局模型使用 (模型名, 布局模型子名称) 作为键
        elif atom_model_name in [AtomicModel.Layout]:
            key = (atom_model_name, layout_model_name)
        # 表格模型使用 (模型名, 表格模型子名称, 语言) 作为键
        elif atom_model_name in [AtomicModel.Table]:
            key = (atom_model_name, table_model_name, lang)
        # 其他模型直接使用模型名作为键
        else:
            key = atom_model_name
        # 如果模型不在缓存中
        if key not in self._models:
            # 调用atom_model_init函数初始化模型，并将结果存入缓存
            self._models[key] = atom_model_init(model_name=atom_model_name, **kwargs)
        # 返回缓存中的模型实例
        return self._models[key]

# 定义原子模型初始化工厂函数
def atom_model_init(model_name: str, **kwargs):
    # 初始化原子模型变量为None
    atom_model = None
    # 如果模型名称是布局模型
    if model_name == AtomicModel.Layout:
        # 如果布局模型名称是LayoutLMv3
        if kwargs.get('layout_model_name') == MODEL_NAME.LAYOUTLMv3:
            # 调用layout_model_init初始化LayoutLMv3模型
            atom_model = layout_model_init(
                kwargs.get('layout_weights'),      # 获取权重路径
                kwargs.get('layout_config_file'), # 获取配置文件路径
                kwargs.get('device')             # 获取运行设备
            )
        # 如果布局模型名称是DocLayout_YOLO
        elif kwargs.get('layout_model_name') == MODEL_NAME.DocLayout_YOLO:
            # 调用doclayout_yolo_model_init初始化DocLayoutYOLO模型
            atom_model = doclayout_yolo_model_init(
                kwargs.get('doclayout_yolo_weights'), # 获取权重路径
                kwargs.get('device')                  # 获取运行设备
            )
        # 如果布局模型名称不被允许
        else:
            logger.error('layout model name not allow')
            exit(1)
    # 如果模型名称是数学公式检测（MFD）
    elif model_name == AtomicModel.MFD:
        # 调用mfd_model_init初始化MFD模型
        atom_model = mfd_model_init(
            kwargs.get('mfd_weights'), # 获取权重路径
            kwargs.get('device')      # 获取运行设备
        )
    # 如果模型名称是数学公式识别（MFR）
    elif model_name == AtomicModel.MFR:
        # 调用mfr_model_init初始化MFR模型
        atom_model = mfr_model_init(
            kwargs.get('mfr_weight_dir'), # 获取权重目录
            kwargs.get('mfr_cfg_path'),   # 获取配置文件路径
            kwargs.get('device')         # 获取运行设备
        )
    # 如果模型名称是OCR
    elif model_name == AtomicModel.OCR:
        # 调用ocr_model_init初始化OCR模型
        atom_model = ocr_model_init(
            kwargs.get('ocr_show_log'),      # 获取是否显示日志
            kwargs.get('det_db_box_thresh'), # 获取DB检测框阈值
            kwargs.get('lang'),             # 获取语言
        )
    # 如果模型名称是表格模型
    elif model_name == AtomicModel.Table:
        # 调用table_model_init初始化表格模型
        atom_model = table_model_init(
            kwargs.get('table_model_name'),      # 获取表格模型名称
            kwargs.get('table_model_path'),      # 获取表格模型路径
            kwargs.get('table_max_time'),        # 获取表格处理最大时间
            kwargs.get('device'),                # 获取运行设备
            kwargs.get('lang'),                  # 获取语言
            kwargs.get('table_sub_model_name')   # 获取表格子模型名称
        )
    # 如果模型名称是语言检测
    elif model_name == AtomicModel.LangDetect:
        # 如果语言检测模型名称是YOLO_V11_LangDetect
        if kwargs.get('langdetect_model_name') == MODEL_NAME.YOLO_V11_LangDetect:
            # 调用langdetect_model_init初始化语言检测模型
            atom_model = langdetect_model_init(
                kwargs.get('langdetect_model_weight'), # 获取模型权重路径
                kwargs.get('device')                  # 获取运行设备
            )
        # 如果语言检测模型名称不被允许
        else:
            logger.error('langdetect model name not allow')
            exit(1)
    # 如果模型名称不被允许
    else:
        logger.error('model name not allow')
        exit(1)
    # 如果模型初始化失败
    if atom_model is None:
        logger.error('model init failed')
        exit(1)
    # 否则，返回初始化后的模型实例
    else:
        return atom_model