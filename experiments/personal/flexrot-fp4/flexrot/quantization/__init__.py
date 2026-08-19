"""NVFP4 与 MXFP4 的可审计 fake-quant 参考实现。"""

from .mxfp4 import fake_quant_mxfp4
from .nvfp4 import calibrate_nvfp4_global_scale, fake_quant_nvfp4

__all__ = [
    "calibrate_nvfp4_global_scale",
    "fake_quant_mxfp4",
    "fake_quant_nvfp4",
]
