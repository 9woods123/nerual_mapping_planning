# params.py
"""
参数管理文件
支持默认值 + 从 YAML 覆盖
"""

from types import SimpleNamespace
from typing import Optional

import yaml
import os


# ---------------- 默认参数 ----------------
camera_params = {
    "fx": 525.0,
    "fy": 525.0,
    "cx": 319.5,
    "cy": 239.5,
    "width": 640,
    "height": 480,
    "near": 0.2,
    "far": 2.5,
    "forecast_margin": 0.25,
}



bounding_box ={
    "min_x": -5.0,
    "max_x": 5.0,
    "min_y": -5.0,
    "max_y": 5.0,
    "min_z": -1.0,
    "max_z": 5.0,
}



mapping_params = {
    "truncation": 0.1,
    "resolution": 0.01,
    "batch_size": 65536,
    "lr": 1e-3,         # Mapper 学习率
    "iters": 100,       # Mapper 内部优化迭代次数
    "downsample_ratio": 0.00005,  # 对输入图像下采样比例
}


tracking_params = {
    "lr": 0.002,           # 优化位姿学习率
    "iters": 40,          # 位姿优化迭代次数
    "downsample_ratio": 0.0001,  # 对输入图像下采样比例
}


model_params = {
    "input_dim": 3,
    "hidden_dim": 256,
    "num_layers": 8,
}


# ============ 工具函数 =============
def dict_to_namespace(d):
    """递归把 dict 转成 namespace"""
    ns = SimpleNamespace(**d)
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, dict_to_namespace(v))
    return ns


def merge_dict(d, u):
    """递归合并 u 到 d"""
    for k, v in u.items():
        if isinstance(v, dict) and k in d:
            merge_dict(d[k], v)
        else:
            d[k] = v



class Params:
    camera: Optional[SimpleNamespace]
    mapping: Optional[SimpleNamespace]
    model: Optional[SimpleNamespace]
    bounding_box: Optional[SimpleNamespace]
    mesher: Optional[SimpleNamespace]    
    def __init__(self, yaml_file=None):
        # 默认参数
        self._dict = {
            "camera": camera_params,
            "mapping": mapping_params,
            "model": model_params,
            "bounding_box":bounding_box,
            "tracking":tracking_params
        }

        # 如果有 yaml 文件，覆盖默认参数
        if yaml_file is not None and os.path.exists(yaml_file):
            with open(yaml_file, "r") as f:
                yaml_dict = yaml.safe_load(f)
            merge_dict(self._dict, yaml_dict)

        # 生成 namespace
        self._ns = dict_to_namespace(self._dict)

    # 支持 dict 风格访问
    def __getitem__(self, item):
        return self._dict[item]

    # 支持属性访问
    def __getattr__(self, item):
        return getattr(self._ns, item)

