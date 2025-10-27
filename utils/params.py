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

# Freiburg 1 RGB 	517.3 	516.5 	318.6 	255.3 	0.2624	-0.9531	-0.0054	0.0026 	1.1633 
# {  repica
#     "camera": {
#         "w": 1200,
#         "h": 680,
#         "fx": 600.0,
#         "fy": 600.0,
#         "cx": 599.5,
#         "cy": 339.5,
#         "scale": 6553.5
#     }
# }
camera_params = {
    "fx": 600.0,
    "fy": 600.0,
    "cx": 599.5,
    "cy": 339.5,

    "width": 1200,
    "height": 680,
    "near": 0.2,
    "far": 3.5,
    "distortion": {
        "k1": 0.2624,
        "k2": -0.9531,
        "p1": -0.0054,
        "p2": 0.0026,
        "k3": 1.1633
    },
}

# camera_params = {
#     "fx": 517.3,
#     "fy": 516.5,
#     "cx": 318.6,
#     "cy": 255.3,

#     "width": 640,
#     "height": 480,
#     "near": 0.2,
#     "far": 2.5,
#     "distortion": {
#         "k1": 0.2624,
#         "k2": -0.9531,
#         "p1": -0.0054,
#         "p2": 0.0026,
#         "k3": 1.1633
#     },
# }


bounding_box ={
    "min_x": -5.0,
    "max_x": 5.0,
    "min_y": -5.0,
    "max_y": 5.0,
    "min_z": -5.0,
    "max_z": 5.0,
}



mapping_params = {
    "truncation": 0.1,
    "resolution": 0.01,
    "batch_size": 65536,
    "lr": 0.001,         # Mapper 学习率
    "iters": 50,       # Mapper 内部优化迭代次数
    "sample_ratio": 0.005,  # 对输入图像下采样比例
    "mesh_every":100,
    "keyframe_every":20,
}


tracking_params = {
    "lr": 0.0005,           # 优化位姿学习率
    "iters": 50,          # 位姿优化迭代次数
    "sample_ratio": 0.005,  # 对输入图像下采样比例
    "ignore_edge_H":100,
    "ignore_edge_W":100
}


model_params = {
    "input_dim": 3,
    "hidden_dim": 512,
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

