# params.py
"""
参数管理文件
支持默认值 + 从 YAML 覆盖
"""

from types import SimpleNamespace
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
    "near": 0.1,
    "far": 10.0,
    "forecast_margin": 0.25,
}

mapping_params = {
    "truncation": 0.1,
    "resolution": 0.01,
    "batch_size": 65536,
    "num_epochs": 200,
    "lr": 1e-3,
}

model_params = {
    "input_dim": 3,
    "hidden_dim": 128,
    "num_layers": 4,
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


# ============ 参数类 =============
class Params:
    def __init__(self, yaml_file=None):
        # 默认参数
        self._dict = {
            "camera": camera_params,
            "mapping": mapping_params,
            "model": model_params,
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


# ============ 导出全局 params ============
# 可以选择传入 YAML 文件
params = Params()  # 默认
# params = Params("config.yaml")  # 使用 YAML 覆盖默认参数
