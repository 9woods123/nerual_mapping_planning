import numpy as np
import torch

# ========== NumPy 版本 ==========
def normalize_numpy(x, min_val=None, max_val=None):
    """
    NumPy 归一化到 [0,1]
    :param x: numpy 数组
    :param min_val: 最小值 (默认自动计算)
    :param max_val: 最大值 (默认自动计算)
    """
    if min_val is None:
        min_val = np.min(x)
    if max_val is None:
        max_val = np.max(x)

    return (x - min_val) / (max_val - min_val + 1e-8), min_val, max_val

def denormalize_numpy(x, min_val, max_val):
    """
    NumPy 反归一化
    """
    return x * (max_val - min_val) + min_val


# ========== PyTorch 版本 ==========
def normalize_torch(x, min_val=None, max_val=None):
    """
    PyTorch 归一化到 [0,1]
    :param x: torch.Tensor
    """
    if min_val is None:
        min_val = torch.min(x)
    if max_val is None:
        max_val = torch.max(x)
    return (x - min_val) / (max_val - min_val + 1e-8), min_val, max_val

def denormalize_torch(x, min_val, max_val):
    """
    PyTorch 反归一化
    """
    return x * (max_val - min_val) + min_val


# ========== 特殊情况 ==========
def normalize_coordinates(coords, center=None, scale=None):
    """
    坐标归一化到 [-1,1]
    :param coords: numpy 或 torch，形状 (..., 3)
    """
    if isinstance(coords, np.ndarray):
        if center is None:
            center = np.mean(coords, axis=0)
        if scale is None:
            scale = np.max(np.linalg.norm(coords - center, axis=1))
        return (coords - center) / (scale + 1e-8), center, scale
    elif isinstance(coords, torch.Tensor):
        if center is None:
            center = torch.mean(coords, dim=0)
        if scale is None:
            scale = torch.max(torch.norm(coords - center, dim=1))
        return (coords - center) / (scale + 1e-8), center, scale
    else:
        raise TypeError("coords 必须是 numpy 或 torch")
    
    
def denormalize_coordinates(coords, center, scale):
    """ 反归一化坐标 """
    return coords * scale + center
