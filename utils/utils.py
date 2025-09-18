import numpy as np
import torch
import cv2






def load_color_image(image_path):
    """
    读取并返回颜色图像
    
    :param image_path: 颜色图像的路径
    :return: 读取的颜色图像 (height, width, 3) 类型为 uint8
    """
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读取为彩色图像
    
    if color_image is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")
    
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

    color_image_np = np.array(color_image)
    color, c_min_val, c_max_val = normalize_numpy(color_image_np, 0, 255)

    return color  


def load_depth_image(image_path, factor=5000.0):
    """
    读取并返回深度图像，并将其转化为真实的深度值
    
    :param image_path: 深度图像的路径
    :param factor: 深度图像的缩放因子，通常为 5000 对应 16-bit 图像
    :return: 真实深度图 (height, width) 类型为 float32
    """

    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 读取为原始深度图像（16-bit 或 32-bit）
    print(depth_image.dtype, depth_image.shape)


    if depth_image is None:
        raise FileNotFoundError(f"无法读取深度图像文件: {image_path}")
    
    # 转换为实际的深度值
    depth_image = depth_image.astype(np.float32) / factor

    nonzero_depths = depth_image[depth_image > 0]

    # print("非零点数量:", nonzero_depths.shape[0])
    # print("前20个非零深度值:", nonzero_depths[:200])
    # print("最小非零深度:", np.min(nonzero_depths))
    # print("最大非零深度:", np.max(nonzero_depths))

    return depth_image  # 返回深度图像







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

    return (x - min_val) / (max_val - min_val), min_val, max_val

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
