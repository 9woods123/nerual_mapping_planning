import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("Agg")  # 使用无界面的后端

import os

import torch


def orthogonalize_rotation(R: torch.Tensor) -> torch.Tensor:
    """
    对旋转矩阵进行正交化，保证 R^T R = I
    使用 SVD 方法。
    
    Args:
        R: (3,3) 旋转矩阵，可能有数值误差
    
    Returns:
        R_ortho: (3,3) 正交化后的旋转矩阵
    """
    U, _, Vt = torch.linalg.svd(R)
    R_ortho = U @ Vt
    return R_ortho


def compute_pose_error(gt_pose, est_pose, device="cpu"):
    """
    计算姿态误差（旋转和平移）
    Args:
        gt_pose: (4,4) torch.Tensor 或 numpy.ndarray, 真值位姿
        est_pose: (4,4) torch.Tensor 或 numpy.ndarray, 估计位姿
        device: str, 计算设备

    Returns:
        trans_error: float, 平移误差（米）
        rot_error_deg: float, 旋转误差（角度）
    """
    if not isinstance(gt_pose, torch.Tensor):
        gt_pose = torch.tensor(gt_pose, dtype=torch.float32, device=device)
    if not isinstance(est_pose, torch.Tensor):
        est_pose = torch.tensor(est_pose, dtype=torch.float32, device=device)

    # 平移误差（欧式距离）
    trans_error = torch.norm(gt_pose[:3, 3] - est_pose[:3, 3]).item()

    # 旋转误差
    R_gt = gt_pose[:3, :3]
    R_est = est_pose[:3, :3]
    dR = R_gt @ R_est.T
    cos_angle = torch.clamp((torch.trace(dR) - 1) / 2, -1.0, 1.0)
    rot_error_deg = (torch.acos(cos_angle) * 180.0 / torch.pi).item()

    return trans_error, rot_error_deg


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


def load_color_image_to_tensor(image_path, device="cuda"):
    """
    读取颜色图像并转换为 torch.Tensor，范围 [0,1]
    :param image_path: 图像路径
    :param device: torch device
    :return: torch.Tensor, shape (H, W, 3), dtype=float32
    """
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_image is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_image = color_image.astype(np.float32) / 255.0  # 归一化到 [0,1]
    color_tensor = torch.from_numpy(color_image).to(device)
    return color_tensor


def load_depth_image_to_tensor(image_path, factor=5000.0, device="cuda"):
    """
    读取深度图并转换为 torch.Tensor，单位为米
    :param image_path: 深度图路径
    :param factor: 深度缩放因子
    :param device: torch device
    :return: torch.Tensor, shape (H, W), dtype=float32
    """
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise FileNotFoundError(f"无法读取深度图像文件: {image_path}")

    depth_image = depth_image.astype(np.float32) / factor
    depth_tensor = torch.from_numpy(depth_image).to(device)
    return depth_tensor

def save_loss_curve(losses, index, save_dir):
    """
    保存单帧的 loss 曲线图
    :param losses: list[float] 每次迭代的 loss
    :param index: int 帧编号
    :param save_dir: str 保存目录（绝对路径）
    """
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, f"loss_curve_{index:04d}.png")

    plt.figure()
    plt.plot(losses, marker='o', linewidth=1)
    plt.title(f"Loss Curve (frame {index})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_keyframe_trajectory(keyframes, save_path, title="Camera Trajectory", axis_length=0.1):
    """
    保存 keyframe 相机轨迹到 PNG，并显示相机朝向
    :param keyframes: dict, 每个 value 需要有 .c2w (4x4) 相机位姿矩阵
    :param save_path: 保存路径 (png 文件)
    :param title: 图像标题
    :param axis_length: 相机坐标系箭头长度
    """
    poses = []
    for kf in keyframes:
        c2w = kf.c2w.detach().cpu().numpy()  # (4,4)
        cam_pos = c2w[:3, 3]  # 相机位置
        poses.append((cam_pos, c2w[:3, :3]))  # (位置, 旋转矩阵)

    if len(poses) == 0:
        print("⚠️ No keyframes to plot.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 相机位置轨迹
    cam_positions = np.array([p[0] for p in poses])
    ax.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 'b-', label="Trajectory")
    ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], c='r', s=10, label="Keyframes")

    # 每个相机画坐标轴
    for pos, rot in poses:
        x_axis = rot[:, 0] * axis_length
        y_axis = rot[:, 1] * axis_length
        z_axis = rot[:, 2] * axis_length

        ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], color='r', linewidth=1)
        ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], color='g', linewidth=1)
        ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], color='b', linewidth=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    # print(f"✅ Saved trajectory with orientations to {save_path}")

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
