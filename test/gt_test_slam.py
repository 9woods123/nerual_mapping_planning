import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.params import Params
from slam_core.slam import SLAM
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

def downsample_color_image_torch(img_tensor, target_h=240, target_w=320):
    # img_tensor: [H, W, 3] -> 需要先转成 [1, 3, H, W]
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    img_resized = F.interpolate(img_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return img_resized.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

def downsample_depth_image_torch(depth_tensor, target_h=240, target_w=320):
    # depth_tensor: [H, W] -> [1, 1, H, W]
    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
    depth_resized = F.interpolate(depth_tensor, size=(target_h, target_w), mode='nearest')
    return depth_resized.squeeze(0).squeeze(0)

# ======================
# === 工具函数部分 ===
# ======================

def load_image_list(txt_path):
    """读取 rgb.txt 或 depth.txt"""
    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            t, path = line.strip().split()
            data.append((float(t), path))
    return data


def load_poses(gt_file):
    """加载 TUM ground truth pose 文件"""
    poses = []
    with open(gt_file, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            vals = list(map(float, line.strip().split()))
            t, tx, ty, tz, qx, qy, qz, qw = vals
            R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = [tx, ty, tz]
            poses.append((t, T))
    return poses


def find_nearest_timestamp(target_t, time_list):
    arr = np.array(time_list)
    idx = np.argmin(np.abs(arr - target_t))
    return idx, arr[idx]



def align_rgb_depth_pose(rgb_list, depth_list, pose_list, max_diff=0.02):
    rgbd_pose_list = []

    depth_times = [t for t, _ in depth_list]
    pose_times = [t for t, _ in pose_list]

    for t_rgb, rgb_path in rgb_list:
        idx_d, t_d = find_nearest_timestamp(t_rgb, depth_times)
        idx_p, t_p = find_nearest_timestamp(t_rgb, pose_times)

        if abs(t_rgb - t_d) < max_diff and abs(t_rgb - t_p) < max_diff:
            rgbd_pose_list.append((
                t_rgb,
                rgb_list[idx_d][1],
                depth_list[idx_d][1],
                pose_list[idx_p][1]
            ))
    return rgbd_pose_list



# ======================
# === 图像加载函数 ===
# ======================

def load_color_image_to_tensor(image_path, K=None, dist_coeffs=None, device="cuda"):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if K is not None and dist_coeffs is not None:
        img = cv2.undistort(img, K, dist_coeffs)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).to(device)


def load_depth_image_to_tensor(image_path, factor=5000.0, device="cuda"):
    depth = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(image_path)
    depth = depth.astype(np.float32) / factor
    return torch.from_numpy(depth).to(device)


# ======================
# === 主程序部分 ===
# ======================

if __name__ == "__main__":

    default_params = Params()
    # 相机内参
    fx, fy, cx, cy = default_params.camera.fx, default_params.camera.fy, default_params.camera.cx, default_params.camera.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    distCoeffs = np.array([
        default_params.camera.distortion.k1,
        default_params.camera.distortion.k2,
        default_params.camera.distortion.p1,
        default_params.camera.distortion.p2,
        default_params.camera.distortion.k3
    ], np.float32)

    # === ✅ 对齐三者 ===
    data_path="sensor_data/rgbd_dataset_freiburg1_360/"
    rgb_list = load_image_list("sensor_data/rgbd_dataset_freiburg1_360/rgb.txt")
    depth_list = load_image_list("sensor_data/rgbd_dataset_freiburg1_360/depth.txt")
    pose_list = load_poses("sensor_data/rgbd_dataset_freiburg1_360/groundtruth.txt")

    aligned_frames = align_rgb_depth_pose(rgb_list, depth_list, pose_list)
    print(f"有效对齐帧数: {len(aligned_frames)}")


    target_h, target_w = 120, 160  # 降采样目标尺寸
    
    # --- 缩放相机内参 ---

    scale_x = target_w /  default_params.camera.width
    scale_y = target_h /  default_params.camera.height
    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y
    K_new = np.array([[fx_new, 0, cx_new],
                        [0, fy_new, cy_new],
                        [0, 0, 1]], np.float32)
    
    default_params.camera.fx, default_params.camera.fy, default_params.camera.cx, default_params.camera.cy=\
    fx_new,fy_new,cx_new,cy_new
    
    default_params.camera.width=target_w
    default_params.camera.height=target_h


    slam = SLAM(default_params)

    for i, (t, rgb_path, depth_path, pose_np) in enumerate(aligned_frames):
        color = load_color_image_to_tensor(data_path+rgb_path, K, distCoeffs)
        depth = load_depth_image_to_tensor(data_path+depth_path)

        # 降采样
        color = downsample_color_image_torch(color, target_h, target_w)
        depth = downsample_depth_image_torch(depth, target_h, target_w)


        pose = torch.from_numpy(pose_np).to("cuda").float()
        slam.main_loop(color, depth, pose, i+1, mesh_output_dir="./meshes")

