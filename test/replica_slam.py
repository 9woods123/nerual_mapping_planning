import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import glob
from utils.params import Params
from slam_core.slam import SLAM
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


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


def load_depth_image_to_tensor(image_path, factor=6553.5, device="cuda"):
    depth = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(image_path)
    depth = depth.astype(np.float32) / factor
    return torch.from_numpy(depth).to(device)


def read_pose_file(pose_file):
    """逐行读取 Replica 的 traj.txt"""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 7:
                tx, ty, tz, qx, qy, qz, qw = vals
                R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = R_mat
                T[:3, 3] = [tx, ty, tz]
                poses.append(T)
            elif len(vals) == 16:
                T = np.array(vals, dtype=np.float32).reshape(4, 4)
                poses.append(T)
            else:
                raise ValueError("Unknown pose format in traj.txt")
    return poses





def run_replica_slam_stream(data_dir, mesh_output_dir="./meshes_replica", device="cuda"):
    """
    单帧逐次加载，避免内存爆炸
    """

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


    os.makedirs(mesh_output_dir, exist_ok=True)

    # === 获取文件列表 ===

    color_paths = sorted(glob.glob(os.path.join(data_dir, "results/frame*.jpg")))
    depth_paths = sorted(glob.glob(os.path.join(data_dir, "results/depth*.png")))
    pose_file = os.path.join(data_dir, "traj.txt")
    poses = read_pose_file(pose_file)

    assert len(color_paths) == len(depth_paths) == len(poses), \
        f"数量不匹配: RGB={len(color_paths)}, Depth={len(depth_paths)}, Pose={len(poses)}"

    # === 初始化 SLAM ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    slam = SLAM(default_params)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Replica frames: {len(color_paths)}")

    # === 主循环 ===
    start_idx = 535  # 从第535帧开始
    for i, (rgb_path, depth_path, pose_np) in enumerate(zip(color_paths[start_idx:], depth_paths[start_idx:], poses[start_idx:]), start=start_idx):
        print(f"\n[Frame {i}/{len(color_paths)}] {os.path.basename(rgb_path)}")

        # --- 每帧单独加载 ---
        color = load_color_image_to_tensor(rgb_path, device=device)
        depth = load_depth_image_to_tensor(depth_path, device=device)

        pose = torch.from_numpy(pose_np).to(device).float()

        # --- 主循环 ---
        slam.main_loop(color, depth, pose, i + 1, mesh_output_dir=mesh_output_dir)

        # --- 释放显存 ---
        del color, depth, pose
        torch.cuda.empty_cache()



if __name__ == "__main__":
    data_path = "sensor_data/Replica/room0"  # 改成你的路径
    run_replica_slam_stream(data_path)
