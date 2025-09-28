import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.params import Params

from slam_core.slam import SLAM
import torch
import cv2
import os
import numpy as np
import cv2
import numpy as np
import torch



def load_color_image_to_tensor(image_path, K=None, dist_coeffs=None, device="cuda", visualize=False):
    """
    读取颜色图像并转换为 torch.Tensor，范围 [0,1]，可选去畸变
    :param image_path: 图像路径
    :param K: 相机内参矩阵 (3x3 numpy array)
    :param dist_coeffs: 畸变系数 (1xN numpy array, 一般是 1x5)
    :param device: torch device
    :param visualize: 是否显示去畸变对比
    :return: torch.Tensor, shape (H, W, 3), dtype=float32
    """
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_image is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    undistorted = color_image.copy()
    if K is not None and dist_coeffs is not None:
        undistorted = cv2.undistort(color_image, K, dist_coeffs)

    if visualize:
        # 拼接原图和去畸变图
        combined = np.hstack((color_image, undistorted))
        # OpenCV 显示需要 BGR
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imshow("Original (Left) | Undistorted (Right)", combined_bgr)
        cv2.waitKey(0)  # 按任意键关闭窗口
        cv2.destroyAllWindows()

    undistorted = undistorted.astype(np.float32) / 255.0  # 归一化到 [0,1]
    color_tensor = torch.from_numpy(undistorted).to(device)
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



# ========================
# FrameLoader 类
# ========================
class FrameLoader:
    def __init__(self, color_dir, depth_dir, camera_K, camera_distortion):
        self.color_dir = color_dir
        self.depth_dir = depth_dir
        self.K = camera_K
        self.distCoeffs = camera_distortion

    def load_frame(self, idx):
        file_name = f"{idx:04d}.png"
        color_file = os.path.join(self.color_dir, file_name)
        depth_file = os.path.join(self.depth_dir, file_name)

        color_tensor = load_color_image_to_tensor(
            color_file,
            K=self.K,
            dist_coeffs=self.distCoeffs
        )

        depth_tensor = load_depth_image_to_tensor(depth_file)

        return color_tensor, depth_tensor


# ========================
# 主程序
# ========================
if __name__ == "__main__":
    default_params = Params()



    # 相机内参矩阵
    fx = default_params.camera.fx
    fy = default_params.camera.fy
    cx = default_params.camera.cx
    cy = default_params.camera.cy
    camera_K = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0,  0,  1]], dtype=np.float32)

    # 畸变参数
    camera_distortion = np.array([
        default_params.camera.distortion.k1,
        default_params.camera.distortion.k2,
        default_params.camera.distortion.p1,
        default_params.camera.distortion.p2,
        default_params.camera.distortion.k3
    ], dtype=np.float32)

    # 帧加载器
    frame_loader = FrameLoader(
        "sensor_data/rgbd_dataset_freiburg1_360/rgb_renamed",
        "sensor_data/rgbd_dataset_freiburg1_360/depth_renamed",
        camera_K,
        camera_distortion
    )
    slam = SLAM(default_params)
    num_frames = 900
    for i in range(num_frames):
        color, depth = frame_loader.load_frame(i+1)
        slam.main_loop(color, depth, i+1, mesh_output_dir="./meshes")