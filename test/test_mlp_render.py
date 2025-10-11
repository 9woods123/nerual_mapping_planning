import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_model.nerual_render_model import NeuralRenderingModel,SimpleMLPModel
from slam_core.ray_casting import RayCasting
from slam_core.keyframe import Keyframe
from slam_core.renderer import Renderer
from network_model.loss_calculate import *
from visualization.mesher import Mesher
from utils.utils import *
from visualization.visual import *

import torch.optim as optim

import torch
import cv2
import numpy as np

def load_color_image(image_path, device="cuda"):
    """
    读取并返回颜色图像 (RGB) -> torch.Tensor
    
    :param image_path: 颜色图像的路径
    :param device: 目标设备 (cpu / cuda)
    :return: torch.Tensor [H, W, 3], dtype=torch.uint8
    """
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR
    if color_image is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")
    
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    color_tensor = torch.from_numpy(color_image).to(device)  # (H,W,3), uint8
    
    return color_tensor


def load_depth_image(image_path, factor=5000.0, device="cuda"):
    """
    读取并返回深度图像 (转为实际深度值) -> torch.Tensor
    
    :param image_path: 深度图像路径
    :param factor: 深度缩放因子 (默认 5000 对应 16-bit 深度)
    :param device: 目标设备 (cpu / cuda)
    :return: torch.Tensor [H, W], dtype=torch.float32
    """
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 原始深度图
    if depth_image is None:
        raise FileNotFoundError(f"无法读取深度图像文件: {image_path}")
    
    depth_image = depth_image.astype(np.float32) / factor
    depth_tensor = torch.from_numpy(depth_image).to(device)  # float32
    
    # Debug 信息
    nonzero_depths = depth_tensor[depth_tensor > 0]
    print("非零点数量:", nonzero_depths.numel())
    if nonzero_depths.numel() > 0:
        print("前20个非零深度值:", nonzero_depths[:20].cpu().numpy())
        print("最小非零深度:", torch.min(nonzero_depths).item())
        print("最大非零深度:", torch.max(nonzero_depths).item())
    
    return depth_tensor





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 步骤 1: 初始化模型
neural_rendering_model = SimpleMLPModel(input_dim=3, hidden_dim=128, num_layers=4)
neural_rendering_model.to(device)  # 将模型移动到设备

optimizer = optim.Adam(neural_rendering_model.parameters(), lr=0.001)


# 步骤 2: 初始化 RayCasting 和 Renderer 类
intrinsic_matrix = np.array([[525.0, 0, 319.5],
                             [0, 525.0, 239.5],
                             [0, 0, 1]])

truncation=0.1  # 10cm




# init_pose = np.eye(4)  # 假设相机位姿是单位矩阵
init_pose = np.array([
    [ 0.9976, -0.0508,  0.0480,  0.4183],
    [-0.0621, -0.3305,  0.9417, -0.4920],
    [-0.0320, -0.9424, -0.3328,  1.6849],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
], dtype=np.float32)


color_map=load_color_image("sensor_data/color/color_1.png")
depth_map=load_depth_image("sensor_data/depth/depth_1.png")

color_map, c_min_val, c_max_val = normalize_torch(color_map, 0, 255)


# 创建 RayCasting 和 Renderer 实例
ray_casting = RayCasting(intrinsic_matrix)
renderer = Renderer(model=neural_rendering_model,tr=truncation)




num_epochs = 200  # 设置训练的轮数


pred_rays_rgbs_tensor=None
sampled_rays_points_tensor=None
all_rays_endpoint_3d=None


all_rays_endpoint_3d_first_frame=None


keyframe_dict=[]
first_keyframe=Keyframe(0.0,init_pose,depth_map,color_map, fx=525.0, fy=525.0, cx=319.5, cy=239.5,
                frame_id=0)
keyframe_dict.append(first_keyframe)


for epoch in range(num_epochs):
    # 步骤 3: 生成射线数据
    
    rays_3d, rgb_values, depths = ray_casting.cast_rays(depth_map, color_map, first_keyframe.c2w, 480, 640)

    # 步骤 4: 沿射线采样
    all_rays_points, all_rays_depths, all_rays_endpoint_3d, all_rays_endpoint_depths = ray_casting.sample_points_along_ray(
    ray_origin=first_keyframe.c2w[:3, 3],  # 射线起点
    rays_direction_world=rays_3d,
    depths_list=depths
    )


    pred_rays_sdfs_tensors, pred_rays_rgbs_tensor = neural_rendering_model(all_rays_points)


    rendered_color, rendered_depth = renderer.render(
        all_rays_depths, 
        pred_rays_sdfs_tensors,
        pred_rays_rgbs_tensor
        )

    total_loss_value,loss_color,loss_depth,loss_surface,loss_free = total_loss(rendered_color, rgb_values, rendered_depth, all_rays_depths, all_rays_endpoint_depths.unsqueeze(-1), pred_rays_sdfs_tensors)

    # 打印损失
    print(f'Epoch {epoch}/{num_epochs}, Loss: {total_loss_value.item()}')

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    total_loss_value.backward()        # 反向传播
    optimizer.step()       # 更新模型参数




mesher = Mesher(min_x=-3, min_y=-3, min_z=-3,
                max_x=3, max_y=3, max_z=3,
                fx=525.0, fy=525.0, cx=319.5, cy=239.5,
                width=640, height=480,
                resolution=0.01)

# 4. 调用 generate_surface_pointcloud
mesher.generate_surface_pointcloud(
    query_fn=renderer.query_sdf_color_function,
    keyframe_dict=keyframe_dict,
    batch_size=65536,
    save_path="./output_surface.ply",
    device=device
)


# import open3d as o3d


# # 假设 all_rays_endpoint_3d 是 (N, 3) torch.Tensor
# points_np = all_rays_endpoint_3d.cpu().numpy()  # 转为 numpy

# # 1️⃣ 射线终点点云
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points_np)
# pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 红色

# # 2️⃣ 世界原点坐标系
# world_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])

# # 3️⃣ 相机初始 pose 坐标系
# cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
# # init_pose 是 T_wc
# cam_coord.transform(init_pose)

# # 4️⃣ 可视化
# o3d.visualization.draw_geometries([pcd, world_coord, cam_coord])
