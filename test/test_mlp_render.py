import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_model.nerual_render_model import NeuralRenderingModel,SimpleMLPModel
from slam_core.ray_casting import RayCasting
from slam_core.renderer import Renderer
from network_model.loss_calculate import *
from utils.utils import *
from visualization.visual import *

import torch.optim as optim

import torch
import numpy as np
import cv2
import numpy as np

# fx = 525.0  # focal length x
# fy = 525.0  # focal length y
# cx = 319.5  # optical center x
# cy = 239.5  # optical center y

# factor = 5000 # for the 16-bit PNG files
# OR: factor = 1 # for the 32-bit float images in the ROS bag files


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


    return np.array(color_image)  # 转为 NumPy 数组格式


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

    print("非零点数量:", nonzero_depths.shape[0])
    print("前20个非零深度值:", nonzero_depths[:200])
    print("最小非零深度:", np.min(nonzero_depths))
    print("最大非零深度:", np.max(nonzero_depths))

    return depth_image  # 返回深度图像





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 步骤 1: 初始化模型
neural_rendering_model = SimpleMLPModel(input_dim=3, hidden_dim=256, num_layers=4)
neural_rendering_model.to(device)  # 将模型移动到设备

optimizer = optim.Adam(neural_rendering_model.parameters(), lr=0.001)


# 步骤 2: 初始化 RayCasting 和 Renderer 类
intrinsic_matrix = np.array([[525.0, 0, 319.5],
                             [0, 525.0, 239.5],
                             [0, 0, 1]])

truncation=0.1  # 10cm

pose = np.eye(4)  # 假设相机位姿是单位矩阵


color_map=load_color_image("sensor_data/color.png")
depth_map=load_depth_image("sensor_data/depth.png")
color_map, c_min_val, c_max_val = normalize_numpy(color_map, 0, 255)


# 创建 RayCasting 和 Renderer 实例
ray_casting = RayCasting(intrinsic_matrix)
renderer = Renderer(model=neural_rendering_model,tr=truncation)




num_epochs = 500  # 设置训练的轮数


pred_rays_rgbs_tensor=None
sampled_rays_points_tensor=None
all_rays_endpoint_3d=None


all_rays_endpoint_3d_first_frame=None




for epoch in range(num_epochs):
    # 步骤 3: 生成射线数据
    
    rays_3d, rgb_values, depths = ray_casting.cast_rays(depth_map, color_map, pose, 480, 640)
    target_rgb = torch.from_numpy(np.array(rgb_values, dtype=np.float32)).to(device)  # ndarray -> Tensor

    # 步骤 4: 沿射线采样

    all_rays_points, all_rays_depths, all_rays_endpoint_3d, all_rays_endpoint_depths = ray_casting.sample_points_along_ray(
    ray_origin=np.array([0, 0, 0]),  # 射线起点
    rays_direction_list=rays_3d,
    depths_list=depths
    )

    # all_rays_points: 1115 rays, each with (35, 3) points
    # all_rays_depths: 1115 rays, each with (35,) depths
    # all_rays_endpoint_3d: 1115 rays, each with (3,) points
    # all_rays_endpoint_depths: 1115 rays, each with () depths
    # 假设 all_rays_points 是 list of np.array，每个 shape = (N_samples, 3)

    sampled_rays_points_tensor = torch.tensor(np.stack(all_rays_points, axis=0), dtype=torch.float32).to(device)    # shape = (N_rays, N_samples, 3)
    sampled_rays_depths_tensor = torch.tensor(np.stack(all_rays_depths, axis=0), dtype=torch.float32).to(device)


    # print("===========================epoch============================")
    # print("----------all_rays_points----------:")
    # print(all_rays_points)
    # print("------------sampled_rays_points_tensor-----------:")
    # print(sampled_rays_points_tensor)
    # print("-----------------------------------------------------------:")

    

    target_depth = torch.tensor(np.stack(all_rays_endpoint_depths, axis=0), dtype=torch.float32).to(device)


    # sampled_rays_depths_tensor, d_min_val, d_max_val = normalize_torch(sampled_rays_depths_tensor, 0, 10.0)
    # sampled_rays_surface_depths_tensor, d_min_val, d_max_val = normalize_torch(sampled_rays_surface_depths_tensor, 0, 10.0)
    sampled_rays_points_tensor=sampled_rays_points_tensor/10.0
    pred_geo_features, pred_rays_sdfs_tensors, pred_rays_rgbs_tensor = neural_rendering_model(sampled_rays_points_tensor)

    # 使用 Renderer 类根据模型的输出进行最终渲染
    print("sampled_rays_depths_tensor shape:", sampled_rays_depths_tensor.shape)
    print("pred_rays_sdfs_tensors shape:", pred_rays_sdfs_tensors.shape)

    rendered_color, rendered_depth = renderer.render(
        sampled_rays_depths_tensor, 
        pred_rays_sdfs_tensors,
        pred_rays_rgbs_tensor
        )


# def total_loss(pred_rgb, gt_rgb, pred_d, observe_depth, surface_depths_tensor, pred_sdfs):



    loss = total_loss(rendered_color, target_rgb, rendered_depth, sampled_rays_depths_tensor, target_depth.unsqueeze(-1), pred_rays_sdfs_tensors)

    # 打印损失
    print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 反向传播
    optimizer.step()       # 更新模型参数



# visualize_point_cloud(all_rays_endpoint_3d,rendered_color)

# visualize_point_cloud(points_tensor=sampled_rays_points_tensor,sdf_tensors=pred_rays_sdfs_tensors,color_tensors=pred_rays_rgbs_tensor)

bounding_box = np.array([[-0.8,-0.8,0.5],[0.8,0.8,2.5]])  # 自定义全局范围
voxel_size = 0.01


surface_points = visualize_global_surface(
    query_fn=renderer.query_sdf_color_function, 
    bounding_box=bounding_box, 
    voxel_size=voxel_size, 
    truncation=0.1,
    device='cuda',
    save_path='./global_surface.ply'
)

