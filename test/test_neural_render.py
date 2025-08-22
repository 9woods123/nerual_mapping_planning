import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_model.nerual_render_model import NeuralRenderingModel
from slam_core.ray_casting import RayCasting
from slam_core.renderer import Renderer
from network_model.loss_calculate import *

import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 步骤 1: 初始化模型
neural_rendering_model = NeuralRenderingModel(input_dim=3, hidden_dim=64, encoding_dim=64)
neural_rendering_model.to(device)  # 将模型移动到设备

# 步骤 2: 初始化 RayCasting 和 Renderer 类
intrinsic_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])  # 相机内参矩阵
pose = np.eye(4)  # 假设相机位姿是单位矩阵

depth_map = np.random.rand(480, 640)  # 假设深度图
color_map = np.random.rand(480, 640, 3)  # 假设颜色图

# 创建 RayCasting 和 Renderer 实例
ray_casting = RayCasting(intrinsic_matrix)
renderer = Renderer(neural_rendering_model)

# 步骤 3: 生成射线数据
rays_3d, rgb_values, depths = ray_casting.cast_rays(depth_map, color_map, pose, 480, 640)

# 步骤 4: 沿射线采样
sampled_points, sampled_depths = ray_casting.sample_points_along_ray(
    ray_origin=np.array([0, 0, 0]),  # 射线起点
    rays_direction_list=rays_3d,
    depths_list=depths
)


sampled_points_tensor = torch.tensor(np.array(sampled_points), dtype=torch.float32).to(device)
sampled_depths_tensor = torch.tensor(np.array(sampled_depths), dtype=torch.float32).to(device)


# 获取模型的输出
geo_features, sdfs_tensors, rgbs_tensor = neural_rendering_model(sampled_points_tensor)



# 步骤 6: 使用 Renderer 类根据模型的输出进行最终渲染
rendered_color, rendered_depth = renderer.render(
    sampled_points_tensor, 
    sampled_depths_tensor, 
    rgbs_tensor
)



# 打印渲染结果
print("Rendered Color:", rendered_color)
print("Rendered Depth:", rendered_depth)
