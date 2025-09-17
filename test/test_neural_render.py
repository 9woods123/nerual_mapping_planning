import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_model.nerual_render_model import NeuralRenderingModel
from slam_core.ray_casting import RayCasting
from slam_core.renderer import Renderer
from network_model.loss_calculate import *
from utils.utils import *

import torch.optim as optim

import torch
import numpy as np
import cv2

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


def load_depth_image(image_path, factor=5000):
    """
    读取并返回深度图像，并将其转化为真实的深度值
    
    :param image_path: 深度图像的路径
    :param factor: 深度图像的缩放因子，通常为 5000 对应 16-bit 图像
    :return: 真实深度图 (height, width) 类型为 float32
    """
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 读取为原始深度图像（16-bit 或 32-bit）
    
    if depth_image is None:
        raise FileNotFoundError(f"无法读取深度图像文件: {image_path}")
    
    # 转换为实际的深度值
    depth_image = depth_image.astype(np.float32) / factor
    return depth_image  # 返回深度图像



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 步骤 1: 初始化模型
neural_rendering_model = NeuralRenderingModel(input_dim=3, hidden_dim=256, encoding_dim=256)
neural_rendering_model.to(device)  # 将模型移动到设备

optimizer = optim.Adam(neural_rendering_model.parameters(), lr=0.01)


# 步骤 2: 初始化 RayCasting 和 Renderer 类
intrinsic_matrix = np.array([[525.0, 0, 319.5],
                             [0, 525.0, 239.5],
                             [0, 0, 1]])

pose = np.eye(4)  # 假设相机位姿是单位矩阵


color_map=load_color_image("sensor_data/color.png")
depth_map=load_depth_image("sensor_data/depth.png")
color_map, c_min_val, c_max_val = normalize_numpy(color_map, 0, 255)





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

# 转成 Tensor 并移动到设备
sampled_points_tensor = torch.tensor(np.array(sampled_points), dtype=torch.float32).to(device)
sampled_depths_tensor = torch.tensor(np.array(sampled_depths), dtype=torch.float32).to(device)
sampled_depths_tensor, d_min_val, d_max_val = normalize_torch(sampled_depths_tensor, 0, 10.0)
# 假设点范围在 [0, depth_max]
scale = 10.0  # 或者 depth_map 的最大深度
sampled_points_tensor = sampled_points_tensor / scale  # 归一化到 0~1
sampled_points_tensor = sampled_points_tensor * 2 - 1   # 再到 -1~1



rgb_values = np.array(rgb_values, dtype=np.float32)  # list -> ndarray
rgb_values = torch.from_numpy(rgb_values).to(device)  # ndarray -> Tensor

num_epochs = 100  # 设置训练的轮数
for epoch in range(num_epochs):

    pred_geo_features, pred_sdfs_tensors, pred_rgbs_tensor = neural_rendering_model(sampled_points_tensor)

    # 使用 Renderer 类根据模型的输出进行最终渲染
    rendered_color, rendered_depth = renderer.render(
        sampled_depths_tensor, 
        pred_rgbs_tensor,
        pred_sdfs_tensors
    )

    # 计算颜色损失
    loss = total_loss(rendered_color, rgb_values, rendered_depth, sampled_depths_tensor)

    # 打印损失
    print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 反向传播
    optimizer.step()       # 更新模型参数



# 打印渲染结果
print("Rendered Color:", rendered_color)
print("Rendered Depth:", rendered_depth)
