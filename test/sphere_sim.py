import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_model.nerual_render_model import NeuralRenderingModel, SimpleMLPModel
from slam_core.ray_casting import RayCasting
from slam_core.renderer import Renderer
from network_model.loss_calculate import *
from utils.utils import *
from visualization.visual import *

import torch.optim as optim
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def ray_sphere_intersection(ray_o, ray_d, sphere_center, radius=0.5):
    """
    射线和球体交点
    ray_o: (3,) 射线起点
    ray_d: (3,) 射线方向 (归一化)
    sphere_center: (3,) 球心
    """
    o_c = ray_o - sphere_center
    a = torch.dot(ray_d, ray_d)
    b = 2.0 * torch.dot(o_c, ray_d)
    c = torch.dot(o_c, o_c) - radius**2

    disc = b * b - 4 * a * c
    if disc < 0:
        return None, None  # 如果没有交点，返回 None

    t0 = (-b - torch.sqrt(disc)) / (2 * a)
    t1 = (-b + torch.sqrt(disc)) / (2 * a)

    # 选择最小的正交点
    if t0 > 0:
        return t0, ray_o + t0 * ray_d
    elif t1 > 0:
        return t1, ray_o + t1 * ray_d
    else:
        return None, None  # 如果两个交点都不在射线前方，返回 None

def generate_rays(H, W, fov=60):
    """
    生成像素射线
    H, W: 图像尺寸
    fov: 视场角 (度)
    """
    fov = torch.tensor(fov * torch.pi / 180.0)  # 转弧度
    aspect = W / H
    fx = fy = 0.5 / torch.tan(fov / 2.0)

    rays_o = []
    rays_d = []
    for i in range(H):
        for j in range(W):
            # 像素归一化到 [-1,1]
            x = (j + 0.5) / W * 2 - 1
            y = (i + 0.5) / H * 2 - 1
            x = x * aspect

            # 相机坐标系方向
            d = torch.tensor([x / fx, -y / fy, 1.0])
            d = d / torch.norm(d)

            rays_o.append(torch.tensor([0.0, 0.0, 0.0]))  # 相机原点
            rays_d.append(d)

    return torch.stack(rays_o), torch.stack(rays_d)

def generate_rgb(depths):
    """
    根据深度生成模拟的 RGB 值
    只有当深度大于 0 时才生成 RGB 值，并确保深度和 RGB 数组维度一致
    """
    # 创建一个与原 depths 相同形状的全零 RGB 值
    rgb_full = torch.zeros_like(depths).unsqueeze(-1).expand(-1, 3)  # [N, 3]

    # 过滤出大于 0 的深度值
    valid_depths_mask = depths > 0  # 创建一个掩码，用于筛选有效的深度值
    
    # 如果没有有效深度，直接返回
    if valid_depths_mask.sum() == 0:
        return rgb_full

    # 对有效深度值进行归一化
    valid_depths = depths[valid_depths_mask]
    norm_depth = torch.clamp(valid_depths / torch.max(valid_depths), 0, 1)
    
    # 根据深度生成 RGB 值，这里从红色到蓝色渐变
    rgb_valid = torch.stack([norm_depth, torch.zeros_like(norm_depth), 1 - norm_depth], dim=-1)

    # 将有效的 RGB 填充到对应位置
    print(f"Shape of rgb_valid: {rgb_valid.shape}")  # (ray_num, 1)

    return rgb_valid



if __name__ == "__main__":
    H, W = 50, 50  # 分辨率
    rays_o, rays_d = generate_rays(H, W, fov=60)

    sphere_center = torch.tensor([0.0, 0.0, 3.0])
    radius = 1.5

    depths = []
    intersection_points = []
    rgb_values = []

    for o, d in zip(rays_o, rays_d):
        t, intersection_point = ray_sphere_intersection(o, d, sphere_center, radius)
        if t is None:
            depths.append(torch.tensor(0.0))  # 没打到球
            intersection_points.append(torch.tensor([0.0, 0.0, 0.0]))  # 没打到球，设置交点为零
        else:
            depths.append(t)
            intersection_points.append(intersection_point)

    # 将深度值转换为RGB
    depths = torch.stack(depths)
    rgb_values = generate_rgb(depths)

    # 将交点保存为 3D 坐标
    rays_3d = torch.stack(intersection_points)  # 射线与球体交点的 3D 坐标

    # 扁平化所有的列表，按照 [ray_num, value] 格式
    rays_3d_flat = rays_3d.view(-1, 3)  # 每个射线的 3D 坐标
    rgb_values_flat = rgb_values.view(-1, 3)  # 每个像素的 RGB 值
    depths_flat = depths.view(-1, 1)  # 每个像素的深度值

    # 将 tensors 转换为 numpy 格式
    rays_3d_flat_numpy = rays_3d_flat.cpu().numpy()  # [ray_num, 3]
    rgb_values_flat_numpy = rgb_values_flat.cpu().numpy()  # [ray_num, 3]
    depths_flat_numpy = depths_flat.cpu().numpy()  # [ray_num, 1]

    # 可视化深度图
    depth_map = depths.reshape(H, W)
    plt.imshow(depth_map.numpy(), cmap="plasma")
    plt.colorbar(label="Depth")
    plt.title("Sphere Depth Map")
    plt.show()

    # 打印输出
    print(f"Shape of rays_3d_flat: {rays_3d_flat_numpy.shape}")  # (ray_num, 3)
    print(f"Shape of rgb_values_flat: {rgb_values_flat_numpy.shape}")  # (ray_num, 3)
    print(f"Shape of depths_flat: {depths_flat_numpy.shape}")  # (ray_num, 1)

    # 打印一些数据
    print("Max depth:", torch.max(depth_map).item())
    print("Some intersection points (3D):", rays_3d_flat_numpy[0])  # 打印第一个像素的交点位置
    print("Some RGB values:", rgb_values_flat_numpy[0])  # 打印第一个像素的RGB值

    # 你可以根据需要将 rays_3d_flat_numpy, rgb_values_flat_numpy 和 depths_flat_numpy 导出或使用


    # print("Max depth:", torch.max(depth_map).item())
    # print("Some intersection points (3D):", rays_3d_flat.numpy())  # 打印第一个像素的交点位置
    # print("Some RGB values:", rgb_values_flat.numpy())  # 打印第一个像素的RGB值

    # 你可以根据需要将 rays_3d_flat, rgb_values_flat 和 depths_flat 导出或使用

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 步骤 1: 初始化模型
    neural_rendering_model = SimpleMLPModel(input_dim=3, hidden_dim=256, num_layers=5)
    neural_rendering_model.to(device)  # 将模型移动到设备

    optimizer = optim.Adam(neural_rendering_model.parameters(), lr=0.001)
    # 步骤 2: 初始化 RayCasting 和 Renderer 类
    intrinsic_matrix = np.array([[525.0, 0, 319.5],
                                [0, 525.0, 239.5],
                                [0, 0, 1]])

    truncation=0.1  # 10cm

    pose = np.eye(4)  # 假设相机位姿是单位矩阵



    # 创建 RayCasting 和 Renderer 实例
    ray_casting = RayCasting(intrinsic_matrix)
    renderer = Renderer(model=neural_rendering_model,tr=truncation)



    num_epochs = 100  # 设置训练的轮数


    pred_rays_rgbs_tensor=None
    sampled_rays_points_tensor=None
    all_rays_endpoint_3d=None


    all_rays_endpoint_3d_first_frame=None




    for epoch in range(num_epochs):
        # 步骤 3: 生成射线数据
        
        target_rgb = torch.from_numpy(np.array(rgb_values_flat_numpy, dtype=np.float32)).to(device)  # ndarray -> Tensor

        # 步骤 4: 沿射线采样

        all_rays_points, all_rays_depths, all_rays_endpoint_3d, all_rays_endpoint_depths = ray_casting.sample_points_along_ray(
        ray_origin=np.array([0, 0, 0]),  # 射线起点
        rays_direction_list=rays_3d_flat_numpy,
        depths_list=depths_flat_numpy
        )


        # all_rays_points: 1115 rays, each with (35, 3) points
        # all_rays_depths: 1115 rays, each with (35,) depths
        # all_rays_endpoint_3d: 1115 rays, each with (3,) points
        # all_rays_endpoint_depths: 1115 rays, each with () depths
        # 假设 all_rays_points 是 list of np.array，每个 shape = (N_samples, 3)

        sampled_rays_points_tensor = torch.tensor(np.stack(all_rays_points, axis=0), dtype=torch.float32).to(device)    # shape = (N_rays, N_samples, 3)
        sampled_rays_depths_tensor = torch.tensor(np.stack(all_rays_depths, axis=0), dtype=torch.float32).to(device)
        sampled_rays_depths_tensor=sampled_rays_depths_tensor.squeeze(-1)

        target_depth = torch.tensor(np.stack(all_rays_endpoint_depths, axis=0), dtype=torch.float32).to(device)
        target_depth=target_depth.squeeze(-1)


        # sampled_rays_depths_tensor, d_min_val, d_max_val = normalize_torch(sampled_rays_depths_tensor, 0, 10.0)
        # sampled_rays_surface_depths_tensor, d_min_val, d_max_val = normalize_torch(sampled_rays_surface_depths_tensor, 0, 10.0)
        # sampled_rays_points_tensor=sampled_rays_points_tensor/10.0

        # ✅ 检查数值
        print(f"[Debug] sampled_rays_points_tensor shape: {sampled_rays_points_tensor.shape}, dtype: {sampled_rays_points_tensor.dtype}, device: {sampled_rays_points_tensor.device}")
        print(f"[Debug] sampled_rays_points_tensor min: {sampled_rays_points_tensor.min().item():.4f}, max: {sampled_rays_points_tensor.max().item():.4f}")
        print(f"[Debug] sampled_rays_points_tensor sample: {sampled_rays_points_tensor[:]}")  # 打印前5个点


        pred_geo_features, pred_rays_sdfs_tensors, pred_rays_rgbs_tensor = neural_rendering_model(sampled_rays_points_tensor)

        # 使用 Renderer 类根据模型的输出进行最终渲染
        # 打印输出
        # print(f"Shape of sampled_rays_depths_tensor: {sampled_rays_depths_tensor.shape}")  # (ray_num, 3)
        # print(f"Shape of pred_rays_sdfs_tensors: {pred_rays_sdfs_tensors.shape}")  # (ray_num, 3)
        # print(f"Shape of pred_rays_rgbs_tensor: {pred_rays_rgbs_tensor.shape}")  # (ray_num, 1)

        rendered_color, rendered_depth = renderer.render(
            sampled_rays_depths_tensor, 
            pred_rays_sdfs_tensors,
            pred_rays_rgbs_tensor
            )


    # def total_loss(pred_rgb, gt_rgb, pred_d, observe_depth, surface_depths_tensor, pred_sdfs):
        # print(f"Shape of rendered_color: {rendered_color.shape}")  # (ray_num, 3)
        # print(f"Shape of target_rgb: {target_rgb.shape}")  # (ray_num, 3)
        # print(f"Shape of target_depth: {target_depth.shape}")  # (ray_num, 3)

        loss = total_loss(rendered_color, target_rgb, rendered_depth, sampled_rays_depths_tensor, target_depth.unsqueeze(-1), pred_rays_sdfs_tensors)

        # 打印损失
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新模型参数


    # visualize_point_cloud(all_rays_endpoint_3d,rendered_color)

    # visualize_point_cloud(points_tensor=sampled_rays_points_tensor,sdf_tensors=pred_rays_sdfs_tensors,color_tensors=pred_rays_rgbs_tensor)

    bounding_box = np.array([[-3.5,-3.5,0],[3.5,3.5,3]])  # 自定义全局范围
    voxel_size = 0.05


    surface_points = visualize_global_surface(
        query_fn=renderer.query_sdf_color_function, 
        bounding_box=bounding_box, 
        voxel_size=voxel_size, 
        truncation=0.1,
        device='cuda',
        save_path='./global_surface.ply'
    )

