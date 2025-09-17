
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from slam_core.keyframe import Keyframe

from frustum_culling import FrustumCulling

import numpy as np
import trimesh
import open3d as o3d
import torch
import mcubes
import trimesh



class Mesher:
    def __init__(self, min_x, min_y, min_z, max_x, max_y, max_z, fx, fy, cx, cy, width, height, resolution=0.01):
        """
        Mesher 类，用于根据场景表示生成网格点并分类
        
        Attributes:
            resolution (float): 网格点的间距
            bound (np.ndarray): 场景边界 [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        self.resolution = resolution
        self.bound = [min_x, min_y, min_z, max_x, max_y, max_z]  
        self.frustum_culler= FrustumCulling(fx, fy, cx, cy, width, height, pose=np.eye(4))


    def generate_grid_points(self):
        """
        根据边界和分辨率生成网格点
        Returns:
            points (np.ndarray): shape (N,3)，场景中均匀分布的点
        """
        assert self.bound is not None, "Scene bound is not set!"
        min_x, min_y, min_z, max_x, max_y, max_z = self.bound

        xs = np.arange(min_x, max_x, self.resolution)
        ys = np.arange(min_y, max_y, self.resolution)
        zs = np.arange(min_z, max_z, self.resolution)

        grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
        points = grid.reshape(-1, 3)
        
        return points



    def point_mask(self, points, keyframe_dict, forecast_margin=0.25):
        """
        将点分类为 seen / forecast / unseen
        Args:
            points (np.ndarray): shape (N,3)，场景网格点
            keyframe_dict (list): 关键帧信息列表
            forecast_margin (float): 预测区域扩展距离
        Returns:
            seen_mask (np.ndarray): 布尔数组，True 表示点被至少一个相机看到
            forecast_mask (np.ndarray): 预测区域内的点
            unseen_mask (np.ndarray): 其他点
        """

        N = points.shape[0]
        seen_mask = np.zeros(N, dtype=bool)
        forecast_mask = np.zeros(N, dtype=bool)

        for i, keyframe in enumerate(keyframe_dict):
            self.frustum_culler.set_camera_pose(keyframe.c2w)
            s, f, _ = self.frustum_culler.split_points_frustum(points, 
                                                          near=0.1, 
                                                          far=10.0, 
                                                          forecast_margin=forecast_margin)
            seen_mask |= s
            forecast_mask |= f

        unseen_mask = ~(seen_mask | forecast_mask)
        return seen_mask, forecast_mask, unseen_mask



    def generate_surface_pointcloud(self, query_fn, keyframe_dict,
                                        batch_size=65536, forecast_margin=0.5, save_path=None,
                                        device='cuda'):
            """
            生成表面点云：网格点 -> 分类 -> 查询 SDF 和颜色 -> 筛选表面点 -> 可视化和保存

            Args:
                query_fn (function): 输入 pts_tensor, 返回 (sdf_values, color_values)
                frustum_cullers (list[FrustumCulling]): 每个关键帧对应的 FrustumCulling 对象
                keyframe_dict (list): 关键帧信息
                batch_size (int): 批量查询大小
                forecast_margin (float): 预测区域扩展
                save_path (str): 保存路径
                device (str): torch 设备

            Returns:
                surface_points (np.ndarray): 表面点位置
            """
            # 1. 生成网格点
            grid = self.generate_grid_points()

            # 2. 分类
            seen_mask, _, _ = self.point_mask(grid,keyframe_dict, forecast_margin)
            grid_seen = grid[seen_mask]

            # 3. 查询 SDF 和颜色
            sdf_values, color_values = [], []
            pts_tensor = torch.from_numpy(grid_seen).float().to(device)
            for start in range(0, pts_tensor.shape[0], batch_size):
                end = start + batch_size
                sdf_batch, color_batch = query_fn(pts_tensor[start:end])
                sdf_values.append(sdf_batch.cpu())
                color_values.append(color_batch.cpu())

            sdf_values = torch.cat(sdf_values, dim=0).numpy()
            color_values = torch.cat(color_values, dim=0).numpy()

            # 4. 筛选表面点
            mask = np.abs(sdf_values) < 0.1
            surface_points = grid_seen[mask.squeeze()]
            surface_colors = color_values[mask.squeeze()]

            # 5. 可视化
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(surface_points)
            pcd.colors = o3d.utility.Vector3dVector(surface_colors)

            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0]
            )
            o3d.visualization.draw_geometries([pcd, coord_frame])

            # 6. 保存
            if save_path is not None:
                os.makedirs(os.path.split(save_path)[0], exist_ok=True)
                o3d.io.write_point_cloud(save_path, pcd)
                print(f"[Saved] Surface point cloud saved to {save_path}")

            return surface_points