
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from visualization.frustum_culling import FrustumCulling

import numpy as np
import open3d as o3d
import torch



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
        self.frustum_culler= FrustumCulling(fx, fy, cx, cy, width, height)


    def generate_grid_points(self, device='cpu'):
        """
        根据边界和分辨率生成网格点
        Returns:
            points (torch.Tensor): shape (N,3)，场景中均匀分布的点
        """
        assert self.bound is not None, "Scene bound is not set!"
        min_x, min_y, min_z, max_x, max_y, max_z = self.bound

        xs = torch.arange(min_x, max_x, self.resolution, device=device)
        ys = torch.arange(min_y, max_y, self.resolution, device=device)
        zs = torch.arange(min_z, max_z, self.resolution, device=device)

        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)
        points_tensor = grid.reshape(-1, 3)
        
        return points_tensor


    def generate_surface_pointcloud(self, query_fn, keyframe_dict,
                                    batch_size=65536,
                                    save_path=None, device='cuda'):
        """
        生成表面点云（GPU 加速多帧 frustum culling）

        Args:
            query_fn (function): 输入 pts_tensor, 返回 (sdf_values, color_values)
            keyframe_dict (list): 关键帧信息
            batch_size (int): 批量查询大小
            forecast_margin (float): 预测区域扩展
            save_path (str): 保存路径
            device (str): torch 设备

        Returns:
            surface_points (np.ndarray): 表面点位置
        """
        # 1. 生成网格点
        grid_points = self.generate_grid_points(device=device)  # (N,3)

        # 2. 从 keyframe_dict 提取 c2w
        c2w_all = torch.stack([torch.tensor(kf.c2w, dtype=torch.float32, device=device) for kf in keyframe_dict], dim=0)  # (F,4,4)

        # 3. GPU 批量 frustum culling
        seen_mask, forecast_mask, _ = self.frustum_culler.split_points_frustum_multi(
            grid_points, c2w_all)
        
        grid_seen = grid_points[seen_mask]

        # 4. 查询 SDF 和颜色
        sdf_values, color_values = [], []
        pts_tensor = grid_seen
        for start in range(0, pts_tensor.shape[0], batch_size):
            end = start + batch_size
            sdf_batch, color_batch = query_fn(pts_tensor[start:end])
            sdf_values.append(sdf_batch.cpu())
            color_values.append(color_batch.cpu())

        sdf_values = torch.cat(sdf_values, dim=0).numpy()
        color_values = torch.cat(color_values, dim=0).numpy()

        # 5. 筛选表面点
        mask = np.abs(sdf_values) < 0.1
        surface_points = grid_seen.cpu().numpy()[mask.squeeze()]
        surface_colors = color_values[mask.squeeze()]

        # 6. 可视化
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_points)
        pcd.colors = o3d.utility.Vector3dVector(surface_colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries([pcd, coord_frame])

        # 7. 保存
        if save_path is not None:
            import os
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"[Saved] Surface point cloud saved to {save_path}")

        return surface_points
