import torch
import numpy as np

class RayCasting:

    def __init__(self, intrinsic_matrix,  
                sample_ratio=0.001, 
                M_c=20, 
                M_f=10, 
                d_s=0.05, 
                ignore_edge_W=None,
                ignore_edge_H=None,
                device="cuda"):
        """
        初始化射线投影类
        """
        if isinstance(intrinsic_matrix, np.ndarray):
            intrinsic_matrix = torch.from_numpy(intrinsic_matrix).float()
        self.intrinsic_matrix = intrinsic_matrix.to(device)
        
        self.sample_ratio = sample_ratio
        self.M_c = M_c
        self.M_f = M_f
        self.d_s = d_s
        self.device = device

        # 新增
        self.ignore_edge_W = ignore_edge_W
        self.ignore_edge_H = ignore_edge_H

    def _sample_pixels(self, height, width):
        """
        随机采样像素索引 (忽略边缘区域)
        """
        ignore_W = self.ignore_edge_W or 0
        ignore_H = self.ignore_edge_H or 0

        u = torch.arange(ignore_W, width - ignore_W, device=self.device)
        v = torch.arange(ignore_H, height - ignore_H, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        coords = torch.stack([uu.flatten(), vv.flatten()], dim=1)

        total_pixels = coords.shape[0]
        sample_size = int(total_pixels * self.sample_ratio)

        sampled_indices = torch.randperm(total_pixels, device=self.device)[:sample_size]
        sampled_uv = coords[sampled_indices]

        u_sampled = sampled_uv[:, 0]
        v_sampled = sampled_uv[:, 1]
        return u_sampled, v_sampled
    
    def cast_rays(self, depth_map, color_map, pose, height, width):
        """
        通过深度图和彩色图生成射线 (Torch版)
        :param depth_map: (H, W) torch.Tensor
        :param color_map: (H, W, 3) torch.Tensor
        :param pose: (4, 4) torch.Tensor
        """
        u, v = self._sample_pixels(height, width)

        depth = depth_map[v.long(), u.long()]

        # 过滤掉无效深度
        valid_mask = depth > 0
        u, v, depth = u[valid_mask], v[valid_mask], depth[valid_mask]

        pixel_coords = torch.stack([u.float(), v.float(), torch.ones_like(u, dtype=torch.float32)], dim=0)  # (3, N)
        
        cam_coords = torch.linalg.inv(self.intrinsic_matrix) @ pixel_coords * depth  # (3, N)
        
        world_coords = (pose @ torch.cat([cam_coords, torch.ones(1, cam_coords.shape[1], device=self.device)], dim=0))  # (4, N)



        rgb = color_map[v, u]  # (N, 3)


        return world_coords[:3].T, rgb, depth

    
    def sample_points_along_ray(self, ray_origin, rays_direction_world, depths_list, M_c=None, M_f=None, d_s=None):
        """
        深度引导采样 (Torch版)
        :param ray_origin: (3,) torch.Tensor
        :param rays_direction_list: (N, 3) torch.Tensor
        :param depths_list: (N,) torch.Tensor
        """
        if M_c is None:
            M_c = self.M_c
        if M_f is None:
            M_f = self.M_f
        if d_s is None:
            d_s = self.d_s

        N = rays_direction_world.shape[0]
        # rays_direction_list: (N, 3) 世界坐标下每个像素对应的点
        # ray_origin: (3,) 世界坐标下的相机中心

        # 计算射线方向
        rays_direction = rays_direction_world - ray_origin.unsqueeze(0)  # (N, 3)
        rays_direction = rays_direction / torch.norm(rays_direction, dim=1, keepdim=True)  # 归一化

        all_points = []
        all_depths = []
        endpoints_3d = []
        endpoints_depths = []

        for i in range(N):
            d = depths_list[i]
            if d <= 0:
                continue

            dir_i = rays_direction[i]


            # torch.linspace(start, end, steps) 的参数含义如下：
            # start：起始值（包含在生成的序列中）。
            # end：结束值（包含在生成的序列中）。
            # steps：生成的元素个数（均匀分布在 [start, end] 区间内）

            # 均匀采样 M_c 个点
            t_c = torch.linspace(1.0 / M_c, 1.0, M_c, device=self.device)
            depths_c = t_c * d
            points_c = ray_origin + depths_c[:, None] * dir_i

            # 近表面采样 M_f 个点
            t_f = torch.linspace(0.0, 1.0, M_f, device=self.device)

            depths_f = (d - d_s) + t_f * (2 * d_s)
            points_f = ray_origin + depths_f[:, None] * dir_i

            points = torch.cat([points_c, points_f], dim=0)
            depths = torch.cat([depths_c, depths_f], dim=0)

            all_points.append(points)
            all_depths.append(depths)
            endpoints_3d.append(ray_origin + d * dir_i)
            endpoints_depths.append(d)
        
        all_points = torch.stack(all_points, dim=0)           # (N, M_c+M_f, 3)
        all_depths = torch.stack(all_depths, dim=0)           # (N, M_c+M_f)
        endpoints_3d = torch.stack(endpoints_3d, dim=0)       # (N, 3)
        endpoints_depths = torch.tensor(endpoints_depths, device=self.device)  # (N,)

        # print("all_points:", all_points.shape)
        # print("all_depths:", all_depths.shape)
        # print("endpoints_3d:", endpoints_3d.shape)
        # print("endpoints_depths:", endpoints_depths.shape)

        return all_points, all_depths, endpoints_3d, endpoints_depths
