import torch
import numpy as np



class RayCasting:
    def __init__(self, intrinsic_matrix, sample_ratio=0.001, M_c=10, M_f=4, d_s=0.1):
        """
        初始化射线投影类
        
        :param intrinsic_matrix: 相机内参矩阵 (3x3) 
                                                      [ f_x   0     c_x 
                                                        0     f_y   c_y 
                                                        0     0      1   ]
        :param sample_ratio: 采样比例，决定从总像素中采样的比例，默认 0.1 即采样 10%
        :param num_samples: 每条射线采样的点数
        :param M_c: 每条射线均匀采样的点数
        :param M_f: 每条射线的近表面采样点数
        :param d_s: 深度偏移量
        """
        self.intrinsic_matrix = intrinsic_matrix
        self.sample_ratio = sample_ratio  # 采样比例
        self.M_c = M_c  # 每条射线均匀采样的点数
        self.M_f = M_f  # 每条射线的近表面采样点数
        self.d_s = d_s  # 深度偏移量


    def _sample_pixels(self, height, width):
        """
        随机采样像素索引
        
        :param height: 图像高度
        :param width: 图像宽度
        :return: 采样的像素索引
        """
        total_pixels = height * width
        sample_size = int(total_pixels * self.sample_ratio)
        sampled_indices = np.random.choice(total_pixels, sample_size, replace=False)
        return sampled_indices


    def cast_rays(self, depth_map, color_map, pose, height, width):
        """
        通过深度图和彩色图生成射线
        
        :param depth_map: 深度图 (height, width)
        :param color_map: 彩色图 (height, width, 3)
        :param pose: 相机位姿，4x4 变换矩阵
        :param height: 图像高度
        :param width: 图像宽度
        
        :return: 射线上的 3D 点，RGB 颜色和深度值
        """
        rays_3d = []
        rgb_values = []
        depths = []
        
        sampled_indices = self._sample_pixels(height, width)
        
        for idx in sampled_indices:
            v, u = divmod(idx, width)
            depth = depth_map[v, u]
            if depth == 0:
                continue
            
            pixel_coords = np.array([u, v, 1.0])
            camera_coords = np.linalg.inv(self.intrinsic_matrix) @ pixel_coords * depth
            world_coords = pose @ np.append(camera_coords, 1)
            rgb = color_map[v, u]
            
            rays_3d.append(world_coords[:3])
            rgb_values.append(rgb)
            depths.append(depth)

        return rays_3d, rgb_values, depths



    def sample_points_along_ray(self, ray_origin, rays_direction_list, depths_list , M_c=None, M_f=None, d_s=None):
        """
        深度引导采样：在射线的远近边界之间均匀采样 M_c 个点，
        对于有有效深度的射线，额外在 [d - d_s, d + d_s] 范围内均匀采样 M_f 个点
        
        :param ray_origin: 射线起点
        :param ray_direction: 射线方向
        :param depths: 深度值
        :param M_c: 每条射线均匀采样的点数
        :param M_f: 每条射线的近表面采样点数
        :param d_s: 深度偏移量
        :return: 采样的 3D 点、颜色和深度值
        """

        if len(rays_direction_list) != len(depths_list):
            raise ValueError("rays_direction_list 和 depths_list 的长度不一致！")

        if M_c is None:
            M_c = self.M_c
        if M_f is None:
            M_f = self.M_f
        if d_s is None:
            d_s = self.d_s
        
        sampled_3d_points = []
        sampled_depths = []
        
        for ray_id in range(len(rays_direction_list)):
            
            ray_direction=rays_direction_list[ray_id]
            ray_direction = ray_direction / np.linalg.norm(ray_direction)  ## 二次确认归一化

            depth= depths_list[ray_id]

            # 采样远近边界上的 M_c 个点
            for i in range(M_c):
                t = (i + 1) / M_c

                sampled_depth = t * depth
                sample_point = ray_origin + sampled_depth * ray_direction

                sampled_3d_points.append(sample_point)
                sampled_depths.append(sampled_depth)

            # 对于有有效深度的射线，采样 M_f 个近表面点
            for j in range(M_f):
                # 近表面点的深度采样范围是 [d - d_s, d + d_s]
                t = (j + 1) / M_f

                sampled_depth =  (depth - d_s + t * 2 * d_s) 
                sample_point = ray_origin + sampled_depth * ray_direction  # 使用深度进行偏移
                
                sampled_3d_points.append(sample_point)
                sampled_depths.append(sampled_depth)


        return sampled_3d_points, sampled_depths