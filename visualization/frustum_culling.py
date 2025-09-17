


import sys
import os


import numpy as np
import trimesh
import mcubes
import torch


class FrustumCulling:
    def __init__(self, fx, fy, cx, cy, width, height, device='cuda'):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.W = width
        self.H = height
        self.device = device


    def split_points_frustum_multi(self, points, c2w_all, near=0.1, far=10.0, forecast_margin=0.25):
        
        """
        GPU frustum culling for multiple frames
        Args:
            points: (N,3) torch.Tensor [world]
            c2w_all: (B,4,4) torch.Tensor [camera-to-world] for all frames
        Returns:
            seen_mask: (N,) bool
            forecast_mask: (N,) bool
            unseen_mask: (N,) bool
        """

        device = points.device
        B = c2w_all.shape[0]
        N = points.shape[0]

        # 1. world -> camera (batch)
        w2c_all = torch.inverse(c2w_all).to(device)   # (B,4,4)
        pts_h = torch.cat([points, torch.ones(N, 1, device=device)], dim=-1)  # (N,4)
        pts_h = pts_h.unsqueeze(0).expand(B, N, 4)  # (B,N,4)

        pts_cam = torch.bmm(pts_h, w2c_all.transpose(1, 2))[:, :, :3]  # (B,N,3)

        # 2. 深度判断
        z = pts_cam[:, :, 2]  # (B,N)
        depth_valid = (z > near) & (z < far)

        # 3. 投影到像素
        x = self.fx * (pts_cam[:, :, 0] / z) + self.cx
        y = self.fy * (pts_cam[:, :, 1] / z) + self.cy
        inside_img = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)

        # 4. seen
        seen = depth_valid & inside_img  # (B,N)

        # 5. forecast: 扩展 margin
        margin = forecast_margin * torch.clamp(z, min=1e-6)
        inside_forecast = (x >= -margin) & (x < self.W + margin) & \
                          (y >= -margin) & (y < self.H + margin)
        forecast = depth_valid & inside_forecast & (~seen)

        # 6. 合并 B 个相机结果
        seen_mask = torch.any(seen, dim=0)         # (N,)
        forecast_mask = torch.any(forecast, dim=0) # (N,)
        unseen_mask = ~(seen_mask | forecast_mask)

        return seen_mask, forecast_mask, unseen_mask




# class FrustumCulling:
#     def __init__(self, fx, fy, cx, cy, width, height, pose=np.eye(4)):
#         """
#         初始化相机参数和视锥体信息

#         Args:
#             fx, fy (float): 相机焦距 (像素单位)
#             cx, cy (float): 相机主点坐标 (像素单位)
#             width, height (int): 图像分辨率
#             pose (4x4 ndarray): 相机位姿矩阵，将世界坐标转换到相机坐标系 (world -> camera)
#         """
#         self.fx = fx
#         self.fy = fy
#         self.cx = cx
#         self.cy = cy
#         self.W = width
#         self.H = height
#         self.pose = pose  # world -> camera

#     def set_camera_pose(self,pose):
#         self.pose=pose


#     def world_to_camera(self, points_world):
#         """
#         将世界坐标系点转换到相机坐标系

#         Args:
#             points_world (N x 3 ndarray): 世界坐标系下的点

#         Returns:
#             cam_points (N x 3 ndarray): 相机坐标系下的点
#         """
#         N = points_world.shape[0]
#         homo = np.hstack([points_world, np.ones((N, 1))])  # 转为齐次坐标
#         cam_points = (self.pose @ homo.T).T[:, :3]         # 乘位姿矩阵 -> 相机坐标
#         return cam_points

#     def in_frustum_projection(self, points_world, near=0.1, far=10.0):
#         """
#         方法 1: 通过投影判断点是否在视锥体内 (沿 Z 正方向为相机前方)
        
#         Args:
#             points_world (N x 3 ndarray): 世界坐标系下的点
#             near (float): 近平面距离
#             far (float): 远平面距离

#         Returns:
#             mask (N bool array): True 表示点在相机视锥内
#         """
#         cam_points = self.world_to_camera(points_world)  # 转到相机坐标系
#         x_cam, y_cam, z_cam = cam_points.T  

#         # 1. Z 必须在近/远平面之间
#         mask_z = (z_cam > near) & (z_cam < far)

#         # 2. 投影到像素坐标 (像素坐标系 u,v)
#         u = self.fx * x_cam / z_cam + self.cx
#         v = self.fy * y_cam / z_cam + self.cy

#         # 3. 投影必须在图像范围内
#         mask_uv = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)

#         mask = mask_z & mask_uv
#         assert mask.shape[0] == points_world.shape[0]
#         return mask

#     def build_frustum_mesh(self, near=0.1, far=10.0):
#         """
#         方法 2: 构造视锥体几何体 (trimesh.ConvexHull)

#         Returns:
#             hull (trimesh.Trimesh): 视锥体凸多面体，用于快速集合判断
#         """
#         def get_corners(z):
#             # 根据深度 z 计算近平面或远平面四个角在相机坐标系下的坐标
#             xs = [(0 - self.cx) * z / self.fx, (self.W - self.cx) * z / self.fx]
#             ys = [(0 - self.cy) * z / self.fy, (self.H - self.cy) * z / self.fy]
#             return [[x, y, -z] for x in xs for y in ys]

#         corners_near = get_corners(near)
#         corners_far = get_corners(far)
#         corners = np.array(corners_near + corners_far)  # 共 8 个角点

#         # 转换到世界坐标系
#         homo = np.hstack([corners, np.ones((8, 1))])
#         corners_world = (np.linalg.inv(self.pose) @ homo.T).T[:, :3]

#         hull = trimesh.convex.convex_hull(corners_world)  # 构建凸包
#         return hull

#     def in_frustum_hull(self, points_world, near=0.1, far=10.0):
#         """
#         方法 2: 使用凸多面体判断点是否在视锥体内
        
#         Args:
#             points_world (N x 3 ndarray): 世界坐标系下的点
#             near, far (float): 近平面和远平面

#         Returns:
#             mask (N bool array): True 表示点在凸包内
#         """
#         hull = self.build_frustum_mesh(near, far)
#         return hull.contains(points_world)

#     def split_points_frustum(self, points_world, near=0.1, far=10.0, forecast_margin=0.5):
#         """
#         将点分为 seen / forecast / unseen 三类

#         Args:
#             points_world (N x 3 ndarray): 世界坐标系下的点
#             near, far (float): 近平面和远平面
#             forecast_margin (float): 预测区域向外扩展距离（像素或深度单位）

#         Returns:
#             seen_mask (N bool array): 在视锥内的点
#             forecast_mask (N bool array): 视锥外但在扩展边界内的点
#             unseen_mask (N bool array): 不在视锥也不在预测区域内的点
#         """
#         cam_points = self.world_to_camera(points_world)
#         x_cam, y_cam, z_cam = cam_points.T

#         # seen mask: 在近平面/远平面之间，并投影在图像内
#         mask_z = (z_cam > near) & (z_cam < far)
#         u = self.fx * x_cam / z_cam + self.cx
#         v = self.fy * y_cam / z_cam + self.cy
#         mask_uv = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
#         seen_mask = mask_z & mask_uv

#         # forecast mask: 在视锥外，但在扩展边界内
#         u_margin = (u >= -forecast_margin) & (u < self.W + forecast_margin)
#         v_margin = (v >= -forecast_margin) & (v < self.H + forecast_margin)
#         z_margin = (z_cam > near - forecast_margin) & (z_cam < far + forecast_margin)
#         forecast_mask = (~seen_mask) & u_margin & v_margin & z_margin

#         # unseen mask: 剩余的点
#         unseen_mask = ~(seen_mask | forecast_mask)

#         return seen_mask, forecast_mask, unseen_mask