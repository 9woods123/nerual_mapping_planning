


import sys
import os


import numpy as np
import trimesh
import torch


class FrustumCulling:
    def __init__(self, fx, fy, cx, cy, width, height, device='cuda',
                 near=0.1, far=2.5, forecast_margin=0.15):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.W = width
        self.H = height
        self.device = device

        # frustum 参数
        self.near = near
        self.far = far
        self.forecast_margin = forecast_margin
            
    def split_points_frustum_multi(self, points, c2w_all, chunk_size=100000):
        """
        Args:
            points: (N,3) torch.Tensor [world]
            c2w_all: (B,4,4) torch.Tensor [camera-to-world] for all frames
            chunk_size: 每次处理多少点，避免OOM
        """
        device = points.device
        B = c2w_all.shape[0]
        N = points.shape[0]

        w2c_all = torch.inverse(c2w_all).to(device)   # (B,4,4)

        seen_mask_total = torch.zeros(N, dtype=torch.bool, device=device)
        forecast_mask_total = torch.zeros(N, dtype=torch.bool, device=device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            pts_chunk = points[start:end]  # (chunk,3)

            pts_h = torch.cat([pts_chunk, torch.ones(end - start, 1, device=device)], dim=-1)  # (chunk,4)
            pts_h = pts_h.unsqueeze(0).expand(B, -1, 4)  # (B,chunk,4)

            pts_cam = torch.bmm(pts_h, w2c_all.transpose(1, 2))[:, :, :3]  # (B,chunk,3)

            # ===== 原逻辑 =====
            z = pts_cam[:, :, 2]
            depth_valid = (z > self.near) & (z < self.far)

            x = self.fx * (pts_cam[:, :, 0] / z) + self.cx
            y = self.fy * (pts_cam[:, :, 1] / z) + self.cy
            inside_img = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)

            seen = depth_valid & inside_img

            margin = self.forecast_margin * torch.clamp(z, min=1e-6)
            inside_forecast = (x >= -margin) & (x < self.W + margin) & \
                            (y >= -margin) & (y < self.H + margin)
            forecast = depth_valid & inside_forecast & (~seen)

            # 合并
            seen_mask_total[start:end] = torch.any(seen, dim=0)
            forecast_mask_total[start:end] = torch.any(forecast, dim=0)

        unseen_mask_total = ~(seen_mask_total | forecast_mask_total)
        return seen_mask_total, forecast_mask_total, unseen_mask_total
