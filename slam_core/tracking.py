import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from slam_core.ray_casting import RayCasting
from slam_core.renderer import Renderer
from network_model.loss_calculate import total_loss


class Tracker:
    def __init__(self, model, fx, fy, cx, cy, truncation=0.1, lr=1e-2, iters=20,downsample_ratio=0.001, device="cuda"):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model
        self.renderer = Renderer(self.model, truncation)
        self.ray_casting = RayCasting(np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]]),sample_ratio=downsample_ratio)
        self.lr = lr
        self.iters = iters

        # 保存前两帧的pose in SE3
        self.last_pose = None
        self.prev_pose = None
        
    def predict_pose(self):
        """用恒速模型预测初始位姿"""
        if self.last_pose is None:
            return np.eye(4, dtype=np.float32)
        if self.prev_pose is None:
            return self.last_pose
        # 相对运动
        relative_motion = np.linalg.inv(self.prev_pose) @ self.last_pose
        # 预测下一帧
        return self.last_pose @ relative_motion


    def track(self, color, depth):
        H, W = depth.shape

        # === 初始化位姿 ===
        pred_pose = self.predict_pose()

        pose_se3 = torch.zeros(6, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([pose_se3], lr=self.lr)

        target_rgb = torch.tensor(color.reshape(-1, 3), dtype=torch.float32).to(self.device)

        for _ in range(self.iters):
            optimizer.zero_grad()

            pose_mat = pred_pose @ self.se3_to_SE3(pose_se3)

            rays_3d, rgb_values, depths = self.ray_casting.cast_rays(
                depth, color, pose_mat.detach().cpu().numpy(), H, W
            )

            all_pts, all_depths, _, all_end_depths = self.ray_casting.sample_points_along_ray(
                ray_origin=pose_mat[:3, 3].detach().cpu().numpy(),
                rays_direction_list=rays_3d,
                depths_list=depths
            )

            sampled_points = torch.tensor(np.stack(all_pts), dtype=torch.float32).to(self.device)
            sampled_depths = torch.tensor(np.stack(all_depths), dtype=torch.float32).to(self.device)
            target_depths = torch.tensor(np.stack(all_end_depths), dtype=torch.float32).to(self.device)

            _, pred_sdfs, pred_colors = self.model(sampled_points)
            rendered_color, rendered_depth = self.renderer.render(sampled_depths, pred_sdfs, pred_colors)

            loss = total_loss(rendered_color, target_rgb, rendered_depth,
                              sampled_depths, target_depths.unsqueeze(-1), pred_sdfs)

            loss.backward()
            optimizer.step()

        final_pose = (pred_pose @ self.se3_to_SE3(pose_se3)).detach().cpu().numpy()

        # 更新last/prev pose
        self.prev_pose = self.last_pose
        self.last_pose = final_pose

        return final_pose, loss.item()

    def se3_to_SE3(self, xi):
        """
        将 se(3) 6维向量转为 4x4 矩阵（旧版）
        xi: (6,) torch [ωx, ωy, ωz, tx, ty, tz]
        """

        omega = xi[:3]
        theta = torch.norm(omega) + 1e-8
        I = torch.eye(3, device=xi.device)

        # Rodrigues 公式
        omega_hat = torch.tensor([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ], device=xi.device)

        R = I + (torch.sin(theta)/theta) * omega_hat + ((1-torch.cos(theta))/(theta**2)) * (omega_hat @ omega_hat)

        # 平移直接塞进去
        t = xi[3:].unsqueeze(-1)

        SE3 = torch.eye(4, device=xi.device)
        SE3[:3,:3] = R
        SE3[:3, 3:4] = t

        return SE3
