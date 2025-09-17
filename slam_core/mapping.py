import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_model.nerual_render_model import NeuralRenderingModel
import torch
import cv2
import numpy as np

import torch.optim as optim
from slam_core.ray_casting import RayCasting
from slam_core.renderer import Renderer
from network_model.loss_calculate import total_loss

class Mapper:
    def __init__(self, fx, fy, cx, cy, truncation=0.1, model=None,  device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.renderer = Renderer(self.model, truncation)
        self.ray_casting = RayCasting(np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]]))

    def update_map(self, color, depth, camera_pose):
        # === 生成射线数据 ===
        rays_3d, rgb_values, depths = self.ray_casting.cast_rays(depth, color, camera_pose, depth.shape[0], depth.shape[1])
        target_rgb = torch.tensor(rgb_values, dtype=torch.float32).to(self.device)

        all_rays_points, all_rays_depths, _, all_rays_endpoint_depths = self.ray_casting.sample_points_along_ray(
            ray_origin = camera_pose[:3, 3],  
            rays_direction_list=rays_3d,
            depths_list=depths
        )

        sampled_rays_points_tensor = torch.tensor(np.stack(all_rays_points), dtype=torch.float32).to(self.device)
        sampled_rays_depths_tensor = torch.tensor(np.stack(all_rays_depths), dtype=torch.float32).to(self.device)
        target_depth = torch.tensor(np.stack(all_rays_endpoint_depths), dtype=torch.float32).to(self.device)

        # === 前向预测 ===
        _, pred_sdfs, pred_colors = self.model(sampled_rays_points_tensor)
        rendered_color, rendered_depth = self.renderer.render(sampled_rays_depths_tensor, pred_sdfs, pred_colors)

        # === Loss & 反向传播 ===
        loss = total_loss(rendered_color, target_rgb, rendered_depth, sampled_rays_depths_tensor, target_depth.unsqueeze(-1), pred_sdfs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

