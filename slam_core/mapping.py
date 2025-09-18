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
    def __init__(self, model, fx, fy, cx, cy, truncation=0.1, lr=1e-3, iters=100,downsample_ratio=0.001, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.renderer = Renderer(self.model, truncation)
        self.ray_casting = RayCasting(np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]]),sample_ratio=downsample_ratio)
        self.iters=iters

    def update_map(self, color, depth, camera_pose, is_frist_frame=False):

        loss_val = 0.0
        iteration_number=0

        if is_frist_frame :
            iteration_number=5*self.iters
        else:
            iteration_number=self.iters



        # === 多轮迭代优化 ===

        for i in range(iteration_number):

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


            _, pred_sdfs, pred_colors = self.model(sampled_rays_points_tensor)
            rendered_color, rendered_depth = self.renderer.render(sampled_rays_depths_tensor, pred_sdfs, pred_colors)

            loss = total_loss(rendered_color, target_rgb, rendered_depth,
                            sampled_rays_depths_tensor, target_depth.unsqueeze(-1), pred_sdfs)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()  # 保存最后一次 loss
            print(f'Epoch {i}/{iteration_number}, Loss: {loss_val}')

        return loss_val

