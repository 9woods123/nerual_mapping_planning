import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_model.nerual_render_model import NeuralRenderingModel
import torch
import cv2
import numpy as np
import torch.nn as nn

import torch.optim as optim
from slam_core.ray_casting import RayCasting
from slam_core.renderer import Renderer
from network_model.loss_calculate import total_loss
from slam_core.se3_utils import se3_to_SE3
import random
from utils.utils import save_loss_curve

def select_window(keyframes, window_size):

    if len(keyframes) <= window_size:
        # 如果关键帧数量不足，直接返回全部
        return keyframes
    else:
        # 最新帧必须包含
        latest_kf = keyframes[-1]
        # 历史帧随机采 window_size-1 个
        historical_kfs = random.sample(keyframes[:-1], window_size-1)
        # 拼接
        return historical_kfs + [latest_kf]

class Mapper:
    def __init__(self, model, fx, fy, cx, cy,  width, height, truncation=0.1, lr=1e-3, track_lr=1e-3, iters=100,downsample_ratio=0.001, device="cuda"):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model

        max_window_size=10

        self.delta_se3s = nn.ParameterList([
            nn.Parameter(torch.zeros(6, device=self.device),requires_grad=True) for _ in range(max_window_size)
        ])

        self.optimizer = torch.optim.Adam([
            {"params": self.model.parameters(), "lr": lr},
            {"params": self.delta_se3s, "lr": track_lr},
        ])


        self.renderer = Renderer(self.model, truncation)
        self.ray_casting = RayCasting(np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]]),sample_ratio=downsample_ratio)
        self.iters=iters
        
        self.width=width
        self.height=height


    def update_map(self, keyframes, is_first_frame, index, window_size=10):

        with torch.no_grad():
            for delta in self.delta_se3s:  # 多关键帧
                delta.zero_()

        iteration_number=0

        if is_first_frame :
            iteration_number=5*self.iters
        else:
            iteration_number=self.iters

        used_keyframes = select_window(keyframes, window_size)
        

        ba_losses=[]
        BA_loss=0

        for i in range(iteration_number):
            self.optimizer.zero_grad()

            BA_loss=0

            for j, kf in enumerate(used_keyframes):

                # 增量应用到初始位姿
                pose_mat = se3_to_SE3(self.delta_se3s[j]) @ kf.c2w

                # 射线采样
                rays_3d, rgb_values, depths = self.ray_casting.cast_rays(kf.depth, kf.color, pose_mat,self.height, self.width)
                
                all_points, all_depths, all_endpoints_3d, all_depths_end = self.ray_casting.sample_points_along_ray(
                    ray_origin=pose_mat[:3, 3],
                    rays_direction_list=rays_3d,
                    depths_list=depths
                )

                _, pred_sdfs, pred_colors = self.model(all_points)
                rendered_color, rendered_depth = self.renderer.render(all_depths, pred_sdfs, pred_colors)

                loss = total_loss(rendered_color, rgb_values, rendered_depth, all_depths, all_depths_end.unsqueeze(-1), pred_sdfs)
                
                BA_loss+=loss
            

            BA_loss.backward()
            ba_losses.append(BA_loss.item())

            self.optimizer.step()



        with torch.no_grad():
            # 优化完成后，把 delta_se3s 应用到关键帧的 c2w
            for j, kf in enumerate(used_keyframes):
                kf._c2w = (se3_to_SE3(self.delta_se3s[j]) @ kf.c2w.to(self.device)).detach().clone()


        # 最新帧位姿
        joint_opt_pose_latest = (se3_to_SE3(self.delta_se3s[-1]) @ 
                                used_keyframes[-1].c2w.to(self.device)).detach().clone()
        
        save_loss_curve(ba_losses, index, "./mlp_results/mapping_loss")


        return BA_loss.item(),joint_opt_pose_latest

