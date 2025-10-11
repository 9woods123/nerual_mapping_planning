import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_model.nerual_render_model import NeuralRenderingModel
import torch
import cv2
import numpy as np
import torch.nn as nn
import time
import torch.optim as optim
from slam_core.ray_casting import RayCasting
from slam_core.renderer import Renderer
from network_model.loss_calculate import total_loss
from slam_core.se3_utils import se3_to_SE3
import random
from utils.utils import save_loss_curve

import random

def select_window(keyframes, window_size):
    """
    选择关键帧窗口
    - 总数 <= window_size → 返回全部
    - 总数 > window_size → 保证包含最新两帧，其余随机选
    """
    n = len(keyframes)
    if n <= window_size:
        return keyframes
    else:
        # 最新两帧必须包含
        latest_two = keyframes[-2:]  # [-2] 和 [-1]
        # 剩余历史帧可选
        remaining = keyframes[:-2]
        # 还需要选择的数量
        num_select = window_size - 2
        historical_kfs = random.sample(remaining, num_select) if num_select > 0 else []
        # 拼接
        return historical_kfs + latest_two


class Mapper:

    def __init__(self, model, 
                 params,
                 device="cuda"):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model
        self.params=params

        self.fx = self.params.camera.fx
        self.fy = self.params.camera.fy
        self.cx = self.params.camera.cx
        self.cy = self.params.camera.cy

        self.height=self.params.camera.height
        self.width=self.params.camera.width
        
        self.truncation=self.params.mapping.truncation

        self.mapping_lr=self.params.mapping.lr
        self.tracking_lr=self.params.tracking.lr

        self.iters=self.params.mapping.iters
        self.sample_ratio=self.params.mapping.sample_ratio
        

        self.renderer = Renderer(self.model, self.truncation)

        self.ray_casting = RayCasting(
            np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0, 0, 1]]),
            sample_ratio=self.sample_ratio,
            ignore_edge_W=0,
            ignore_edge_H=0
        )


        max_window_size=10

        self.delta_rot = nn.ParameterList([
            nn.Parameter(torch.zeros(3, device=self.device), requires_grad=True) for _ in range(max_window_size)
        ])
        self.delta_trans = nn.ParameterList([
            nn.Parameter(torch.zeros(3, device=self.device), requires_grad=True) for _ in range(max_window_size)
        ])

        self.optimizer = torch.optim.Adam([
            {"params": self.model.parameters(), "lr": self.mapping_lr},
            {"params": self.delta_trans, "lr": self.tracking_lr},
            {"params": self.delta_rot, "lr": self.tracking_lr},
        ])


    def update_map(self, keyframes, is_first_frame, index, window_size=10):

        with torch.no_grad():
            for delta in self.delta_rot:  # 多关键帧
                delta.zero_()
            for delta in self.delta_trans:  # 多关键帧
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
            iter_start = time.time()
            self.optimizer.zero_grad()

            BA_loss = 0

            for j, kf in enumerate(used_keyframes):
                kf_start = time.time()

                # 增量应用到初始位姿
                pose_se3= torch.cat([self.delta_rot[j],self.delta_trans[j]])
                pose_mat = se3_to_SE3(pose_se3) @ kf.c2w

                # --- timing breakdown ---
                t0 = time.time()
                rays_3d, rgb_values, depths = self.ray_casting.cast_rays(
                    kf.depth, kf.color, pose_mat, self.height, self.width
                )
                t1 = time.time()
                all_points, all_depths, all_endpoints_3d, all_depths_end = self.ray_casting.sample_points_along_ray(
                    ray_origin=pose_mat[:3, 3],
                    rays_direction_world=rays_3d,
                    depths_list=depths
                )
                t2 = time.time()

                pred_sdfs, pred_colors = self.model(all_points)
                
                t3 = time.time()
                
                rendered_color, rendered_depth = self.renderer.render(
                    all_depths, pred_sdfs, pred_colors
                )
                
                t4 = time.time()
                
                total_loss_value, loss_color, loss_depth, loss_surface, loss_free = total_loss(
                    rendered_color, rgb_values, rendered_depth,
                    all_depths, all_depths_end.unsqueeze(-1), pred_sdfs
                )
                
                t5 = time.time()

                BA_loss += total_loss_value

                # print per-frame time
                # print(f"[Keyframe {j}] cast={t1-t0:.3f}s, sample={t2-t1:.3f}s, "
                #     f"model={t3-t2:.3f}s, render={t4-t3:.3f}s, loss={t5-t4:.3f}s, total={t5-kf_start:.3f}s")

            BA_loss = BA_loss / len(used_keyframes)
            BA_loss.backward()
            ba_losses.append(BA_loss.item())
            self.optimizer.step()

            iter_end = time.time()

            # print(f"[Iter {i}] BA_loss={BA_loss.item():.6f}, total_time={iter_end-iter_start:.3f}s")


        with torch.no_grad():
            # 优化完成后，把 delta_se3s 应用到关键帧的 c2w
            for j, kf in enumerate(used_keyframes):
                pose_se3= torch.cat([self.delta_rot[j],self.delta_trans[j]])
                kf._c2w = (se3_to_SE3(pose_se3) @ kf.c2w.to(self.device)).detach().clone()


        # 最新帧位姿
        pose_se3_latest= torch.cat([self.delta_rot[-1],self.delta_trans[-1]])
        joint_opt_pose_latest = (se3_to_SE3(pose_se3_latest) @ 
                                used_keyframes[-1].c2w.to(self.device)).detach().clone()
        
        save_loss_curve(ba_losses, index, "./mlp_results/mapping_loss")



        return BA_loss.item(),joint_opt_pose_latest

