import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from slam_core.ray_casting import RayCasting
from slam_core.renderer import Renderer
from network_model.loss_calculate import total_loss


import torch
import numpy as np
from slam_core.ray_casting import RayCasting
from slam_core.renderer import Renderer
from slam_core.se3_utils import se3_to_SE3,orthogonalize_rotation,SE3_to_se3

from network_model.loss_calculate import total_loss
from utils.utils import *

class Tracker:
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

        self.lr=self.params.tracking.lr
        self.iters=self.params.tracking.iters
        self.sample_ratio=self.params.tracking.sample_ratio
        self.ignore_edge_H=self.params.tracking.ignore_edge_H
        self.ignore_edge_W=self.params.tracking.ignore_edge_W    
           
        self.renderer = Renderer(self.model, self.truncation)

        self.ray_casting = RayCasting(
            np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0, 0, 1]]),
            sample_ratio=self.sample_ratio,
            height=self.height,
            width=self.width,
            ignore_edge_W=self.ignore_edge_W,
            ignore_edge_H=self.ignore_edge_H
        )


        self.delta_rot = torch.zeros(3, device=self.device, requires_grad=True)
        self.delta_trans = torch.zeros(3, device=self.device, requires_grad=True)

        self.optimizer = torch.optim.Adam([
            {"params": self.delta_trans, "lr": self.lr},
            {"params": self.delta_rot, "lr": 0.1 *self.lr},
        ])

        # 保存前两帧 pose（torch Tensor）
        self.last_pose: torch.Tensor = None
        self.prev_pose: torch.Tensor = None

    def predict_pose(self):
        """用恒速模型预测初始位姿 (torch版)"""
        if self.last_pose is None:
            return torch.eye(4, dtype=torch.float32, device=self.device)
        if self.prev_pose is None:
            self.prev_pose=self.last_pose
            return self.last_pose

        
        return self.last_pose 

        # relative_motion = torch.linalg.inv(self.prev_pose) @ self.last_pose
        # return self.last_pose @ relative_motion


    def record_pose(self, last_pose, prev_pose):
        self.prev_pose = prev_pose
        self.last_pose = last_pose

    def update_last_pose(self,last_pose):
        ##TODO this has anologic problem
        self.prev_pose = self.last_pose
        self.last_pose = last_pose


    def track(self, color, depth, is_first_frame, index,gt_pose):


        # === 初始化位姿 ===
        pred_pose = self.predict_pose()  # torch [4,4]

        if is_first_frame:
            return 0, pred_pose


        with torch.no_grad():
            self.delta_trans.zero_()  # 将 tensor 所有元素置0
            self.delta_rot.zero_()  # 将 tensor 所有元素置0
            self.optimizer.state.clear()   # 清空动量、方差等历史   


        best_loss = float('inf')
        best_delta_rot = None
        best_delta_trans = None

        losses = []
        
        u_sampled,v_sampled = self.ray_casting.sample_pixels()



        curr_se3 = SE3_to_se3(pred_pose)  # [6]
        # 拆分旋转和平移部分，并设为可优化参数
        self.curr_rot = torch.nn.Parameter(curr_se3[:3].clone().detach())
        self.curr_trans = torch.nn.Parameter(curr_se3[3:].clone().detach())

        # 定义优化器
        self.optimizer = torch.optim.Adam([
            {"params": [self.curr_rot], "lr": self.lr * 0.1},
            {"params": [self.curr_trans], "lr": self.lr},
        ])




        for _ in range(self.iters):
            self.optimizer.zero_grad()

            # pose_se3=torch.cat([self.delta_rot,self.delta_trans])
            # pose_mat =  se3_to_SE3(pose_se3) @ pred_pose   # torch [4,4]
            # pose_mat[:3, :3] = orthogonalize_rotation(pose_mat[:3, :3])

            pose_se3=torch.cat([self.curr_rot,self.curr_trans])
            pose_mat =  se3_to_SE3(pose_se3)



            sampled_rgb, ray_points_3d, ray_points_depths, surface_points_3d, surface_points_depths = self.ray_casting.get_rays_points_from_pixels(
                u_sampled, v_sampled, depth, color, pose_mat)

            pred_sdfs, pred_colors = self.model(ray_points_3d)
            rendered_color, rendered_depth = self.renderer.render(ray_points_depths, pred_sdfs, pred_colors)

            total_loss_value,loss_color,loss_depth,loss_surface,loss_free = total_loss(rendered_color, sampled_rgb, rendered_depth,
                             ray_points_depths, surface_points_depths.unsqueeze(-1), pred_sdfs)

            with torch.no_grad():
                # 记录最优增量
                if total_loss_value.item() < best_loss:
                    best_loss = total_loss_value.item()
                    best_delta_rot = self.delta_rot.clone().detach()
                    best_delta_trans = self.delta_trans.clone().detach()


            total_loss_value.backward()
            self.optimizer.step()
            losses.append(total_loss_value.item())


        # final_pose = se3_to_SE3(torch.cat([best_delta_rot, best_delta_trans])) @ pred_pose

        final_pose = se3_to_SE3(torch.cat([self.curr_rot,self.curr_trans])) 

        # visualize_sparse_render(
        #     u_sampled,
        #     v_sampled,
        #     rendered_color,
        #     rendered_depth,
        #     color,
        #     depth,
        #     save_dir="./mlp_results/tracking_img", index=index
        # )


        save_loss_curve(losses, index, "./mlp_results/tracking_loss")

        return losses[-1],final_pose

