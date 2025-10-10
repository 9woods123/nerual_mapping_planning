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
from slam_core.se3_utils import se3_to_SE3

from network_model.loss_calculate import total_loss
from utils.utils import save_loss_curve

class Tracker:
    def __init__(self, model, fx, fy, cx, cy, width, height, distortion ,truncation=0.1, lr=1e-2, iters=20, downsample_ratio=0.001, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model
        self.renderer = Renderer(self.model, truncation)
        self.ray_casting = RayCasting(
            np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]]),
                      distortion,
            sample_ratio=downsample_ratio
        )
        self.width=width
        self.height=height

        self.lr = lr
        self.iters = iters

        self.delta_rot = torch.zeros(3, device=self.device, requires_grad=True)
        self.delta_trans = torch.zeros(3, device=self.device, requires_grad=True)

        self.optimizer = torch.optim.Adam([
            {"params": self.delta_trans, "lr": self.lr},
            {"params": self.delta_rot, "lr": 0.1*self.lr},
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


    def track(self, color, depth, is_first_frame, index):


        # === 初始化位姿 ===
        pred_pose = self.predict_pose()  # torch [4,4]

        if is_first_frame:
            return 0, pred_pose


        with torch.no_grad():
            self.delta_trans.zero_()  # 将 tensor 所有元素置0
            self.delta_rot.zero_()  # 将 tensor 所有元素置0
            self.optimizer.state.clear()   # 清空动量、方差等历史   



        losses = []
        for _ in range(self.iters):
            self.optimizer.zero_grad()

            pose_se3=torch.cat([self.delta_rot,self.delta_trans])
            pose_mat =  se3_to_SE3(pose_se3) @ pred_pose   # torch [4,4]
            
            rays_3d, rgb_values, depths = self.ray_casting.cast_rays(depth, color, pose_mat,self.height, self.width)
            all_points, all_depths, all_endpoints_3d, all_depths_end = self.ray_casting.sample_points_along_ray(
                ray_origin=pose_mat[:3, 3],
                rays_direction_list=rays_3d,
                depths_list=depths
            )
            
            pred_sdfs, pred_colors = self.model(all_points)
            rendered_color, rendered_depth = self.renderer.render(all_depths, pred_sdfs, pred_colors)

            total_loss_value,loss_color,loss_depth,loss_surface,loss_free = total_loss(rendered_color, rgb_values, rendered_depth,
                             all_depths, all_depths_end.unsqueeze(-1), pred_sdfs)


            total_loss_value.backward()
         
            # print(f"[iter {_}] loss={total_loss_value.item():.6f}")
            # print(" rot:", self.delta_rot.detach().cpu().numpy(),
            #     " grad:", self.delta_rot.grad.detach().cpu().numpy())
            # print(" trans:", self.delta_trans.detach().cpu().numpy(),
            #     " grad:", self.delta_trans.grad.detach().cpu().numpy())
            self.optimizer.step()


            losses.append(total_loss_value.item())

        # rot_angle = torch.norm(self.delta_rot).item() * 180.0 / np.pi
        # trans_norm = torch.norm(self.delta_trans).item()
        # print(f"[Iter {_}] Loss={total_loss_value.item():.4f}, "
        # f"rot={rot_angle:.3f} deg, trans={trans_norm:.4f} m")

        pose_se3=torch.cat([self.delta_rot,self.delta_trans])

        final_pose = (se3_to_SE3(pose_se3) @ pred_pose).clone().detach()  # torch [4,4]

        save_loss_curve(losses, index, "./mlp_results/tracking_loss")

        return losses[-1],final_pose

