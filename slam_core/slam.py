# slam_core/slam_system.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2, glob
import numpy as np
from slam_core.tracking import Tracker
from slam_core.mapping import Mapper
from slam_core.keyframe import Keyframe
from visualization.mesher import Mesher
from network_model.nerual_render_model import SimpleMLPModel

from utils.utils import *
from utils.params import Params
import time  # 用于生成时间戳


class SLAM:

    def __init__(self, params, device="cuda"):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.params = params

        # === 相机参数 ===
        self.fx = self.params.camera.fx
        self.fy = self.params.camera.fy
        self.cx = self.params.camera.cx
        self.cy = self.params.camera.cy
        self.height=self.params.camera.height
        self.width=self.params.camera.width

        self.device = device

        self.model = SimpleMLPModel(
            input_dim=self.params.model.input_dim,
            hidden_dim=self.params.model.hidden_dim,
            num_layers=self.params.model.num_layers
        ).to(self.device)


        self.tracker = Tracker(
            model=self.model,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            truncation=self.params.mapping.truncation,
            lr=self.params.tracking.lr,
            iters=self.params.tracking.iters,
            downsample_ratio=self.params.tracking.downsample_ratio,
            device=self.device
        )


        self.mapper = Mapper(
            model=self.model,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            truncation=self.params.mapping.truncation,
            lr=self.params.mapping.lr,
            iters=self.params.mapping.iters,
            downsample_ratio=self.params.mapping.downsample_ratio,
            device=self.device
        )

        self.mesher_resolution = self.params.mapping.resolution
        self.mesh_every = 1

        self.keyframes = []


        self.is_first_frame=True

    def main_loop(self, color, depth, index,mesh_output_dir="./"):
            
            timestamp = time.time()  # 当前时间戳，单位秒
            
            if self.is_first_frame:
                track_pose=np.eye(4, dtype=np.float32)
                loss = self.mapper.update_map(color, depth, track_pose, self.is_first_frame)
            else:
                track_pose, _ = self.tracker.track(color, depth)
                loss = self.mapper.update_map(color, depth, track_pose, self.is_first_frame)
            

            # --- 保存 Keyframe ---
            self.keyframes.append(Keyframe(index, track_pose, depth, color, self.fx, self.fy, self.cx, self.cy, timestamp))

            # --- 每隔 mesh_every 帧生成一次点云 ---
            if index % self.mesh_every == 0:
                mesher = Mesher(-3,-3,-3,3,3,3, self.fx, self.fy, self.cx, self.cy, 640, 480, self.mesher_resolution)
                mesher.generate_surface_pointcloud(
                    query_fn=self.mapper.renderer.query_sdf_color_function,
                    keyframe_dict=self.keyframes,
                    batch_size=65536,
                    save_path=f"{mesh_output_dir}/mesh_{index}.ply",
                    device=self.device
                )

