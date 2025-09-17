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


class SLAM:

    def __init__(self, fx, fy, cx, cy, truncation=0.1, mesher_resolution=0.05, mesh_every=5, device="cuda"):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.device = device

        self.model = SimpleMLPModel(input_dim=3, hidden_dim=128, num_layers=4).to(self.device)

        self.tracker = Tracker()
        self.mapper = Mapper(fx, fy, cx, cy, truncation, model=self.model,device=device)
        
        self.mesher_resolution = mesher_resolution
        self.mesh_every = mesh_every

        self.keyframes = []


    def main_loop(self, color_file_path, depth_file_path, mesh_output_dir="./"):


            color = load_color_image(image_path=color_file_path)
            depth = load_depth_image(image_path=depth_file_path)

            # --- Tracking ---
            pose = self.tracker.track()

            # --- Mapping / 更新神经隐式 SDF ---
            loss = self.mapper.update_map(color, depth, pose)
            print(f"Frame {i}: Loss = {loss:.4f}")

            # --- 保存 Keyframe ---
            self.keyframes.append(Keyframe(i, pose, depth, color, self.fx, self.fy, self.cx, self.cy, i))

            # --- 每隔 mesh_every 帧生成一次点云 ---
            if i % self.mesh_every == 0:
                mesher = Mesher(-3,-3,-3,3,3,3, self.fx, self.fy, self.cx, self.cy, 640, 480, self.mesher_resolution)
                mesher.generate_surface_pointcloud(
                    query_fn=self.mapper.renderer.query_sdf_color_function,
                    keyframe_dict=self.keyframes,
                    batch_size=65536,
                    save_path=f"{mesh_output_dir}/mesh_{i}.ply",
                    device=self.device
                )
