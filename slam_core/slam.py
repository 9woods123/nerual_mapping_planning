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
from network_model.nerual_render_model import SimpleMLPModel,NeuralRenderingModel

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
            params=params,
            device=self.device
        )


        self.mapper = Mapper(
            model=self.model,
            params=params,
            device=self.device
        )

        self.mesher_resolution = self.params.mapping.resolution
        self.mesh_every = self.params.mapping.mesh_every

        self.keyframes = []

        self.keyframe_every=self.params.mapping.keyframe_every
        ##TODO ,for rgbd_dataset_freiburg1_360 , must be 2 , or we get a bad result.


        self.is_first_frame=True

        self.last_pose = None
        self.prev_pose = None
        self.mesher = Mesher(-3,-3,-3,3,3,3, self.fx, self.fy, self.cx, self.cy, 
                              self.width, self.height, 
                              self.mesher_resolution,
                              self.params.camera.near, 
                              self.params.camera.far)



    def main_loop(self, color, depth, gt_pose, index, mesh_output_dir="./"):
        timestamp = time.time()

        track_loss, track_pose = self.tracker.track(color, depth, self.is_first_frame,index,gt_pose)

        # track_loss=0
        # track_pose=gt_pose
        
        if self.is_first_frame:
            track_pose=gt_pose


        map_loss = 0

        if self.is_first_frame or index % self.keyframe_every == 0:
            # 保存关键帧
            self.keyframes.append(
                Keyframe(index, track_pose, depth, color,
                        self.fx, self.fy, self.cx, self.cy, timestamp)
                        )
            
                        # --- 使用关键帧集合更新地图 ---
            map_loss, joint_opt_pose_latest = self.mapper.update_map(
                keyframes=self.keyframes,
                is_first_frame=self.is_first_frame,
                index=index
            )

            self.tracker.update_last_pose(joint_opt_pose_latest)


        trans_err, rot_err = compute_pose_error(gt_pose, track_pose, device=self.device)

        print(
            f"Frame {index:04d} | Track Loss: {track_loss:.4f} | Map Loss: {map_loss:.4f} "
            f"| Trans Err: {trans_err:.4f} m | Rot Err: {rot_err:.3f}°"
        )


        # --- 每隔 mesh_every 帧生成点云 ---
        if index % self.mesh_every == 0:
            self.mesher.generate_surface_pointcloud(
                query_fn=self.mapper.renderer.query_sdf_color_function,
                keyframe_dict=self.keyframes,
                batch_size=65536,
                save_path=f"{mesh_output_dir}/mesh_{index}.ply",
                device=self.device
            )


        if self.is_first_frame:
            self.is_first_frame = False