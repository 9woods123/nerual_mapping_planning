import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.params import Params
from utils.utils import *

from slam_core.slam import SLAM

import cv2
import os
import numpy as np

class FrameLoader:
    def __init__(self, color_dir, depth_dir):
        self.color_dir = color_dir
        self.depth_dir = depth_dir

    def load_frame(self, idx):
        # 注意 idx 从 1 开始
        file_name = f"{idx:04d}.png"
        color_file = os.path.join(self.color_dir, file_name)
        depth_file = os.path.join(self.depth_dir, file_name)

        color_tensor = load_color_image_to_tensor(color_file)
        depth_tensor = load_depth_image_to_tensor(depth_file)

        return color_tensor, depth_tensor



# frame_loader = FrameLoader("sensor_data/rgbd_dataset_freiburg1_360/rgb_renamed",
#                           "sensor_data/rgbd_dataset_freiburg1_360/depth_renamed")

# frame_loader = FrameLoader("sensor_data/rgbd_dataset_freiburg1_xyz/rgb_renamed",
#                           "sensor_data/rgbd_dataset_freiburg1_xyz/depth_renamed")

frame_loader = FrameLoader("sensor_data/rgbd_dataset_freiburg1_plant_secret/rgb_renamed",
                          "sensor_data/rgbd_dataset_freiburg1_plant_secret/depth_renamed")

default_params=Params()
slam = SLAM(default_params)

num_frames = 900
for i in range(num_frames):
    color, depth = frame_loader.load_frame(i+1)
    slam.main_loop(color, depth, i+1, mesh_output_dir="./meshes")
