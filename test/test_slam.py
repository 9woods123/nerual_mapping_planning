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
        color_file = os.path.join(self.color_dir, f"color_{idx}.png")
        depth_file = os.path.join(self.depth_dir, f"depth_{idx}.png")

        color = load_color_image(color_file)
        color, c_min_val, c_max_val = normalize_numpy(color, 0, 255)

        depth = load_depth_image(depth_file)
        return color, depth



frame_loader = FrameLoader("sensor_data/color", "sensor_data/depth")
default_params=Params()
slam = SLAM(default_params)

num_frames = 100
for i in range(num_frames):
    color, depth = frame_loader.load_frame(i+1)
    slam.main_loop(color, depth, i+1, mesh_output_dir="./meshes")
