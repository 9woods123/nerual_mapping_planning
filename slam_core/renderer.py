import torch
import numpy as np

class Renderer:
    def __init__(self, tr=0.01):
        self.tr = tr

    def render(self, depth_values,sdf_values,color_values):
        """
        depth_values: (N_rays, N_samples)
        color_values: (N_rays, N_samples, 3)
        sdf_values: (N_rays, N_samples)
        """
        


        weights = self.compute_weights(sdf_values)  # (N_rays, N_samples)

        weights_sum = torch.sum(weights, dim=1) + 1e-8  # 防止除零


        # [Render] depth_values shape: torch.Size([231, 20])
        # [Render] color_values shape: torch.Size([231, 20, 3])
        # [Render] sdf_values shape: torch.Size([231, 20, 1])
        # [Render] weights shape: torch.Size([231, 20, 1])
        # [Render] weights_sum shape: torch.Size([231, 1, 1])
        # [Render] weight_color shape: torch.Size([231, 3])



        weight_color=torch.sum(weights * color_values, dim=1)
        rendered_color = weight_color / weights_sum


        weight_depth=torch.sum(weights.squeeze(-1) * depth_values, dim=1).unsqueeze(-1)
        rendered_depth = weight_depth/ weights_sum


        # print(f"[Render] rendered_color shape: {rendered_color.shape}")
        # print(f"[Render] rendered_depth shape: {rendered_depth.shape}")

        return rendered_color, rendered_depth


    def compute_weights(self, sdf_values):
        # 使用 SDF 计算权重
        sigmoid1 = torch.sigmoid(sdf_values / self.tr)
        sigmoid2 = torch.sigmoid(-sdf_values / self.tr)
        weights = sigmoid1 * sigmoid2

        return weights

