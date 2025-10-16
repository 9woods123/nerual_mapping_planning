import torch
import numpy as np

class Renderer:
    def __init__(self, model, device="cuda", tr=0.1):
        self.tr = tr
        self.model = model
        self.device = device


    def render(self, depth_values,sdf_values,color_values):
        """
        depth_values: (N_rays, N_samples)
        color_values: (N_rays, N_samples, 3)
        sdf_values: (N_rays, N_samples)
        """
        

        weights = self.compute_weights(sdf_values,depth_values)  # (N_rays, N_samples)

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


    def compute_weights(self, sdf_values,depth_values):
        # 使用 SDF 计算权重
        sigmoid1 = torch.sigmoid(sdf_values / self.tr)
        sigmoid2 = torch.sigmoid(-sdf_values / self.tr)
        # sigmoid1 = torch.sigmoid(sdf_values)
        # sigmoid2 = torch.sigmoid(-sdf_values)
        weights = sigmoid1 * sigmoid2

        # 2. 找到沿每条射线第一次穿过表面的索引
        # signs: 相邻采样点 SDF 符号相乘
        signs = sdf_values[:, 1:] * sdf_values[:, :-1]  # [N_rays, N_samples-1]

        mask_surface = (signs < 0).float()  # 符号翻转位置标记 1
        first_surface_idx = torch.argmax(mask_surface, dim=1)  # [N_rays]


        # 3. 对应深度
        first_surface_idx_clamped = torch.clamp(first_surface_idx, max=depth_values.shape[1]-1)



        inds = first_surface_idx_clamped.long()  # [N_rays, 1]



        z_min = torch.gather(depth_values, 1, inds)  # [N_rays,1]

        # 4. 屏蔽第一次表面后的点
        mask = (depth_values <= z_min +  self.tr).float()  # [N_rays, N_samples]



        weights = weights * mask.unsqueeze(-1)

        # 5. 归一化
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)

        return weights



    @torch.no_grad()
    def query_sdf_color_function(self, points, device='cuda'):
        """
        输入: 
            points: (N, 3) torch.Tensor 或 np.ndarray
            model: 你的神经隐式模型, 需要有 sdf/color 的输出
        输出:
            sdf_values: (N, 1)
            color_values: (N, 3)
        """
        if not torch.is_tensor(points):
            points = torch.from_numpy(points).float().to(device)
        else:
            points = points.float().to(device)

        # 假设 model 返回 (sdf, color)
        sdf_values, color_values = self.model(points)

        return sdf_values, color_values