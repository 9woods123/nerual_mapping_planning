import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_model.nerual_render_model import NeuralRenderingModel
import torch


# class MappingNode(nn.Module):
#     def __init__(self, input_dim=3, grid_resolution=128, grid_size=1.0, model=None):
#         """
#         地图节点类，用于管理和更新隐式地图。
        
#         :param input_dim: 输入坐标的维度（默认为 3D）
#         :param grid_resolution: 网格分辨率
#         :param grid_size: 网格大小
#         :param model: 用于生成SDF和颜色的神经网络模型
#         """
#         super(MappingNode, self).__init__()

#         self.input_dim = input_dim
#         self.grid_resolution = grid_resolution
#         self.grid_size = grid_size
#         self.model = model

#         # 初始化网格，假设网格的每个元素表示一个SDF值和颜色值
#         self.grid_sdf = torch.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=torch.float32)
#         self.grid_color = torch.zeros((grid_resolution, grid_resolution, grid_resolution, 3), dtype=torch.float32)  # RGB颜色


#     def forward(self, positions):
#         """
#         根据给定的位置预测 SDF 和颜色值
        
#         :param positions: 形状为 (batch_size, input_dim) 的输入坐标
#         :return: 形状为 (batch_size, 1) 的 SDF 和形状为 (batch_size, 3) 的颜色
#         """
#         assert self.model is not None, "Model is not defined!"
        
#         sdf_values, rgb_values = self.model(positions)  # 使用模型预测 SDF 和颜色值
#         return sdf_values, rgb_values


#     def update(self, new_positions):
#         """
#         接收新的位姿（或深度信息），更新地图
        
#         :param new_positions: 新的位姿数据（例如，相机的 3D 位置信息）
#         """
#         # 获取新位置的 SDF 和颜色值
#         sdf_values, rgb_values = self(new_positions)  # 使用模型生成的 SDF 和颜色
        
#         # 将位置映射到网格索引
#         indices = self.get_grid_indices(new_positions)

#         # 更新对应位置的 SDF 和颜色
#         for idx, sdf, rgb in zip(indices, sdf_values, rgb_values):
#             self.grid_sdf[idx[0], idx[1], idx[2]] = sdf.item()
#             self.grid_color[idx[0], idx[1], idx[2]] = rgb.detach().cpu().numpy()

#     def get_grid_indices(self, positions):
#         """
#         将空间位置映射到网格索引
        
#         :param positions: 形状为 (batch_size, input_dim) 的位置坐标
#         :return: 返回网格中的索引
#         """
#         # 假设 positions 已经归一化到 [0, 1] 范围内
#         indices = (positions / self.grid_size * self.grid_resolution).long()
#         return indices

#     def get_map(self):
#         """
#         获取当前地图的 SDF 和颜色
#         """
#         return self.grid_sdf, self.grid_color

#     def clear_map(self):
#         """
#         清除当前地图数据
#         """
#         self.grid_sdf = torch.zeros_like(self.grid_sdf)
#         self.grid_color = torch.zeros_like(self.grid_color)



# # 测试代码
# def test_mapping():
#     # 假设你已经有了训练好的神经网络模型
#     model = NeuralRenderingModel(input_dim=3, hidden_dim=64, encoding_dim=64)  # 或者加载已训练的模型
#     mapping_node = MappingNode(model=model)


#     # 随机生成位置数据 (batch_size, 3)
#     batch_size = 16
#     new_positions = torch.rand(batch_size, 3)


#     # 更新地图
#     mapping_node.update(new_positions)
    
#     # 获取当前地图
#     current_map_sdf, current_map_color = mapping_node.get_map()
#     print(f"Updated map shape (SDF): {current_map_sdf.shape}")
#     print(f"Updated map shape (Color): {current_map_color.shape}")

#     # 清除地图
#     mapping_node.clear_map()
#     cleared_map_sdf, cleared_map_color = mapping_node.get_map()
#     print(f"Cleared map shape (SDF): {cleared_map_sdf.shape}")
#     print(f"Cleared map shape (Color): {cleared_map_color.shape}")


# if __name__ == "__main__":
#     test_mapping()
