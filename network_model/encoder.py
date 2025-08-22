import torch
import torch.nn as nn
import torch.nn.functional as F

from hash_grid_encoding.encoding import MultiResHashGrid



class HashGridMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=64, num_layers=1, n_levels=16, n_features_per_level=2):
        """
        初始化网络架构
        :param input_dim: 输入特征的维度 (例如空间位置 x, y, z)
        :param hidden_dim: 隐藏层的维度
        :param output_dim: 输出层的维度
        :param num_layers: MLP 层数
        :param n_levels: 多分辨率的层数，控制哈希网格的细致程度
        :param n_features_per_level: 每个网格层的特征数
        """
        super(HashGridMLPEncoder, self).__init__()
        
        self.param_init(input_dim, hidden_dim=64, output_dim=64, num_layers=1, n_levels=16, n_features_per_level=2)

        # 初始化 MultiResHashGrid 编码器
        self.hash_grid_encoder = MultiResHashGrid(
            dim=input_dim, 
            n_levels=n_levels, 
            n_features_per_level=n_features_per_level
        )

        # MLP 网络结构
        layers = [nn.Linear(self.hash_grid_encoder.output_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        # 将层堆叠为网络
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入数据，形状为 (batch_size, input_dim)
        :return: 输出数据，形状为 (batch_size, output_dim)
        """
        # 通过 MultiResHashGrid 编码器生成特征
        hash_grid_features = self.hash_grid_encoder(x)
        
        # 扁平化输入以适应 MLP 输入格式
        hash_grid_features = hash_grid_features.view(hash_grid_features.size(0), -1)
        
        # 通过 MLP 网络进行处理
        return self.network(hash_grid_features)
    
    def param_init(self,input_dim, hidden_dim=64, output_dim=64, num_layers=1, n_levels=16, n_features_per_level=2):

        self.output_dim=output_dim
        ##TODO 
    
    def get_encoder_output_dim(self):

        return self.output_dim


# 测试网络
if __name__ == "__main__":
    # 假设输入维度是 3 (例如空间位置 x, y, z)，输出维度为 128
    input_dim = 3
    hidden_dim = 64
    output_dim = 64
    batch_size = 16
    n_levels = 16
    n_features_per_level = 2

    # 初始化编码器
    encoder = HashGridMLPEncoder(input_dim, hidden_dim, output_dim, n_levels=n_levels, n_features_per_level=n_features_per_level)

    # 模拟输入
    x = torch.rand(batch_size, input_dim)

    # 前向传播
    output = encoder(x)
    print(f"Output shape: {output.shape}")  # 应该输出 (batch_size, output_dim)
