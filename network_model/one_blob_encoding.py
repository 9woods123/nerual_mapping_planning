import torch
import torch.nn as nn
import torch.nn.functional as F
from hash_grid_encoding import MultiResHashGrid


class OneBlobEncoding(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        One-Blob Encoding: 将输入坐标映射到一个特征空间
        
        :param input_dim: 输入空间的维度（例如 3D 坐标）
        :param output_dim: 输出特征空间的维度
        """
        super(OneBlobEncoding, self).__init__()
        
        # 线性层，将输入坐标映射到一个高维特征空间
        self.encoding_layer = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        """
        前向传播：将输入的空间坐标通过 One-Blob 编码映射到特征空间
        
        :param x: 输入坐标，形状为 (batch_size, input_dim)
        :return: 编码后的特征向量，形状为 (batch_size, output_dim)
        """
        # 通过线性层编码
        encoded_features = self.encoding_layer(x)
        
        return encoded_features
