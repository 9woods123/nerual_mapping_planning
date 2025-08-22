import torch
import torch.nn as nn
import torch.nn.functional as F

class OneBlobEncoding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        One-Blob Encoding: 将输入坐标映射到一个特征空间
        
        :param input_dim: 输入空间的维度（例如 3D 坐标）
        :param hidden_dim: 隐藏层的维度
        :param output_dim: 输出特征空间的维度
        """
        super(OneBlobEncoding, self).__init__()
        
        # 第一层线性变换
        self.layer1 = nn.Linear(input_dim, hidden_dim)   
        # 第二层线性变换
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    
    def forward(self, x):
        """
        前向传播：将输入的空间坐标通过 One-Blob 编码映射到特征空间
        
        :param x: 输入坐标，形状为 (batch_size, input_dim)
        :return: 编码后的特征向量，形状为 (batch_size, output_dim)
        """
        # 第一层：通过线性变换和 ReLU 激活
        x = F.relu(self.layer1(x))
        
        # 第二层：通过线性变换
        encoded_features = self.layer2(x)
        
        return encoded_features
