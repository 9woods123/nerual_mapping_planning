import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometryDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, h_dim, s_dim=1):
        """
        :param input_dim: 输入的特征维度 (例如，点的坐标维度)
        :param hidden_dim: 隐藏层的维度
        :param h_dim: 特征向量 h 的维度
        :param s_dim: SDF 值的维度，通常为 1
        """

        super(GeometryDecoder, self).__init__()

        self.output_dim=h_dim+s_dim
        self.h_dim=h_dim

        # MLP 用于生成特征向量 h 和 SDF 值 s
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出层，分别预测 h 和 s
        self.fc_h = nn.Linear(hidden_dim, h_dim)  # 输出特征向量 h
        self.fc_s = nn.Linear(hidden_dim, s_dim)  # 输出 SDF 值 s


    def forward(self, x):
        """
        前向传播
        :param x: 输入数据，形状为 (batch_size, input_dim)
        :return: 输出的特征向量 h 和 SDF 值 s
        """
        # 前向传播过程
        x = F.relu(self.fc1(x))  # 第一层激活
        x = F.relu(self.fc2(x))  # 第二层激活

        h = self.fc_h(x)  # 输出高维特征向量 h
        s = self.fc_s(x)  # 输出 SDF 值 s

        return h, s
    
    def get_decoder_output_dim(self):

        return self.output_dim
    
    def get_geo_features_output_dim(self):

        return self.h_dim

class ColorDecoder(nn.Module):
    def __init__(self, h_dim, hidden_dim, color_dim=3):
        """
        :param h_dim: 输入特征向量 h 的维度
        :param hidden_dim: 隐藏层的维度
        :param color_dim: 输出的颜色维度，默认是 3，表示 RGB
        """
        super(ColorDecoder, self).__init__()


        # MLP 用于生成颜色
        self.fc1 = nn.Linear(h_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_color = nn.Linear(hidden_dim, color_dim)  # 输出颜色


    def forward(self, h):
        """
        前向传播
        :param h: 输入的特征向量，形状为 (batch_size, h_dim)
        :return: 输出颜色，形状为 (batch_size, color_dim)
        """
        # 前向传播过程
        x = F.relu(self.fc1(h))  # 第一层激活
        x = F.relu(self.fc2(x))  # 第二层激活
        color = torch.sigmoid(self.fc_color(x))  # 输出颜色，通常使用 sigmoid 将值限制在 [0, 1] 范围内

        return color
