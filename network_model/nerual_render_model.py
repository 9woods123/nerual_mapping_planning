import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network_model.one_blob_encoding import OneBlobEncoding
from network_model.encoder import HashGridMLPEncoder
from network_model.decoder import ColorDecoder, GeometryDecoder



class NeuralRenderingModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, encoding_dim=256):
        """
        神经渲染模型
        
        :param input_dim: 输入坐标的维度（如 3D 坐标）
        :param hidden_dim: 隐藏层维度
        :param encoding_dim: One-Blob 编码后的特征维度
        :param tr: 截断距离
        :param M: 每条射线采样的点数
        """
        super(NeuralRenderingModel, self).__init__()
        
        # One-Blob 编码器
        self.oneblob_encoding = OneBlobEncoding(input_dim, encoding_dim,output_dim=encoding_dim)
        
        # Hash Grid 编码器
        self.hash_grid_encoder = HashGridMLPEncoder(input_dim, hidden_dim=128, output_dim=128)

        # 几何解码器
        self.geometry_decoder = GeometryDecoder(encoding_dim + self.hash_grid_encoder.get_encoder_output_dim(), hidden_dim=64, h_dim=64, s_dim=1)  # SDF 或其他几何信息
        
        # 颜色解码器
        self.color_decoder = ColorDecoder(encoding_dim + self.geometry_decoder.get_geo_features_output_dim(), hidden_dim=64, color_dim=3)  # RGB 输出
        


        
    def forward(self, x):
        """
        前向传播：通过 One-Blob 和 Hash Grid 编码，然后生成几何信息和颜色
        
        :param x: 输入坐标，形状为 (batch_size, input_dim)
        :return: 几何信息 (如 SDF) 和 颜色 (RGB)
        """

        oneblob_features = self.oneblob_encoding(x)        # One-Blob Encoding
        hash_grid_features = self.hash_grid_encoder(x)        # Hash Grid Encoding

        
        geometry_input = torch.cat([oneblob_features, hash_grid_features], dim=-1)  # 将 One-Blob 和 Hash Grid 特征拼接
        geo_features, sdf = self.geometry_decoder(geometry_input)  # 生成几何信息（如 SDF）
        
        color_input = torch.cat([oneblob_features, geo_features], dim=-1)        # 将 One-Blob 特征和 Geometry Decoder 输出拼接
        rgb = self.color_decoder(color_input)        # 生成颜色（如 RGB）



        return geo_features, sdf, rgb


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim) 
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        h = self.input_proj(x)
        return h + self.block(h)


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim=3, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim

        freq_bands = 2.0 ** torch.arange(num_freqs).float() * torch.pi  # [1,2,4,8,...]
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        """
        x: [B, input_dim]
        returns: [B, input_dim * 2 * num_freqs]
        """
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
            
        return torch.cat(out, dim=-1)


class SimpleMLPModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=4, num_freqs=120):
        super().__init__()

        self.pe = PositionalEncoding(input_dim=input_dim, num_freqs=num_freqs)
        pe_dim = input_dim * (2*num_freqs +1)  # 原始 + sin/cos


        self.res_block = ResidualBlock(pe_dim, hidden_dim, num_layers)
        # self.res_block = ResidualBlock(input_dim, hidden_dim, num_layers)

        # SDF head
        self.sdf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # RGB head
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_pe = self.pe(x)

        features = self.res_block(x_pe)
        sdf = self.sdf_head(features)
        rgb = self.rgb_head(features)
        return features, sdf, rgb

