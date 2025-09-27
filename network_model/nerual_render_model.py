import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from network_model.one_blob_encoding import OneBlobEncoding
from network_model.encoder import HashGridMLPEncoder
from network_model.decoder import ColorDecoder, GeometryDecoder



class NeuralRenderingModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128,num_layers=4, encoding_dim=256,num_freqs=90):
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
        self.position_encoding =  LearnableGaussianEncoding(input_dim=input_dim, num_freqs=num_freqs)
        
        pe_out_dim =   input_dim+num_freqs   # 原始 + sin/cos


        # Hash Grid 编码器
        self.hash_grid_encoder = HashGridMLPEncoder(input_dim, hidden_dim=128, output_dim=128)

        # 几何解码器
        self.geometry_decoder = GeometryDecoder(pe_out_dim + self.hash_grid_encoder.get_encoder_output_dim(), hidden_dim=64, h_dim=64, s_dim=1)  # SDF 或其他几何信息
        
        # 颜色解码器
        self.color_decoder = ColorDecoder(pe_out_dim + self.geometry_decoder.get_geo_features_output_dim(), hidden_dim=64, color_dim=3)  # RGB 输出
        


        
    def forward(self, x):
        """
        前向传播：通过 One-Blob 和 Hash Grid 编码，然后生成几何信息和颜色
        
        :param x: 输入坐标，形状为 (batch_size, input_dim)
        :return: 几何信息 (如 SDF) 和 颜色 (RGB)
        """

        oneblob_features = self.position_encoding(x)        # One-Blob Encoding
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

class MLPBlock(nn.Module):
    def __init__(self, in_dim ,out_dim, hidden_dim, num_layers=2):
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i==0:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))  
            if i !=(num_layers-1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))  
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)

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

class GaussianPositionalEncoding(nn.Module):
    def __init__(self, input_dim=3, num_freqs=64, sigma=10.0):
        """
        高斯随机位置编码
        :param input_dim: 输入维度 (e.g., xyz=3)
        :param num_freqs: 输出特征的一半维度 (sin 和 cos 会翻倍)
        :param sigma: 控制频率分布的尺度
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs

        # 随机高斯矩阵 B ~ N(0, sigma^2)
        B = torch.randn(input_dim, num_freqs) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        """
        :param x: [B, input_dim]
        :return: [B, 2 * num_freqs]
        """
        # [B, input_dim] @ [input_dim, num_freqs] = [B, num_freqs]
        x_proj = 2 * math.pi * x @ self.B  

        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class LearnableGaussianEncoding(nn.Module):
    def __init__(self, input_dim=3, num_freqs=64, sigma=25.0, learnable=True):
        """
        Learnable Gaussian Positional Encoding (Fourier Feature Networks + SIREN)
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        self.sigma = sigma

        B = torch.randn(num_freqs, input_dim) * sigma
        
        if learnable:
            self.B = nn.Parameter(B)  # 可学习
        else:
            self.register_buffer("B", B)  # 固定

    def forward(self, x):
        # x: [B, input_dim]
        # [B, input_dim] @ [input_dim, num_freqs]^T = [B, num_freqs]

        x_proj =  x @ self.B.T  

        return torch.cat([x, torch.sin(x_proj)], dim=-1)  # 只用 sin(Bp)，符合你引用的文献


class SimpleMLPModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=4, num_freqs=80):
        super().__init__()

        self.pe = LearnableGaussianEncoding(input_dim=input_dim, num_freqs=num_freqs)
        pe_dim =   input_dim+num_freqs   # 原始 + sin/cos

        self.mlp_block = MLPBlock(in_dim= pe_dim, out_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        self.sdf_feat = nn.Sequential(
            nn.Linear(pe_dim+hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.sdf_out = nn.Linear(hidden_dim, 1)  # 只输出一个 SDF 值

        # RGB head (接收 sdf feature 作为输入)
        self.rgb_head = nn.Sequential(
            nn.Linear(pe_dim+hidden_dim+hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )


    def forward(self, x):
        
        ## normlize to [-1,1]
        x=x/5.0

        x_pe = self.pe(x)
        features = self.mlp_block(x_pe)
        features=torch.cat([x_pe,features], dim=-1) 

        sdf_feat = self.sdf_feat(features)
        sdf = self.sdf_out(sdf_feat)               # 3. sdf 值

        rgb_input = torch.cat([features, sdf_feat], dim=-1)  # 4. 拼接
        rgb = self.rgb_head(rgb_input)

        return features, sdf, rgb
