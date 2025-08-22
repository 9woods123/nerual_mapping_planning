
import torch
import torch.nn as nn
import torch.nn.functional as F

from network_model.one_blob_encoding import OneBlobEncoding
from network_model.encoder import  HashGridMLPEncoder
from network_model.decoder import ColorDecoder,GeometryDecoder


class NeuralRenderingModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, encoding_dim=256, output_dim=1):
        """
        神经渲染模型
        
        :param input_dim: 输入坐标的维度（如 3D 坐标）
        :param hidden_dim: 隐藏层维度
        :param encoding_dim: One-Blob 编码后的特征维度
        :param output_dim: 输出维度（如 1 表示 SDF，3 表示颜色）
        """
        super(NeuralRenderingModel, self).__init__()
        
        # One-Blob 编码器
        self.oneblob_encoding = OneBlobEncoding(input_dim, encoding_dim)
        
        # Hash Grid 编码器
        self.hash_grid_encoder = HashGridMLPEncoder(input_dim, hidden_dim=64, output_dim=64)

        # 几何解码器
        self.geometry_decoder = GeometryDecoder(encoding_dim + self.hash_grid_encoder.get_encoder_output_dim(), hidden_dim=64, h_dim=64, s_dim=1)  # SDF 或其他几何信息
        ## 几何解码器 outputs geo features "h",  and SDF value s.
        
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

        
        return geo_features,sdf, rgb