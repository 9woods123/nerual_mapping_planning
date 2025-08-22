import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_model.nerual_render_model import NeuralRenderingModel

import torch
import torch

allocated_memory = torch.cuda.memory_allocated()
reserved_memory = torch.cuda.memory_reserved()
free_memory = reserved_memory - allocated_memory

allocated_memory_gb = allocated_memory / (1024 ** 3)
reserved_memory_gb = reserved_memory / (1024 ** 3)
free_memory_gb = free_memory / (1024 ** 3)

print(f"Allocated memory: {allocated_memory_gb:.2f} GB")
print(f"Reserved memory: {reserved_memory_gb:.2f} GB")
print(f"Free memory: {free_memory_gb:.2f} GB")


def test_neural_rendering_model():
    # 设置设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数
    input_dim = 3  # 假设输入是 3D 坐标（x, y, z）
    hidden_dim = 64
    encoding_dim = 64
    output_dim = 1  # 假设输出 SDF，维度为 1

    # 初始化模型
    model = NeuralRenderingModel(input_dim=input_dim, hidden_dim=hidden_dim, encoding_dim=encoding_dim)
    model.to(device)  # 将模型移动到设备

    # 创建随机输入数据 (batch_size, input_dim)
    batch_size = 16  # 假设批量大小为 16
    x = torch.rand(batch_size, input_dim).to(device)  # 随机生成输入坐标 (batch_size, input_dim)

    # 打印模型结构
    print("Model architecture:")
    print(model)


    # 前向传播
    geo_features, sdf, rgb = model(x)

    # 输出结果的形状
    print(f"geo_features shape: {geo_features.shape}")  # 生成的几何特征（如 h）
    print(f"sdf shape: {sdf.shape}")  # 生成的 SDF 值
    print(f"rgb shape: {rgb.shape}")  # 生成的 RGB 颜色

    # 进行简单的断言检查，确保输出的维度正确
    assert geo_features.shape == (batch_size, 64), f"Expected geo_features shape (batch_size, 64), got {geo_features.shape}"
    assert sdf.shape == (batch_size, 1), f"Expected sdf shape (batch_size, 1), got {sdf.shape}"
    assert rgb.shape == (batch_size, 3), f"Expected rgb shape (batch_size, 3), got {rgb.shape}"

    print("Test passed successfully!")

if __name__ == "__main__":
    test_neural_rendering_model()
