import torch
import numpy as np

class Renderer:
    def __init__(self, model, tr=0.1):
        """
        渲染类，负责基于神经网络模型进行射线渲染
        
        :param model: 神经渲染模型
        :param tr: 截断距离
        """
        self.model = model
        self.tr = tr

    def render(self, depth_values, color_values, sdf_values):
        """
        渲染颜色和深度
        
        :param depth_values: 每个采样点的深度值
        :param color_values: 每个采样点的颜色值
        :return: 渲染后的颜色和深度
        """
        weights = self.compute_weights(sdf_values)

        rendered_color = torch.sum(weights * color_values, dim=0) / torch.sum(weights, dim=0)
        rendered_depth = torch.sum(weights * depth_values, dim=0) / torch.sum(weights, dim=0)

        return rendered_color, rendered_depth

    def compute_weights(self, sdf_values):
        """
        根据 SDF 值计算权重
        
        :param sdf_values: SDF 值
        :return: 计算得到的权重
        """
        sigmoid1 = torch.sigmoid(sdf_values / self.tr)
        sigmoid2 = torch.sigmoid(-sdf_values / self.tr)
        weights = sigmoid1 * sigmoid2
    
        return weights

