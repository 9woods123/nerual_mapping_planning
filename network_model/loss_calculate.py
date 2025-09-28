import torch
import torch.nn.functional as F





# 颜色损失函数
def color_loss(pred_rgb, gt_rgb):
    """
    计算颜色损失：用于评估预测的 RGB 和真实 RGB 之间的差异
    """
 
    return torch.mean((pred_rgb - gt_rgb)**2)


def depth_loss(pred_depth, gt_depth):
    """
    计算深度损失：用于评估预测深度和真实深度的差异
    """

    loss = torch.mean((pred_depth - gt_depth) ** 2)

    return loss



def free_space_loss(pred_sdfs, surface_depths_tensor, observe_depth, truncation=0.1, scale=1.0):
    """
    自由空间损失 (Free-space Loss)
    对远离表面的点 |D - d| > tr，强制 SDF 接近截断距离 tr

    :param pred_sdfs: (N,) 网络预测 SDF
    :param gt_depths: (N,) 深度图真实值
    :param sampled_depths: (N,) 射线采样点深度
    :param truncation: 截断距离 tr
    :param scale: 缩放因子（可用于放大判断阈值）
    :return: 自由空间损失
    """


    surface_depths_broadcast = surface_depths_tensor.expand_as(observe_depth)
    mask = ((surface_depths_broadcast - observe_depth )> truncation/scale)


    if mask.any():
        
        loss_fs = ((pred_sdfs[mask]  - 1.0) ** 2).mean()      #TODO ： this is stupid

        return loss_fs
    else:
        return torch.tensor(0.0, device=pred_sdfs.device)






def sdf_surface_loss(pred_sdfs, observe_depth, surface_depths_tensor, truncation=0.1, scale=1.0):
    """
    近表面 SDF 监督 (Near-surface Supervision)
    对 |D - d| <= tr 的点，监督 SDF 接近真实值 D - d
    """
    # 广播 surface_depths_tensor 到 observe_depth 的 shape
    surface_depths_broadcast = surface_depths_tensor.expand_as(observe_depth)
    D_diff = torch.abs(surface_depths_broadcast - observe_depth)
    mask = (D_diff <= truncation / scale)



    if mask.any():
        
        
        gt_sdf = scale * (surface_depths_broadcast[mask] - observe_depth[mask])

        gt_sdf = gt_sdf / truncation


        gt_sdf = gt_sdf.unsqueeze(-1)  # shape [N_mask,1]

        diff = pred_sdfs[mask].view(-1,1) - gt_sdf

        loss_surface = (diff ** 2).mean()


        return loss_surface
    else:
        print("No valid near-surface points (mask all False)")
        print("===================================================================================")
        return torch.tensor(0.0, device=pred_sdfs.device)




def total_loss(pred_rgb, gt_rgb, pred_d, observe_depth, surface_depths_tensor, pred_sdfs):
    # """
    # 总损失计算，包含颜色损失、深度损失和 SDF 损失
    # """
    # print(f"[Shape] pred_rgb: {pred_rgb.shape}")
    # print(f"[Shape] gt_rgb: {gt_rgb.shape}")
    # print(f"[Shape] pred_d: {pred_d.shape}")
    # print(f"[Shape] gt_depth: {gt_depth.shape}")
    # print(f"[Shape] pred_sdfs: {pred_sdfs.shape}")

    loss_color = 20* color_loss(pred_rgb, gt_rgb)  # 颜色损失
    loss_depth = 10* depth_loss(surface_depths_tensor, pred_d)  # 深度损失
    loss_surface = 50* sdf_surface_loss(pred_sdfs, observe_depth, surface_depths_tensor)
    loss_free = 250*free_space_loss(pred_sdfs, surface_depths_tensor, observe_depth)

    total_loss_value = loss_color + loss_depth + loss_surface + loss_free


    return total_loss_value,loss_color,loss_depth,loss_surface,loss_free




def total_loss_balanced(pred_rgb, gt_rgb, pred_d, observe_depth, surface_depths_tensor, pred_sdfs, eps=1e-6):
    """
    总损失计算，自动均衡各个分量权重
    - 计算每个分量损失
    - 根据各自的均方值自动归一化
    - 返回加权总损失和每个分量
    """
    # === 单个损失 ===
    loss_c = color_loss(pred_rgb, gt_rgb)
    loss_d = depth_loss(surface_depths_tensor, pred_d)
    loss_s = sdf_surface_loss(pred_sdfs, observe_depth, surface_depths_tensor)
    loss_f = free_space_loss(pred_sdfs, surface_depths_tensor, observe_depth)

    # === 自动归一化权重 ===
    # 用每个损失的平方均值来计算权重，让梯度量级接近
    weights = []
    for l in [loss_c, loss_d, loss_s, loss_f]:
        grad_proxy = torch.sqrt(torch.mean(l**2)) + eps  # 防止除零
        weights.append(1.0 / grad_proxy)  # 倒数作为权重，使大值自动被缩小

    weights = torch.tensor(weights, device=pred_rgb.device, dtype=pred_rgb.dtype)
    weights = weights / torch.sum(weights)  # 归一化，总和为1

    # === 加权总损失 ===
    total = weights[0]*loss_c + weights[1]*loss_d + weights[2]*loss_s + weights[3]*loss_f

    return total, weights[0]*loss_c, weights[1]*loss_d, weights[2]*loss_s, weights[3]*loss_f
