import torch
import torch.nn.functional as F

# 基础损失函数：均方误差和L1损失
def mse_loss(pred, target):
    return F.mse_loss(pred, target)

def l1_loss(pred, target):
    return torch.abs(pred - target).mean()

# SDF 损失函数
def sdf_loss(pred_sdf, gt_sdf):
    """
    计算 SDF 损失：用于评估预测的 SDF 和真实 SDF 之间的差异
    """
    loss = torch.mean((pred_sdf - gt_sdf)**2)
    return loss

# 自由空间损失
def free_space_loss(pred_sdf, gt_depth, tr):
    """
    计算自由空间损失：用于评估远离表面（自由空间）点的 SDF 预测值
    """
    # 获取远离表面区域的点
    mask_free_space = torch.abs(gt_depth - tr) > tr  # 自由空间区域
    
    loss_fs = torch.mean((pred_sdf[mask_free_space] - tr)**2)
    return loss_fs

# SDF 损失（近表面点）
def sdf_surface_loss(pred_sdf, gt_depth, pred_d, tr):
    """
    计算 SDF 损失：用于评估预测的 SDF 和真实的 SDF 之间的差异
    """
    # 获取在截断区域内的点
    mask_truncation = torch.abs(gt_depth - pred_d) <= tr  # 截断区域
    
    loss_sdf = torch.mean((pred_sdf[mask_truncation] - (gt_depth[mask_truncation] - pred_d))**2)
    return loss_sdf

# 颜色损失函数
def color_loss(pred_rgb, gt_rgb):
    """
    计算颜色损失：用于评估预测的 RGB 和真实 RGB 之间的差异
    """
    return torch.mean((pred_rgb - gt_rgb)**2)


# 颜色损失函数
def depth_loss(pred_depth, gt_depth):
    """
    计算颜色损失：用于评估预测的 RGB 和真实 RGB 之间的差异
    """
    return torch.mean((pred_depth - gt_depth)**2)


def total_loss(pred_rgb, gt_rgb, pred_d, gt_depth):
    """
    计算总损失：结合 SDF 损失、自由空间损失、颜色损失等
    """

    loss_color = color_loss(pred_rgb, gt_rgb)  # 颜色损失
    loss_depth = depth_loss(gt_depth, pred_d)  # 颜色损失

    # 可调整损失权重
    total_loss_value = loss_color + loss_depth 
    return total_loss_value



# # 总损失函数
# def total_loss(pred_sdf, pred_rgb, gt_rgb, gt_depth, pred_d, tr):
#     """
#     计算总损失：结合 SDF 损失、自由空间损失、颜色损失等
#     """
#     # 计算 SDF 损失
#     loss_sdf = sdf_surface_loss(pred_sdf, gt_depth, pred_d, tr)  # 使用近表面 SDF 损失
#     loss_fs = free_space_loss(pred_sdf, gt_depth, tr)  # 使用自由空间损失
#     loss_color = color_loss(pred_rgb, gt_rgb)  # 颜色损失

#     # 可调整损失权重
#     total_loss_value = loss_sdf + loss_fs + loss_color
#     return total_loss_value, loss_sdf, loss_fs, loss_color
