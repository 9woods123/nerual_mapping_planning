import torch
import torch.nn.functional as F





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

def free_space_loss(pred_sdfs, surface_depths_tensor, observe_depth, truncation=0.1, scale=10.0):
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
    mask = surface_depths_broadcast - observe_depth > truncation/scale


    if mask.any():
        
        loss_fs = ((pred_sdfs[mask] - truncation) ** 2).mean()

        print(f"[FreeSpace] #points={mask.sum().item()}, "
              f"pred_sdfs mean={pred_sdfs[mask].mean().item():.4f}, "
              f"target={truncation}")

        return loss_fs
    else:
        return torch.tensor(0.0, device=pred_sdfs.device)


def sdf_surface_loss(pred_sdfs, observe_depth, surface_depths_tensor, truncation=0.1, scale=10.0):
    """
    近表面 SDF 监督 (Near-surface Supervision)
    对 |D - d| <= tr 的点，监督 SDF 接近真实值 D - d
    """
    # 广播 surface_depths_tensor 到 observe_depth 的 shape
    surface_depths_broadcast = surface_depths_tensor.expand_as(observe_depth)
    diff = torch.abs(surface_depths_broadcast - observe_depth)
    mask = diff <= truncation / scale

    num_total = mask.numel()
    num_mask = mask.sum().item()
    ratio = num_mask / num_total if num_total > 0 else 0.0
    print(f"[Surface] mask points={num_mask}/{num_total} ({ratio*100:.2f}%) "
    f"min diff={diff.min().item():.6f}, max diff={diff.max().item():.6f}")

    if mask.any():
        gt_sdf =  scale*(surface_depths_broadcast[mask] - observe_depth[mask])   ## TODO : scale 
        loss_surface = ((pred_sdfs[mask] - gt_sdf) ** 2).mean()
        print(f"[Surface] #points={mask.sum().item()}, "
        f"pred_sdfs mean={pred_sdfs[mask].mean().item():.4f}, "
        f"gt_sdf mean={gt_sdf.mean().item():.4f}")
        num_pos = (gt_sdf > 0).sum().item()
        num_neg = (gt_sdf < 0).sum().item()
        print(f"[Surface] gt_sdf: mean={gt_sdf.mean().item():.4f}, "
                f"std={gt_sdf.std().item():.4f}, "
                f"pos={num_pos}, neg={num_neg}, zero={(gt_sdf==0).sum().item()}")
    
        return loss_surface
    else:
        print("[Surface] no points")
        return torch.tensor(0.0, device=pred_sdfs.device)



def total_loss(pred_rgb, gt_rgb, pred_d, observe_depth, surface_depths_tensor, pred_sdfs):
    """
    总损失计算，包含颜色损失、深度损失和 SDF 损失
    """
    # print(f"[Shape] pred_rgb: {pred_rgb.shape}")
    # print(f"[Shape] gt_rgb: {gt_rgb.shape}")
    # print(f"[Shape] pred_d: {pred_d.shape}")
    # print(f"[Shape] gt_depth: {gt_depth.shape}")
    # print(f"[Shape] pred_sdfs: {pred_sdfs.shape}")

    loss_color = color_loss(pred_rgb, gt_rgb)  # 颜色损失
    loss_depth = depth_loss(surface_depths_tensor, pred_d)  # 深度损失
    loss_surface = sdf_surface_loss(pred_sdfs, observe_depth, surface_depths_tensor)
    loss_free = free_space_loss(pred_sdfs, surface_depths_tensor, observe_depth)

    total_loss_value = loss_color + 50*loss_depth + 50*loss_surface + 1000*loss_free

    print(f"[Loss] color: {loss_color.item():.6f}, "
          f"depth: {loss_depth.item():.6f}, "
          f"surface_sdf: {loss_surface.item():.6f}, "
          f"free_sdf: {loss_free.item():.6f}, "
          f"total: {total_loss_value.item():.6f}")

    return total_loss_value

