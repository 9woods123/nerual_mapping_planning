import torch
import torch.nn.functional as F





# 颜色损失函数
def color_loss(pred_rgb, gt_rgb):
    """
    计算颜色损失：用于评估预测的 RGB 和真实 RGB 之间的差异
    """
    # print("[color_loss] pred_rgb :", pred_rgb[:].squeeze().detach().cpu().numpy())
    # print("[color_loss] gt_rgb   :", gt_rgb[:].squeeze().detach().cpu().numpy())

    return torch.mean((pred_rgb - gt_rgb)**2)


def depth_loss(pred_depth, gt_depth):
    """
    计算深度损失：用于评估预测深度和真实深度的差异
    """

    loss = torch.mean((pred_depth - gt_depth) ** 2)

    # 打印前几个值看看
    # print("[depth_loss] pred_depth (first 5):", pred_depth[:].squeeze().detach().cpu().numpy())
    # print("[depth_loss] gt_depth   (first 5):", gt_depth[:].squeeze().detach().cpu().numpy())
    # print("[depth_loss] loss:", loss.item())

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

    # print("================================[Free Loss Debug]================================")
    # print("surface_depths_broadcast shape:", surface_depths_broadcast.shape)
    # print("observe_depth shape:", observe_depth.shape)
    # print("pred_sdfs shape:", pred_sdfs.shape)
    # print("mask True count:", mask.sum().item(), "/", mask.numel())
    # print("pred_sdfs[:] (all):", pred_sdfs.detach().cpu().numpy())
    # print("mask[:] (all):", mask.detach().cpu().numpy())


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

    # 基本信息总是打印
    # print("================================[Surface Loss Debug]================================")
    # print("surface_depths_broadcast shape:", surface_depths_broadcast.shape)
    # print("observe_depth shape:", observe_depth.shape)
    # print("pred_sdfs shape:", pred_sdfs.shape)

    # print("mask True count:", mask.sum().item(), "/", mask.numel())

    # print("surface_depths_broadcast (all):", surface_depths_broadcast.detach().cpu().numpy())
    # print("observe_depth (all):", observe_depth.detach().cpu().numpy())
    # print("D_diff (all):", D_diff.detach().cpu().numpy())
    # print("mask (all):", mask.detach().cpu().numpy())

    # print("surface_depths_broadcast[mask] (all):", surface_depths_broadcast[mask].detach().cpu().numpy())
    # print(" observe_depth[mask] (all):", observe_depth[mask].detach().cpu().numpy())
    # print(" pred_sdfs (all):", pred_sdfs.detach().cpu().numpy())



    # print("================================================================================")

    if mask.any():
        
        
        gt_sdf = scale * (surface_depths_broadcast[mask] - observe_depth[mask])

        gt_sdf = gt_sdf / truncation

        # print("before  gt_sdf (all):", gt_sdf.detach().cpu().numpy())

        gt_sdf = gt_sdf.unsqueeze(-1)  # shape [N_mask,1]

        diff = pred_sdfs[mask].view(-1,1) - gt_sdf

        # print("================================[Mask Values]================================")
        # print("gt_sdf (all):", gt_sdf.detach().cpu().numpy())
        # print("diff (all):", diff.detach().cpu().numpy())
        # print("================================================================================")

        loss_surface = (diff ** 2).mean()


        # print("pred_sdfs (first 10):", pred_sdfs[mask][:].detach().cpu().numpy())
        # print("loss_surface:", loss_surface.item())
        # print("===================================================================================")

        return loss_surface
    else:
        print("No valid near-surface points (mask all False)")
        print("===================================================================================")
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

    total_loss_value = 100*loss_color + loss_depth + 5000*loss_surface + 1000*loss_free

    print(f"[Loss] color: {loss_color.item():.6f}, "
          f"depth: {loss_depth.item():.6f}, "
          f"surface_sdf: {loss_surface.item():.6f}, "
          f"free_sdf: {loss_free.item():.6f}, "
          f"total: {total_loss_value.item():.6f}")

    return total_loss_value

