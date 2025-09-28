import torch

# def se3_to_SE3(xi):
#     """
#     将 se(3) 6维向量转为 4x4 矩阵
#     xi: (6,) torch [ωx, ωy, ωz, tx, ty, tz]
#     """
#     omega = xi[:3]
#     theta = torch.norm(omega) + 1e-8
#     I = torch.eye(3, device=xi.device)

#     # 正确构造 omega_hat，保持梯度链
#     omega_hat = torch.zeros(3, 3, device=xi.device, dtype=xi.dtype)
#     omega_hat[0,1] = -omega[2]
#     omega_hat[0,2] =  omega[1]
#     omega_hat[1,0] =  omega[2]
#     omega_hat[1,2] = -omega[0]
#     omega_hat[2,0] = -omega[1]
#     omega_hat[2,1] =  omega[0]
    
#     R = I + (torch.sin(theta) / theta) * omega_hat + \
#         ((1 - torch.cos(theta)) / (theta ** 2)) * (omega_hat @ omega_hat)

#     t = xi[3:].unsqueeze(-1)

#     SE3 = torch.eye(4, device=xi.device)
#     SE3[:3, :3] = R
#     SE3[:3, 3:4] = t
#     return SE3


def se3_to_SE3(xi):
    """
    将 se(3) 6维向量转为 4x4 矩阵 (指数映射)
    xi: (6,) torch.Tensor [ωx, ωy, ωz, tx, ty, tz]
    """
    omega = xi[:3]
    v = xi[3:]

    theta = torch.norm(omega) + 1e-8
    I = torch.eye(3, device=xi.device, dtype=xi.dtype)

    # 反对称矩阵 ω̂
    omega_hat = torch.zeros(3, 3, device=xi.device, dtype=xi.dtype)
    omega_hat[0,1] = -omega[2]
    omega_hat[0,2] =  omega[1]
    omega_hat[1,0] =  omega[2]
    omega_hat[1,2] = -omega[0]
    omega_hat[2,0] = -omega[1]
    omega_hat[2,1] =  omega[0]

    # Rodrigues 公式计算旋转
    R = I + (torch.sin(theta)/theta) * omega_hat + \
        ((1 - torch.cos(theta))/(theta**2)) * (omega_hat @ omega_hat)

    # V 矩阵修正平移
    V = I + ((1 - torch.cos(theta))/(theta**2)) * omega_hat + \
        ((theta - torch.sin(theta))/(theta**3)) * (omega_hat @ omega_hat)

    t = V @ v.unsqueeze(-1)  # (3,1)

    SE3 = torch.eye(4, device=xi.device, dtype=xi.dtype)
    SE3[:3, :3] = R
    SE3[:3, 3:] = t
    return SE3
