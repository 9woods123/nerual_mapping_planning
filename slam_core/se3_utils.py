import torch

def se3_to_SE3(xi):
    """
    将 se(3) 6维向量转为 4x4 矩阵
    xi: (6,) torch [ωx, ωy, ωz, tx, ty, tz]
    """
    omega = xi[:3]
    theta = torch.norm(omega) + 1e-8
    I = torch.eye(3, device=xi.device)

    omega_hat = torch.tensor([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ], device=xi.device)

    R = I + (torch.sin(theta) / theta) * omega_hat + \
        ((1 - torch.cos(theta)) / (theta ** 2)) * (omega_hat @ omega_hat)

    t = xi[3:].unsqueeze(-1)

    SE3 = torch.eye(4, device=xi.device)
    SE3[:3, :3] = R
    SE3[:3, 3:4] = t
    return SE3
