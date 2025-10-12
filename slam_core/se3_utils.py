import torch


def orthogonalize_rotation(R: torch.Tensor) -> torch.Tensor:
    """
    对旋转矩阵进行正交化，保证 R^T R = I
    使用 SVD 方法。
    
    Args:
        R: (3,3) 旋转矩阵，可能有数值误差
    
    Returns:
        R_ortho: (3,3) 正交化后的旋转矩阵
    """
    U, _, Vt = torch.linalg.svd(R)
    R_ortho = U @ Vt
    return R_ortho


def se3_to_SE3(xi):
    """
    se(3) -> SE(3) 指数映射
    xi: (6,) torch.Tensor, [ωx, ωy, ωz, tx, ty, tz]
    """

    omega = xi[:3]
    v = xi[3:]
    theta = torch.norm(omega)

    I = torch.eye(3, device=xi.device, dtype=xi.dtype)

    omega_hat = torch.zeros(3, 3, device=xi.device, dtype=xi.dtype)
    omega_hat[0,1] = -omega[2]
    omega_hat[0,2] =  omega[1]
    omega_hat[1,0] =  omega[2]
    omega_hat[1,2] = -omega[0]
    omega_hat[2,0] = -omega[1]
    omega_hat[2,1] =  omega[0]

    if theta < 1e-5:
        # 小角度展开
        R = I + omega_hat + 0.5 * (omega_hat @ omega_hat)
        V = I + 0.5 * omega_hat + (1.0/6.0) * (omega_hat @ omega_hat)
    else:
        A = torch.sin(theta) / theta
        B = (1 - torch.cos(theta)) / (theta**2)
        C = (theta - torch.sin(theta)) / (theta**3)
        R = I + A * omega_hat + B * (omega_hat @ omega_hat)
        V = I + B * omega_hat + C * (omega_hat @ omega_hat)

    t = V @ v.unsqueeze(-1)

    SE3 = torch.eye(4, device=xi.device, dtype=xi.dtype)
    SE3[:3, :3] = R
    SE3[:3, 3:] = t
    return SE3
