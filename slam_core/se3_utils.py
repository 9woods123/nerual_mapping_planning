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


def SE3_to_se3(SE3_mat):
    """
    Convert SE(3) matrix to se(3) vector (6D twist)
    Input:
        SE3_mat: (..., 4, 4) homogeneous transform matrix
    Output:
        se3_vec: (..., 6) [ωx, ωy, ωz, tx, ty, tz]
    """
    # 旋转和平移分离
    R = SE3_mat[..., :3, :3]
    t = SE3_mat[..., :3, 3]

    # 计算旋转部分的 log(R)
    cos_theta = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)

    # 防止除零
    small_angle = theta.abs() < 1e-5

    # 反对称矩阵提取
    lnR = (R - R.transpose(-1, -2)) / (2 * torch.sin(theta)[..., None, None] + 1e-8)
    w = torch.stack([lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]], dim=-1) * theta[..., None]

    # 小角度近似
    w[small_angle] = 0.5 * torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1)[small_angle]

    # 计算 V⁻¹ * t
    wx = skew_matrix(w)
    theta_expand = theta[..., None, None]
    I = torch.eye(3, device=R.device, dtype=R.dtype)
    A = (1 - torch.cos(theta_expand)) / (theta_expand ** 2 + 1e-8)
    B = (theta_expand - torch.sin(theta_expand)) / (theta_expand ** 3 + 1e-8)
    Vinv = I - 0.5 * wx + (1 / theta_expand**2 - (1 + torch.cos(theta_expand)) / (2 * theta_expand * torch.sin(theta_expand) + 1e-8)) * wx @ wx
    v = (Vinv @ t.unsqueeze(-1)).squeeze(-1)

    # 拼接
    se3_vec = torch.cat([w, v], dim=-1)
    return se3_vec

def skew_matrix(w):
    """
    Convert 3D vector to skew-symmetric matrix
    Input: (..., 3)
    Output: (..., 3, 3)
    """
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)
    skew = torch.stack([
        torch.stack([O, -wz, wy], dim=-1),
        torch.stack([wz, O, -wx], dim=-1),
        torch.stack([-wy, wx, O], dim=-1)
    ], dim=-2)
    return skew
