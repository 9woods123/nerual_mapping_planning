import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_camera_trajectory(c2w_all, save_path="trajectory.png", title="Camera Trajectory", axis_length=0.05):
    """
    根据 c2w_all 绘制相机轨迹并显示相机朝向
    :param c2w_all: torch.Tensor or np.ndarray, shape (N, 4, 4)，相机位姿矩阵
    :param save_path: str, 保存 PNG 路径
    :param title: str, 图像标题
    :param axis_length: float, 每个相机坐标系的箭头长度
    """
    # 转 numpy
    if hasattr(c2w_all, "detach"):  # torch.Tensor
        c2w_all = c2w_all.detach().cpu().numpy()

    poses = []
    for c2w in c2w_all:
        cam_pos = c2w[:3, 3]
        rot = c2w[:3, :3]
        poses.append((cam_pos, rot))

    if len(poses) == 0:
        print("⚠️ No poses to plot.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 相机位置轨迹
    cam_positions = np.array([p[0] for p in poses])
    ax.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 'b-', label="Trajectory")
    ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], c='r', s=15, label="Keyframes")

    # 每个相机画坐标轴
    for pos, rot in poses:
        x_axis = rot[:, 0] * axis_length
        y_axis = rot[:, 1] * axis_length
        z_axis = rot[:, 2] * axis_length
        ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], color='r', linewidth=1)
        ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], color='g', linewidth=1)
        ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], color='b', linewidth=1)

    # 美化
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(title)
    ax.view_init(elev=30, azim=-60)  # 视角

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()  # 直接弹出窗口
    plt.close(fig)
    print(f"✅ 轨迹保存到 {save_path}")


if __name__ == "__main__":
    # 手动初始化你给的 c2w_all
    c2w_all = torch.tensor([
        [[ 0.9973, -0.0740,  0.0020, -0.0468],
         [ 0.0740,  0.9964, -0.0407,  0.0396],
         [ 0.0011,  0.0408,  0.9992,  0.0367],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[ 0.9981, -0.0616,  0.0099, -0.0295],
         [ 0.0596,  0.9887,  0.1374,  0.0207],
         [-0.0182, -0.1366,  0.9905,  0.0590],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[ 0.9972, -0.0453,  0.0592, -0.0053],
         [ 0.0321,  0.9778,  0.2070, -0.0177],
         [-0.0673, -0.2045,  0.9765,  0.1011],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[ 0.9910,  0.0261,  0.1311,  0.0056],
         [-0.0609,  0.9614,  0.2685,  0.0676],
         [-0.1190, -0.2741,  0.9543, -0.1163],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[ 0.9778, -0.1202,  0.1716,  0.0941],
         [ 0.0685,  0.9575,  0.2802,  0.0327],
         [-0.1979, -0.2622,  0.9445,  0.0437],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[ 0.9749, -0.1451,  0.1686,  0.0258],
         [ 0.1037,  0.9670,  0.2327,  0.0307],
         [-0.1968, -0.2094,  0.9578,  0.1228],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[ 0.9784, -0.0214,  0.2054, -0.0044],
         [-0.0182,  0.9818,  0.1889,  0.0087],
         [-0.2057, -0.1886,  0.9603,  0.1584],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[ 0.9691,  0.0297,  0.2448, -0.0661],
         [-0.0758,  0.9806,  0.1809,  0.0322],
         [-0.2346, -0.1939,  0.9525,  0.1470],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],
    ], device="cpu")

    plot_camera_trajectory(c2w_all, save_path="output/camera_traj.png")
