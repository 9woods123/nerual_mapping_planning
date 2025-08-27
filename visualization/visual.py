
import os
import numpy as np
import trimesh
import open3d as o3d
import torch


def visualize_mesh_ply(mesh_path):
    """
    可视化 PLY 网格文件
    :param mesh_path: PLY 文件路径
    """
    if not os.path.exists(mesh_path):
        print(f"[Error] Mesh file does not exist: {mesh_path}")
        return

    mesh = trimesh.load(mesh_path)
    print(f"[Mesh] vertices: {mesh.vertices.shape}, faces: {mesh.faces.shape}")

    # 如果有颜色，会自动显示
    mesh.show()

def visualize_mesh_open3d(mesh_path):
    """
    用 Open3D 可视化 PLY 网格文件
    :param mesh_path: PLY 文件路径
    """
    if not os.path.exists(mesh_path):
        print(f"[Error] Mesh file does not exist: {mesh_path}")
        return

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    print(mesh)

    o3d.visualization.draw_geometries([mesh])

def visualize_point_cloud(points_tensor, colors_tensor=None):
    """
    显示点云
    :param points_tensor: torch.Tensor, (N_rays, N_samples, 3) 或 (N_points, 3)
    :param colors_tensor: torch.Tensor, (N_rays, N_samples, 3) 或 (N_points, 3), 范围 0~1
    """

    with torch.no_grad():

        if isinstance(points_tensor, torch.Tensor):
            points = points_tensor.reshape(-1, 3).cpu().numpy()
        else:
            points = np.array(points_tensor).reshape(-1, 3)



        if colors_tensor is not None:
            if isinstance(colors_tensor, torch.Tensor):
                colors = colors_tensor.reshape(-1, 3).cpu().numpy()
            else:
                colors = np.array(colors_tensor).reshape(-1, 3)
            colors = np.clip(colors, 0, 1)      # 保证颜色在 0~1


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)  # 手动设置颜色

        o3d.visualization.draw_geometries([pcd])

def visualize_point_cloud_red(points_tensor, sdf_tensors=None, truncation=0.1):
    """
    :param points_tensor: torch.Tensor, (N_rays, N_samples, 3) 或 (N_points, 3)
    :param sdf_tensors: torch.Tensor 或 np.array, 与 points_tensor 对应
    :param truncation: float, sdf 截断阈值
    """
    with torch.no_grad():
        # 转 numpy
        if isinstance(points_tensor, torch.Tensor):
            points = points_tensor.reshape(-1, 3).cpu().numpy()
        else:
            points = np.array(points_tensor).reshape(-1, 3)

        # 处理 sdf 并筛选
        if sdf_tensors is not None:
            if isinstance(sdf_tensors, torch.Tensor):
                sdf = sdf_tensors.reshape(-1).cpu().numpy()
            else:
                sdf = np.array(sdf_tensors).reshape(-1)

            mask = sdf <= truncation
            points = points[mask]
            sdf = sdf[mask]

            # 打印前几个 point 和 sdf
            print("前几个点和对应的 SDF 值:")
            for i in range(min(20, len(points))):
                print(f"Point {i}: {points[i]}, SDF: {sdf[i]}")

        # 全部红色
        colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (points.shape[0], 1))

        # 构建 Open3D 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 可视化
        o3d.visualization.draw_geometries([pcd])



def save_point_cloud(points_tensor, save_path):
    """
    保存点云到 PLY
    """
    if isinstance(points_tensor, torch.Tensor):
        points = points_tensor.reshape(-1, 3).cpu().numpy()
    else:
        points = np.array(points_tensor).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"[Saved] Point cloud saved to {save_path}")


@torch.no_grad()
def visualize_global_surface(query_fn, bounding_box, voxel_size=0.05, truncation=0.1, device='cuda', save_path=None):
    """
    查询全局点云，只显示表面点，并可视化
    """

    x_min, y_min, z_min = bounding_box[0]
    x_max, y_max, z_max = bounding_box[1]

    # 生成体素网格
    xs = np.arange(x_min, x_max, voxel_size)
    ys = np.arange(y_min, y_max, voxel_size)
    zs = np.arange(z_min, z_max, voxel_size)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1).reshape(-1, 3)

    print(f"[Shape] grid: {grid.shape}")  # (N, 3)

    # 查询网络
    batch_size = 1024*64
    sdf_values = []
    color_values = []
    for i in range(0, grid.shape[0], batch_size):
        pts = torch.from_numpy(grid[i:i+batch_size]).float().to(device)
        sdf_batch, color_batch = query_fn(pts)
        sdf_values.append(sdf_batch.cpu())
        color_values.append(color_batch.cpu())

    sdf_values = torch.cat(sdf_values, dim=0).numpy()
    color_values = torch.cat(color_values, dim=0).numpy()

    print(f"[Shape] sdf_values: {sdf_values.shape}")      # (N,) or (N,1)
    print(f"[Shape] color_values: {color_values.shape}")  # (N,3)

    # 筛选表面点
    mask = (sdf_values > 0) & (sdf_values < truncation)

    print(f"[Shape] mask: {mask.shape}")  # 应该是 (N,)

    # 保证 mask 是一维的
    mask = mask.squeeze()

    surface_points = grid[mask]
    surface_colors = color_values[mask]

    print(f"[Shape] surface_points: {surface_points.shape}")  
    print(f"[Shape] surface_colors: {surface_colors.shape}")  

    # 可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_points)
    pcd.colors = o3d.utility.Vector3dVector(surface_colors)
    o3d.visualization.draw_geometries([pcd])

    # 保存
    if save_path is not None:
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"[Saved] Surface point cloud saved to {save_path}")

    return surface_points


# if __name__ == "__main__":
#     # 测试用例
#     print("visual.py loaded. 请调用 visualize_mesh_ply / visualize_mesh_open3d / visualize_point_cloud")
