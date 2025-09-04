
import os
import numpy as np
import trimesh
import open3d as o3d
import torch
import mcubes

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


# def visualize_point_cloud(points_tensor, sdf_tensors=None, color_tensors=None, truncation=0.1):
#     """
#     可视化点云，可选 SDF 筛选和颜色输入

#     :param points_tensor: torch.Tensor 或 np.array, (N_rays, N_samples, 3) 或 (N_points, 3)
#     :param sdf_tensors: torch.Tensor 或 np.array, 与 points_tensor 对应，可选
#     :param color_tensors: None 或 (N_points,3) 或 (N_rays,N_samples,3)
#     :param truncation: float, sdf 截断阈值，只显示 |sdf| < truncation 的点
#     """


#     with torch.no_grad():
#         # 转 numpy



#         if isinstance(points_tensor, torch.Tensor):
#             points = points_tensor.reshape(-1, 3).cpu().numpy()
#         else:
#             points = np.array(points_tensor).reshape(-1, 3)

#         if sdf_tensors is not None:
#             if isinstance(sdf_tensors, torch.Tensor):
#                 sdf = sdf_tensors.reshape(-1).cpu().numpy()
#             else:
#                 sdf = np.array(sdf_tensors).reshape(-1)

#             mask = (np.abs(sdf)*truncation <0.01)

#             # ---- 函数刚进来就打印 ----
#             print("================================[Raw Point Cloud Debug]================================")
#             print("Total points (before mask):", len(points))
#             print("Sample points (first 20):")
#             for i in range(min(20, len(points))):
#                 print(f"Point {i}: {points[i]}, SDF: {sdf[i]}")
#             print("=======================================================================================")

#             points = points[mask]
#             sdf = sdf[mask]

#             # 如果有 color，也要筛选
#             if color_tensors is not None:
#                 if isinstance(color_tensors, torch.Tensor):
#                     colors = color_tensors.reshape(-1,3).detach().cpu().numpy()
#                 else:
#                     colors = np.array(color_tensors).reshape(-1,3)
#                 colors = colors[mask]
#             else:
#                 colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (points.shape[0], 1))  # 默认红色

#             # 打印调试信息
#             print("================================[Point Cloud Debug]================================")
#             print("Total points after mask:", len(points))
#             print("Sample points (first 20):")
#             for i in range(min(20, len(points))):
#                 print(f"Point {i}: {points[i]}, SDF: {sdf[i]}, Color: {colors[i]}")
#             print("=====================================================================================")

#             # 构建 Open3D 点云
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(points)
#             pcd.colors = o3d.utility.Vector3dVector(colors)

#             # 可视化
#             o3d.visualization.draw_geometries([pcd])


def visualize_point_cloud(points_tensor, sdf_tensors=None, color_tensors=None, 
                          truncation=0.1, use_mcubes=False, resolution=64):
    """
    可视化点云，可选 SDF 筛选 + Marching Cubes 网格化

    :param points_tensor: torch.Tensor 或 np.array, (N_points, 3)
    :param sdf_tensors: torch.Tensor 或 np.array, 与 points_tensor 对应，可选
    :param color_tensors: torch.Tensor 或 np.array, (N_points,3)，可选
    :param truncation: float, sdf 截断阈值，只显示 |sdf| < truncation 的点
    :param use_mcubes: bool, 是否用 Marching Cubes 重建表面
    :param resolution: int, 网格分辨率 (mcubes 用)
    """

    with torch.no_grad():
        # ---- 转 numpy ----
        if isinstance(points_tensor, torch.Tensor):
            points = points_tensor.reshape(-1, 3).cpu().numpy()
        else:
            points = np.array(points_tensor).reshape(-1, 3)

        if sdf_tensors is not None:
            if isinstance(sdf_tensors, torch.Tensor):
                sdf = sdf_tensors.reshape(-1).cpu().numpy()
            else:
                sdf = np.array(sdf_tensors).reshape(-1)

            # ---- Debug 打印 ----
            print("================================[Raw Point Cloud Debug]================================")
            print("Total points (before mask):", len(points))
            for i in range(min(20, len(points))):
                print(f"Point {i}: {points[i]}, SDF: {sdf[i]}")
            print("=======================================================================================")

            # ---- 点云筛选 (可选) ----
            mask = np.abs(sdf) < 0.5
            points = points[mask]
            sdf = sdf[mask]

            if color_tensors is not None:
                if isinstance(color_tensors, torch.Tensor):
                    colors = color_tensors.reshape(-1, 3).detach().cpu().numpy()
                else:
                    colors = np.array(color_tensors).reshape(-1, 3)
                colors = colors[mask]
            else:
                colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (points.shape[0], 1))

            print("================================[Point Cloud Debug]================================")
            print("Total points after mask:", len(points))
            for i in range(min(20, len(points))):
                print(f"Point {i}: {points[i]}, SDF: {sdf[i]}, Color: {colors[i]}")
            print("=====================================================================================")

            geometries = []

            # ---- 点云可视化 ----
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(pcd)

            # ---- Marching Cubes 重建网格 ----
            if use_mcubes:
                # 体素化 (插值到规则网格)
                xmin, ymin, zmin = points.min(axis=0)
                xmax, ymax, zmax = points.max(axis=0)

                X, Y, Z = np.meshgrid(
                    np.linspace(xmin, xmax, resolution),
                    np.linspace(ymin, ymax, resolution),
                    np.linspace(zmin, zmax, resolution),
                    indexing="ij"
                )
                grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

                # 最近邻插值 SDF
                from scipy.spatial import cKDTree
                tree = cKDTree(points)
                dist, idx = tree.query(grid_points, k=1)
                grid_sdf = sdf[idx].reshape(resolution, resolution, resolution)

                # Marching Cubes
                print("Running Marching Cubes ...")
                vertices, triangles = mcubes.marching_cubes(grid_sdf, 0.0)

                # 映射回坐标系
                scale = np.array([xmax-xmin, ymax-ymin, zmax-zmin]) / resolution
                vertices = vertices * scale + np.array([xmin, ymin, zmin])

                # 转 Open3D
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(triangles)
                mesh.compute_vertex_normals()
                geometries.append(mesh)

            # ---- 显示 ----
            o3d.visualization.draw_geometries(geometries)


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

    pts_tensor = torch.from_numpy(grid).float().to(device)
    pts_tensor=pts_tensor/10.0
    for start in range(0, pts_tensor.shape[0], batch_size):
        end = start + batch_size
        sdf_batch, color_batch = query_fn(pts_tensor[start:end])
        sdf_values.append(sdf_batch.cpu())
        color_values.append(color_batch.cpu())

    sdf_values = torch.cat(sdf_values, dim=0).numpy()
    color_values = torch.cat(color_values, dim=0).numpy()

    print(f"[Shape] sdf_values: {sdf_values.shape}")      # (N,) or (N,1)
    print(f"[Shape] color_values: {color_values.shape}")  # (N,3)

    # 筛选表面点
    mask =  (np.abs(sdf_values)< 0.1)
    

    print(f"[Shape] mask: {mask.shape}")  # 应该是 (N,)

    # 保证 mask 是一维的
    mask = mask.squeeze()

    surface_points = grid[mask]
    surface_colors = color_values[mask]

    print(f"[Shape] surface_points: {surface_points.shape}")  
    print(f"[Shape] surface_colors: {surface_colors.shape}")  

    # 可视化
    # 可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_points)
    pcd.colors = o3d.utility.Vector3dVector(surface_colors)

    # 添加原点坐标系 (size=1.0 可以调整大小)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries([pcd, coord_frame])


    # 保存
    if save_path is not None:
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"[Saved] Surface point cloud saved to {save_path}")

    return surface_points


# if __name__ == "__main__":
#     # 测试用例
#     print("visual.py loaded. 请调用 visualize_mesh_ply / visualize_mesh_open3d / visualize_point_cloud")
