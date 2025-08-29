import mcubes
import open3d as o3d

# 测试一下
import numpy as np
X, Y, Z = np.mgrid[:30, :30, :30]
u = (X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2   # 球体

vertices, triangles = mcubes.marching_cubes(u, 0)

print("verts:", vertices.shape, "tris:", triangles.shape)


mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
