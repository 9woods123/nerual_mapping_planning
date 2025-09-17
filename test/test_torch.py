import torch

# 模拟输入
surface_depths_tensor = torch.arange(5).reshape(5, 1)   # [5,1], 值是 [[0],[1],[2],[3],[4]]
observe_depth = torch.tensor([
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [1.0, 1.1, 1.2, 1.3, 1.4],
    [2.0, 2.1, 2.2, 2.3, 2.4],
    [3.0, 3.1, 3.2, 3.3, 3.4],
    [4.0, 4.1, 4.2, 4.3, 4.4]
])  # [5,5]

print("surface_depths_tensor:\n", surface_depths_tensor)
print("observe_depth:\n", observe_depth)

# expand 把 [5,1] 广播成 [5,5]
expanded_surface = surface_depths_tensor.expand_as(observe_depth)

print("\nexpanded_surface:\n", expanded_surface)

# 做差值
diff = observe_depth - expanded_surface

print("\ndiff:\n", diff)



A = torch.randn(10, 3, 4)   # 10个 (3x4) 矩阵
B = torch.randn(10, 4, 5)   # 10个 (4x5) 矩阵

C = torch.bmm(A, B)         # 结果是 10个 (3x5) 矩阵
print(C.shape)              # torch.Size([10, 3, 5])