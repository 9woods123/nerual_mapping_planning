import torch

# 假设 delta_se3 是要优化的变量
delta = torch.nn.Parameter(torch.ones(1, requires_grad=True))

# 假设 keyframe 的 c2w（其实就是某个 pose），不是 optimizer 的参数
kf_c2w = torch.tensor([2.0], requires_grad=True)

# Optimizer 只管 delta
opt = torch.optim.SGD([delta], lr=0.1)

for step in range(111):
    opt.zero_grad()
    
    # --------------------------
    # 两种情况对比
    # --------------------------
    pose1 = delta * kf_c2w           # 不 detach
    
    loss1 = (pose1 - 10).pow(2)   # 假装 loss


    # 分别 backward
    loss1.backward()  
    opt.step()

    print("kf_c2w:",kf_c2w)