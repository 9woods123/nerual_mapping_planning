import cv2
import numpy as np
import os
import shutil

def rename_and_show(txt_file, dst_folder, window_name="Image"):
    # 打开文件列表
    with open(txt_file, "r") as f:
        lines = f.readlines()

    # 跳过注释行
    files = [line.strip().split()[1] for line in lines if not line.startswith("#")]

    

    # 创建重命名后的文件夹
    os.makedirs(dst_folder, exist_ok=True)

    for idx, file in enumerate(files):
        # 读取图像
        img = cv2.imread(file)
        if img is None:
            print(f"无法读取 {file}")
            continue

        # 可视化
        cv2.imshow(window_name, img)

        # 构造新文件名
        new_name = f"{idx+1:04d}.png"
        dst_path = os.path.join(dst_folder, new_name)

        # 复制文件到新文件夹
        shutil.copy(file, dst_path)

        key = cv2.waitKey(1)  # 1 ms 等待
        if key == 27:  # ESC 键退出
            break

    cv2.destroyAllWindows()
    print(f"{txt_file} 重命名完成，文件已保存到 {dst_folder}/")

# 对 depth 做处理
rename_and_show("depth.txt", "depth_renamed", window_name="Depth")

# 对 rgb 做处理
rename_and_show("rgb.txt", "rgb_renamed", window_name="RGB")
