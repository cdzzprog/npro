import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 读取原始单通道mask图像
mask_image_path = r'E:\dataset1\bijie\landlside\mask\df002.png'  # 替换为你的mask文件路径
mask_image = Image.open(mask_image_path)

# 将mask图像转换为NumPy数组
mask_array = np.array(mask_image)

# 确保mask图像是二值化的
if np.unique(mask_array).tolist() != [0, 255]:
    print("错误：该图像不是二值图像。")
else:
    # 将mask_array复制到三个通道
    mask_rgb = np.stack([mask_array] * 1, axis=-1)  # 复制到三个通道，生成(512, 512, 3)形状的数组

    # 打印形状以确认
    print(f"转换后的图像形状: {mask_rgb.shape}")  # 应为 (512, 512, 3)

    # 可视化mask_rgb图像
    plt.imshow(mask_rgb)
    plt.title("三通道 mask 图像")
    plt.show()
