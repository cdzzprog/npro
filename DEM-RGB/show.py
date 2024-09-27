import matplotlib.pyplot as plt
from dataset import TRAIN_XX, TRAIN_YY
img = 0
fig, axs = plt.subplots(1, 6, figsize=(15, 10))

# 绘制RGB图像
axs[0].set_title("RGB")
axs[0].imshow(TRAIN_XX[img, :, :, 0:3] * 255)  # 乘以255以恢复到0-255范围  # 假设RGB通道分别在0, 1, 2通道
axs[0].axis('off')
axs[1].set_title("elevation")
axs[1].imshow(TRAIN_XX[img, :, :, 3],cmap='gray')  # 假设高程在第3个通道
axs[1].axis('off')
axs[2].set_title(" slope")
axs[2].imshow(TRAIN_XX[img, :, :, 4],cmap='gray')  
axs[2].axis('off')
axs[3].set_title("aspect")
axs[3].imshow(TRAIN_XX[img, :, :, 5],cmap='gray')  
axs[3].axis('off')
axs[4].set_title("curvature")
axs[4].imshow(TRAIN_XX[img, :, :, 6],cmap='gray')  
axs[4].axis('off')
axs[5].set_title("MASK")
axs[5].imshow(TRAIN_YY[img, :, :, 0])  
axs[5].axis('off')


plt.show()