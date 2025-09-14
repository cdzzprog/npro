# import os







# data_path = r'E:\dataset1\bijie\landlside'




# datas=os.listdir(data_path)
# img=os.path.join(data_path,'image')
# mask=os.path.join(data_path,'mask')
# dem=os.path.join(data_path,'dem')
# img_list=os.listdir(img)

# img1=img_list[0]
# print(img1)
# def _addNoise(self, img):

#         # return cv2.GaussianBlur(img, (11, 11), 0)
#     return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True) * 255







import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import cv2
import time
import random


# 模拟你的 _addNoise 方法
def _addNoise(img):
    # 添加高斯噪声并将像素值缩放到[0, 255]范围
    noisy_img = random_noise(img, mode='gaussian', seed=int(time.time()), clip=True) * 255
    return noisy_img.astype(np.uint8)


def _changeLight(img):
        alpha = random.uniform(0.35, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)


# 加载图像 (例如使用 OpenCV 或 matplotlib)
img1 = cv2.imread(r'E:\dataset1\bijie\landlside\image\df022.png')  # 使用合适的路径加载图像
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB (如果使用 OpenCV)

# 显示原始图像
plt.subplot(1, 3, 1)
plt.imshow(img1)
plt.title('Original Image')
plt.axis('off')

# 添加噪声
noisy_img = _addNoise(img1)

# 显示添加噪声后的图像
plt.subplot(1, 3, 2)
plt.imshow(noisy_img)
plt.title('Noisy Image')
plt.axis('off')


changeLight=_changeLight(img1)
plt.subplot(1, 3, 3)
plt.imshow(changeLight)
plt.title('changeLight')
plt.axis('off')


# 显示图像
plt.show()








