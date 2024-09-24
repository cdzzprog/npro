import numpy as np
import imageio
import glob
from skimage.transform import resize
from numpy import gradient

TRAIN_PATH = r"E:\数据集\山体滑坡数据集\landslide\image\*.png"
TRAIN_MASK = r"E:\数据集\山体滑坡数据集\landslide\mask\*.png"
TRAIN_DEM = r"E:\\数据集\\山体滑坡数据集\\landslide\\dem\\*.png"
TRAIN_XX = np.zeros((770, 7, 256, 256))  # 更新为7个特征
TRAIN_YY = np.zeros((770, 1, 256, 256))

all_train = sorted(glob.glob(TRAIN_PATH))
all_dem = sorted(glob.glob(TRAIN_DEM))
all_mask = sorted(glob.glob(TRAIN_MASK))

# 处理图像和掩码
for i, (img_path, mask_path) in enumerate(zip(all_train, all_mask)):
    # 读取RGB图像并调整大小
    data = imageio.imread(img_path)
    data = resize(data, (256, 256), anti_aliasing=True)
    data[np.isnan(data)] = 0.000001
    
    # 提取RGB特征并归一化
    TRAIN_XX[i, 0,:, :] = data[:, :, 0] / 255.0  # RED
    TRAIN_XX[i, 1,:, :] = data[:, :, 1] / 255.0  # GREEN
    TRAIN_XX[i, 2,:, :] = data[:, :, 2] / 255.0  # BLUE

    # 读取掩码并调整大小
    mask_data = imageio.imread(mask_path)
    mask_data = resize(mask_data, (256, 256), anti_aliasing=True)
    TRAIN_YY[i, 0,:, :] = mask_data

# 处理DEM数据
for i, dem_file in enumerate(all_dem):
    dem_data = imageio.imread(dem_file)  # 读取DEM图像
    dem_data = resize(dem_data, (256, 256), anti_aliasing=True)  # 调整为256x256
    
    # 计算高程
    elevation = dem_data.astype(float)  # 高程数据
    TRAIN_XX[i,3, :, :] = elevation / np.max(elevation)  # 高程归一化

    # 计算坡度
    gradient_x, gradient_y = np.gradient(elevation)  # 计算梯度
    slope = np.sqrt(gradient_x**2 + gradient_y**2)  # 计算坡度
    TRAIN_XX[i, 4,:, :] = slope / np.max(slope)  # SLOPE 归一化
    
    # 计算坡向
    aspect = np.arctan2(gradient_y, gradient_x)  # 计算坡向（弧度）
    aspect = np.degrees(aspect)  # 转换为度
    aspect[aspect < 0] += 360  # 将负值转为正值
    TRAIN_XX[i, 5,:, :] = aspect / 360.0  # ASPECT 归一化到[0, 1]

    # 计算曲率
    gradient_xx, gradient_xy = np.gradient(gradient_x)
    gradient_yx, gradient_yy = np.gradient(gradient_y)
    curvature = gradient_xx + gradient_yy  # 简单曲率计算
    TRAIN_XX[i,6, :, :] = curvature / np.max(curvature)  # CURVATURE 归一化

# 处理缺失值
TRAIN_XX[np.isnan(TRAIN_XX)] = 0.000001 
TRAIN_XX = TRAIN_XX[:1500]
TRAIN_YY = TRAIN_YY[:1500]




# print(TRAIN_XX.shape)