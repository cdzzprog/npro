import rasterio
import numpy as np
import matplotlib.pyplot as plt

def normalize_to_255(band):
    """
    对单个波段数据进行归一化，使得最小值为0，最大值为255
    :param band: 输入的波段数据
    :return: 归一化到[0, 255]范围的波段数据
    """
    band_min = band.min()
    band_max = band.max()
    band_normalized = (band - band_min) / (band_max - band_min) * 255
    return band_normalized.astype(np.uint8)

def sentinel2_to_rgb(tif_path, output_path):
    """
    将哨兵2号的4波段tif文件转为RGB图像
    :param tif_path: 输入的哨兵2号四波段TIF文件路径
    :param output_path: 输出的RGB图像保存路径
    """
    # 打开TIF文件
    with rasterio.open(tif_path) as src:
        # 读取波段
        band4 = src.read(4)  # 红色波段 (Band 4)
        band3 = src.read(3)  # 绿色波段 (Band 3)
        band2 = src.read(2)  # 蓝色波段 (Band 2)

        # 查看波段数值范围，确保数据读取正确
        print(f"Band 4 Min: {band4.min()}, Max: {band4.max()}")
        print(f"Band 3 Min: {band3.min()}, Max: {band3.max()}")
        print(f"Band 2 Min: {band2.min()}, Max: {band2.max()}")

        # 对每个波段进行归一化到[0, 255]范围
        band4_norm = normalize_to_255(band4)  # 红色波段
        band3_norm = normalize_to_255(band3)  # 绿色波段
        band2_norm = normalize_to_255(band2)  # 蓝色波段

        # 创建RGB图像
        rgb = np.stack([band4_norm, band3_norm, band2_norm], axis=-1)  # 顺序为: 红色，蓝色, 绿色, 

    # 使用matplotlib显示并保存图像
    plt.imshow(rgb)
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)

    print(f"RGB图像已保存为: {output_path}")

# 示例用法
tif_path =r'E:\湿地制图\11.9\WetlandXJ_202308_T45_AW_70.tif'  # 替换为实际的文件路径
output_path = 'sentinel2_rgb_image.png'  # 输出的RGB图像路径

sentinel2_to_rgb(tif_path, output_path)
