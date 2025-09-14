# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# def normalize_to_255(band):
#     """
#     对单个波段数据进行归一化，使得最小值为0，最大值为255
#     :param band: 输入的波段数据
#     :return: 归一化到[0, 255]范围的波段数据
#     """
#     # 替换无效值 -32768 为 NaN（表示缺失值）
#     band = np.where(band ==65535, np.nan, band)
    
#     # 获取最小值和最大值（忽略 NaN）
#     band_min = np.nanmin(band)
#     band_max = np.nanmax(band)
    
#     # 归一化操作
#     band_normalized = (band - band_min) / (band_max - band_min) * 255
#     return band_normalized.astype(np.uint8)

# def sentinel2_to_rgb(tif_path, output_path):
#     """
#     将哨兵2号的4波段tif文件转为RGB图像
#     :param tif_path: 输入的哨兵2号四波段TIF文件路径
#     :param output_path: 输出的RGB图像保存路径
#     """
#     # 打开TIF文件
#     with rasterio.open(tif_path) as src:
#         # 读取波段
#         band4 = src.read(1)  # 红色波段 (Band 4)
#         band3 = src.read(2)  # 绿色波段 (Band 3)
#         band2 = src.read(3)  # 蓝色波段 (Band 2)

#         # 获取图像的尺寸（宽度和高度）
#         width = src.width
#         height = src.height

#         # 查看波段数值范围，确保数据读取正确
#         print(f"Band 4 Min: {band4.min()}, Max: {band4.max()}")
#         print(f"Band 3 Min: {band3.min()}, Max: {band3.max()}")
#         print(f"Band 2 Min: {band2.min()}, Max: {band2.max()}")

#         # 对每个波段进行归一化到[0, 255]范围
#         band4_norm = normalize_to_255(band4)  # 红色波段
#         band3_norm = normalize_to_255(band3)  # 绿色波段
#         band2_norm = normalize_to_255(band2)  # 蓝色波段

#         # 创建RGB图像
#         rgb = np.stack([band4_norm, band3_norm, band2_norm], axis=-1)  # 顺序为: 红色，绿色，蓝色

#     # 使用matplotlib显示并保存图像
#     fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # 设置合适的图像大小
#     ax.imshow(rgb)
#     ax.axis('off')  # 不显示坐标轴
    
#     # 保存图像并检查分辨率
#     plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0,dpi=100)

#     # 读取保存的图像以检查分辨率
#     saved_img = Image.open(output_path)
#     print(f"保存的RGB图像分辨率: {saved_img.size} (宽度, 高度)")
#     print(f"原始图像分辨率: (宽度: {width}, 高度: {height})")

#     print(f"RGB图像已保存为: {output_path}")

# # 示例用法
# tif_path = r'E:\sentinel2\pamieryou\22.10.15\S2B_MSIL2A_20221015T055819_N0400_R091_T43SCC_20221015T083003.SAFE\S2B_MSIL2A_20221015T055819_N0400_R091_T43SCC_20221015T083003.tif'  # 替换为实际的文件路径
# output_path = 'im100.png'  # 输出的RGB图像路径

# sentinel2_to_rgb(tif_path, output_path)
import rasterio
import numpy as np
from PIL import Image

def normalize_to_255(band):
    """
    对单个波段数据进行归一化，使得最小值为0，最大值为255
    :param band: 输入的波段数据
    :return: 归一化到[0, 255]范围的波段数据
    """
    # 替换无效值 -32768 为 NaN（表示缺失值）
    band = np.where(band == 65535, np.nan, band)
    
    # 获取最小值和最大值（忽略 NaN）
    band_min = np.nanmin(band)
    band_max = np.nanmax(band)
    
    # 归一化操作
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
        band4 = src.read(1)  # 红色波段 (Band 4)
        band3 = src.read(2)  # 绿色波段 (Band 3)
        band2 = src.read(3)  # 蓝色波段 (Band 2)

        # 获取图像的尺寸（宽度和高度）
        width = src.width
        height = src.height

        # 查看波段数值范围，确保数据读取正确
        print(f"Band 4 Min: {band4.min()}, Max: {band4.max()}")
        print(f"Band 3 Min: {band3.min()}, Max: {band3.max()}")
        print(f"Band 2 Min: {band2.min()}, Max: {band2.max()}")

        # 对每个波段进行归一化到[0, 255]范围
        band4_norm = normalize_to_255(band4)  # 红色波段
        band3_norm = normalize_to_255(band3)  # 绿色波段
        band2_norm = normalize_to_255(band2)  # 蓝色波段

        # 创建RGB图像
        rgb = np.stack([band4_norm, band3_norm, band2_norm], axis=-1)  # 顺序为: 红色，绿色，蓝色

    # 使用Pillow直接保存图像
    rgb_image = Image.fromarray(rgb)  # 将NumPy数组转换为PIL图像
    rgb_image = rgb_image.convert('RGB')  # 确保图像是RGB格式

    # 保存图像
    rgb_image.save(output_path)

    # 打印输出图像分辨率信息
    print(f"RGB图像已保存为: {output_path}")
    print(f"保存的RGB图像分辨率: {rgb_image.size} (宽度, 高度)")
    print(f"原始图像分辨率: (宽度: {width}, 高度: {height})")

# 示例用法
tif_path = r'E:\sentinel2\pamieryou\2022.7.22\S2A_MSIL2A_20220722T055651_N0400_R091_T43SCC_20220722T092103.SAFE\S2A_MSIL2A_20220722T055651_N0400_R091_T43SCC_20220722T092103.tif'  # 替换为实际的文件路径
output_path = 'im101.png'  # 输出的RGB图像路径

sentinel2_to_rgb(tif_path, output_path)
