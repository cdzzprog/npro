import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from shapely.geometry import mapping

def shapefile_to_segmentation_label(shapefile_path, raster_template_path, output_path, class_value=1):
    """
    将shapefile标签转换为深度学习语义分割标签图像
    :param shapefile_path: 输入的shapefile路径
    :param raster_template_path: 栅格图像模板路径，提供空间分辨率和尺寸
    :param output_path: 输出的标签图像路径
    :param class_value: 类别值 (通常是标签的ID)
    """
    
    # 读取shapefile
    gdf = gpd.read_file(shapefile_path)

    # 读取栅格模板（可以是一个标签图像的参考）
    with rasterio.open(raster_template_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    # 初始化标签图像，所有值为0（背景为黑色）
    label_img = np.zeros(out_shape, dtype=np.uint8)
    
    # 遍历shapefile中的每个几何图形，转换为标签图像
    for geom in gdf.geometry:
        # 创建一个掩膜，选择当前几何形状覆盖的像素
        mask = geometry_mask([geom], transform=transform, invert=True, out_shape=out_shape)
        # 将掩膜区域设置为指定的类别值
        label_img[mask] = class_value
    
    # 保存标签图像（黑白图像）
    plt.imsave(output_path, label_img, cmap='gray')  # 使用gray colormap保存为黑白图像

    print(f"标签图像保存为: {output_path}")

# 示例用法
shapefile_path = r'E:\湿地制图\11.9\11.9result\WetlandXJ_202308_T45_AW_70_label.shp'
raster_template_path = r'E:\湿地制图\11.9\WetlandXJ_202308_T45_AW_70.tif'  # 栅格图像模板
output_path = 'output_label_image.png'

shapefile_to_segmentation_label(shapefile_path, raster_template_path, output_path, class_value=1)
