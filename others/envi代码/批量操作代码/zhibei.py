import os
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes

# 影像文件夹路径和输出Shapefile文件夹路径
input_folder = r'C:\Users\龙儿璨\Desktop\湿地制图\img1'
output_folder = r'C:\Users\龙儿璨\Desktop\湿地制图\张宇杰\label2'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历影像文件夹中的所有TIF文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.tif'):
        # 构建影像文件的完整路径
        input_file = os.path.join(input_folder, file_name)
        
        # 打开TIF文件并读取波段数据
        with rasterio.open(input_file) as src:
            # 假设绿色波段是第3波段，近红外波段是第8波段
            green = src.read(4).astype('float32')  # 读取绿色波段 (B3)
            nir = src.read(3).astype('float32')    # 读取近红外波段 (B8)
            
            # 计算NDWI
            ndwi = (green - nir) / (green + nir)
            
            # 应用阈值，生成水体掩膜
            water_mask = np.where((ndwi >= 0.14) & (ndwi <= 1), 1, 0)
            
            # 获取栅格图像的原始坐标参考系和仿射变换
            transform = src.transform
            crs = src.crs  # 自动获取图像的CRS
        
        # 提取水体区域的形状
        mask_shapes = shapes(water_mask, transform=transform)
        
        # 将提取的形状转换为矢量格式
        geoms = []
        for geom, value in mask_shapes:
            if value == 1:  # 仅提取水体区域
                geoms.append(shape(geom))
        
        # 创建GeoDataFrame，并添加一个新字段 'label'，所有值都设为2
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
        
        # 新增 'label' 字段，设为2
        gdf['label'] = 2
        
        # # 删除所有字段，只保留 'label' 和 'geometry'
        gdf = gdf[['label', 'geometry']]
        
        # 构建输出Shapefile文件的路径，文件名与原TIF文件名相同但后缀为.shp
        output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + 'label2.shp')
        
        # 将结果保存为Shapefile
        gdf.to_file(output_file)
        
        print(f"Shapefile saved: {output_file}")