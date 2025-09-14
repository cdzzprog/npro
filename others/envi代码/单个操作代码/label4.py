import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes

# 打开多波段TIF文件并读取其坐标参考系
with rasterio.open(r'E:\data001\test\insar\insare10.tif') as src:
    
    band1 = src.read(1).astype('float32')  # 读取波段1并转换为float32类型
    
    # 生成掩膜：提取波段1中值在58到67之间的区域
    target_mask = np.where((band1 >= -132) & (band1 <=-10), 1, 0)
    
    # 获取栅格图像的原始坐标参考系和仿射变换
    transform = src.transform
    crs = src.crs  # 自动获取图像的CRS

# 提取目标区域的形状
mask_shapes = shapes(target_mask, transform=transform)

# 将提取的形状转换为矢量格式
geoms = []
for geom, value in mask_shapes:
    if value == 1:  # 仅提取目标区域
        geoms.append(shape(geom))

# 创建GeoDataFrame并保存为矢量文件 (Shapefile)，使用从图像中提取的坐标系
gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)  # 动态获取CRS并应用
gdf['label'] = 0  # 新增字段label并赋值为4
gdf = gdf[['label', 'geometry']]

# 输出为Shapefile
gdf.to_file(r'E:\data001\test\insarDA0.shp')

#AW88  54-78
#AW79  54-70  53-70 40-90
#AW28  54-70  54-90  40-90   40-100  40-110  40-120 40-115
#PW_10 
#PW_61 40-85 40-90
#AW1  50-70  50-75  50-80  50-84
#AW28 50-80  40-90 40-100
#PW_16 50-80
#AW67
#AW96 52-75