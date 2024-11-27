import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes

# 打开多波段TIF文件并读取其坐标参考系
with rasterio.open(r'E:\BaiduNetdiskDownload\初始数据\original\luding_after_g6_image.tif') as src:
    
    green = src.read(2).astype('float32')  
    nir = src.read(4).astype('float32')    
    
    # 计算NDWI
    ndwi = (green - nir) / (green + nir)
    
    # 应用阈值（0到1），生成水体掩膜
    water_mask = np.where((ndwi >= -0.1) & (ndwi <= 1), 1, 0)
    
    # 获取栅格图像的原始坐标参考系和仿射变换
    transform = src.transform
    crs = src.crs  # 自动获取图像的CRS
    print(crs)
提取水体区域的形状
mask_shapes = shapes(water_mask, transform=transform)

# 将提取的形状转换为矢量格式
geoms = []
for geom, value in mask_shapes:
    if value == 1:  # 仅提取水体区域
        geoms.append(shape(geom))

# 创建GeoDataFrame并保存为矢量文件 (Shapefile)，使用从图像中提取的坐标系
gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)  # 动态获取CRS并应用
gdf['label'] = 1  # 新增字段label并赋值为1
gdf = gdf[['label', 'geometry']]
# 输出为Shapefile
gdf.to_file(r'C:\Users\龙儿璨\Desktop\湿地制图\11.9\119label\WetlandXJ_202308_T45_AW_83label111.shp')