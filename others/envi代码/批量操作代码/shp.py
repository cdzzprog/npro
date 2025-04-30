# import os
# import geopandas as gpd
# from shapely.geometry import Polygon

# # 影像文件夹路径和输出Shapefile文件夹路径
# input_folder = r'E:\湿地制图\11.9'
# output_folder = r'E:\湿地制图\11.9\119label2222'

# # 确保输出文件夹存在
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 遍历影像文件夹中的所有TIF文件
# for file_name in os.listdir(input_folder):
#     if file_name.endswith('.tif'):
#         # 构建输出Shapefile文件的路径，文件名与原TIF文件名相同但后缀为.shp
#         output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.shp')
        
#         # 创建一个空的GeoDataFrame
#         # 定义一个空的几何数据列表，使用一个空的Polygon作为占位符（因为Shapefile需要几何数据）
#         empty_geometries = [Polygon()]
        
#         # 创建空的GeoDataFrame，定义CRS为WGS84 (可以根据需要修改CRS)
#         gdf = gpd.GeoDataFrame(geometry=empty_geometries, crs='EPSG:4326')
        
#         # 新增 'label' 字段，设置为2
#         gdf['label'] = 0
        
#         # 删除所有字段，只保留 'label' 和 'geometry'
#         gdf = gdf[['label', 'geometry']]
        
#         # 将空GeoDataFrame保存为Shapefile
#         gdf.to_file(output_file)
        
#         print(f"Empty Shapefile saved: {output_file}")



import os
import geopandas as gpd
from shapely.geometry import Polygon

# 影像文件夹路径和输出Shapefile文件夹路径
input_folder = r'E:\湿地制图\11.9'
output_folder = r'E:\湿地制图\11.9\119label2222'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历影像文件夹中的所有TIF文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.tif'):
        # 构建输出Shapefile文件的路径，文件名与原TIF文件名相同但后缀为.shp
        output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.shp')
        
        # 创建一个空的GeoDataFrame
        # 定义一个空的几何数据列表，使用一个空的Polygon作为占位符（因为Shapefile需要几何数据）
        empty_geometries = [Polygon()]
        
        # 创建空的GeoDataFrame，定义CRS为WGS84 (可以根据需要修改CRS)
        gdf = gpd.GeoDataFrame(geometry=empty_geometries, crs='EPSG:4326')
        
        # 新增 'label' 字段，设置为None
        gdf['label'] = 0 # 创建字段但不赋具体值
        
        # 删除所有字段，只保留 'label' 和 'geometry'
        gdf = gdf[['label', 'geometry']]
        
        # 将空GeoDataFrame保存为Shapefile
        gdf.to_file(output_file)
        
        print(f"Empty Shapefile saved: {output_file}")
