import os
import geopandas as gpd
import pandas as pd

# 定义输入文件夹路径和输出文件夹路径
input_folder_1 = r'C:\Users\龙儿璨\Desktop\湿地制图\label1'  # 替换为你的输入水体文件夹路径
input_folder_2 = r'C:\Users\龙儿璨\Desktop\湿地制图\label2'  # 替换为你的输入植被文件夹路径
output_folder = r'C:\Users\龙儿璨\Desktop\湿地制图\merge'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历第一个文件夹中的所有shapefile文件
for file_name in os.listdir(input_folder_1):
    if file_name.endswith('.shp'):
        # 获取去掉最后一个字符的文件名前缀
        file_prefix = file_name[:-5]  # 去掉最后一个字符及".shp"后缀
        # 构建两个输入文件夹中对应文件的完整路径
        file_1 = os.path.join(input_folder_1, file_name)
        file_2 = os.path.join(input_folder_2, file_prefix + '2.shp')  # 假设第二个文件的后缀是2
        
        # 确保第二个文件夹中存在对应的矢量文件
        if os.path.exists(file_2):
            # 读取第一个文件夹中的矢量文件
            gdf_1 = gpd.read_file(file_1)
            # 读取第二个文件夹中的矢量文件
            gdf_2 = gpd.read_file(file_2)
            
            # 先对每个文件内的要素进行合并，合并为一个几何
            dissolved_gdf_1 = gdf_1.dissolve()
            dissolved_gdf_2 = gdf_2.dissolve()
            
            # 将两个文件的合并后的结果再次合并，并保留所有属性字段
            combined_gdf = gpd.GeoDataFrame(pd.concat([dissolved_gdf_1, dissolved_gdf_2], ignore_index=True), crs=gdf_1.crs)
            
            # 构建输出文件的路径
            output_file = os.path.join(output_folder, file_prefix + '12.shp')
            
            # 将合并后的矢量文件保存为新的Shapefile
            combined_gdf.to_file(output_file)
            
            print(f"Combined shapefile saved with dissolved geometries and attributes: {output_file}")
        else:
            print(f"Corresponding file not found in folder 2 for: {file_name}")
