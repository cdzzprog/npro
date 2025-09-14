import os
import geopandas as gpd
import pandas as pd

# 定义输入文件夹路径和输出文件夹路径
input_folder_1 = r'C:\Users\龙儿璨\Desktop\湿地制图\label1'  # 水体文件夹路径
input_folder_2 = r'C:\Users\龙儿璨\Desktop\湿地制图\label2'  # 植被文件夹路径
input_folder_3 = r'C:\Users\龙儿璨\Desktop\湿地制图\label3'  # 额外的文件夹1
input_folder_4 = r'C:\Users\龙儿璨\Desktop\湿地制图\label4'  # 额外的文件夹2
output_folder = r'C:\Users\龙儿璨\Desktop\湿地制图\merge00'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历第一个文件夹中的所有shapefile文件
for file_name in os.listdir(input_folder_1):
    if file_name.endswith('.shp'):
        file_prefix = file_name[:-5]  # 去掉".shp"后缀
        file_prefix0 = file_name[:-10]
        # 构建四个输入文件夹中对应文件的完整路径
        file_1 = os.path.join(input_folder_1, file_name)
        file_2 = os.path.join(input_folder_2, file_prefix + '2.shp')
        file_3 = os.path.join(input_folder_3, file_prefix + '3.shp')
        file_4 = os.path.join(input_folder_4, file_prefix + '4.shp')

        gdfs = []  # 存储读取的 GeoDataFrames

        # 尝试读取文件1
        if os.path.exists(file_1):
            gdf_1 = gpd.read_file(file_1)
            dissolved_gdf_1 = gdf_1.dissolve()
            gdfs.append(dissolved_gdf_1)
        else:
            print(f"File not found: {file_1}")

        # 尝试读取文件2
        if os.path.exists(file_2):
            gdf_2 = gpd.read_file(file_2)
            dissolved_gdf_2 = gdf_2.dissolve()
            gdfs.append(dissolved_gdf_2)
        else:
            print(f"File not found: {file_2}")

        # 尝试读取文件3
        if os.path.exists(file_3):
            gdf_3 = gpd.read_file(file_3)
            dissolved_gdf_3 = gdf_3.dissolve()
            gdfs.append(dissolved_gdf_3)
        else:
            print(f"File not found: {file_3}")

        # 尝试读取文件4
        if os.path.exists(file_4):
            gdf_4 = gpd.read_file(file_4)
            dissolved_gdf_4 = gdf_4.dissolve()
            gdfs.append(dissolved_gdf_4)
        else:
            print(f"File not found: {file_4}")

        # 合并所有已读取的 GeoDataFrames
        if gdfs:
            # 使用 pd.concat 进行合并，确保不会因为空列表而出错
            combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
            
            # 构建输出文件的路径
            output_file = os.path.join(output_folder, file_prefix0 + '_label.shp')

            # 将合并后的矢量文件保存为新的Shapefile
            combined_gdf.to_file(output_file)
            print(f"Combined shapefile saved: {output_file}")
        else:
            print(f"No valid files found for merging: {file_prefix}")
