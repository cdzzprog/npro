# import geopandas as gpd

# # 加载Shapefile
# shapefile_path = r'C:\Users\龙儿璨\Desktop\湿地制图\label4\WetlandXJ_202308_T44_AW_16label4.shp'  # 替换为你的Shapefile路径
# gdf = gpd.read_file(shapefile_path)

# # 假设最小单元的条件是某个特定的几何形状，比如面积最小的几何
# # 计算每个几何的面积
# gdf['area'] = gdf.geometry.area

# # 找到最小面积的单元
# min_area_index = gdf['area'].idxmin()

# # 删除最小单元
# gdf_filtered = gdf.drop(index=min_area_index)

# # 可选：删除临时计算的面积列
# gdf_filtered = gdf_filtered.drop(columns='area')

# # 保存处理后的结果
# gdf_filtered.to_file(r'C:\Users\龙儿璨\Desktop\湿地制图\label4\WetlandXJ_202308_T44_AW_16label44.shp')

# print("处理完成，已保存为 filtered_file.shp")


import geopandas as gpd

# 加载Shapefile
shapefile_path = r'C:\Users\龙儿璨\Desktop\湿地制图\label\WetlandXJ_202308_T43_PW_34label11.shp'
gdf = gpd.read_file(shapefile_path)

# 计算每个几何的面积
gdf['area'] = gdf.geometry.area

# 找到最小面积
min_area = gdf['area'].min()
print(min_area)
# 删除所有面积等于最小面积的单元
gdf_filtered = gdf[gdf['area'] > 100000*min_area]


gdf_filtered = gdf_filtered.drop(columns='area')

# 保存处理后的结果
gdf_filtered.to_file(r'C:\Users\龙儿璨\Desktop\湿地制图\label\WetlandXJ_202308_T43_PW_34label11quxiaokuan.shp')


