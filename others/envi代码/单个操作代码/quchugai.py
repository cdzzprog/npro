import geopandas as gpd
from shapely.geometry import box

# 加载Shapefile
shapefile_path = r'C:\Users\龙儿璨\Desktop\湿地制图\RESULT_final\PW61\label4.shp'
gdf = gpd.read_file(shapefile_path)

# 方法1: 直接删除特定索引
gdf_filtered = gdf.drop(index=[0])  # 假设要删除第一个像元

# 方法2: 根据属性值过滤
# gdf_filtered = gdf[gdf['label'] != '要删除的值']

# 方法3: 删除特定区域内的像元
# 定义要删除的区域
x_min, y_min = 100, 100  # 设置坐标
gdf_filtered = gdf[~gdf.geometry.intersects(box(x_min, y_min, x_min + 10, y_min + 10))]

# 保存处理后的结果
gdf_filtered.to_file(r'C:\Users\龙儿璨\Desktop\湿地制图\RESULT_final\PW16\WetlandXJ_202308_T43_PW_16_label4_filtered.shp')

