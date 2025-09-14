import geopandas as gpd
from shapely.geometry import MultiPolygon




# 读取shp文件
gdf = gpd.read_file(r'C:\Users\龙儿璨\Desktop\湿地制图\label1\WetlandXJ_202308_T43_PW_10label1.shp')

# 合并几何体，填补空洞
gdf['geometry'] = gdf['geometry'].buffer(0)  # 修复几何
combined_geometry = gdf.geometry.union_all()  # 使用 union_all 方法合并几何

# 创建新的 GeoDataFrame
fixed_gdf = gpd.GeoDataFrame(geometry=[combined_geometry], crs=gdf.crs)

# 保存修复后的shp文件
fixed_gdf.to_file(r'C:\Users\龙儿璨\Desktop\湿地制图\xiubu\fixed_water_bodies1.shp')
