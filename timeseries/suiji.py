import gdal
import random
import geopandas as gpd
from shapely.geometry import Point

# 读取TIFF文件
tif_path = 'your_image.tif'  # 你的TIFF影像文件路径
dataset = gdal.Open(tif_path)

# 获取影像的地理变换（坐标系信息）
geo_transform = dataset.GetGeoTransform()

# 获取影像的行数和列数
cols = dataset.RasterXSize
rows = dataset.RasterYSize

# 获取影像的空间范围（像素坐标 -> 地理坐标）
xmin = geo_transform[0]
ymax = geo_transform[3]
xmax = xmin + geo_transform[1] * cols
ymin = ymax + geo_transform[5] * rows

# 设置生成随机点的数量
num_points = 100

# 生成随机点
random_points = []
for _ in range(num_points):
    # 生成随机的像素坐标
    col = random.randint(0, cols - 1)
    row = random.randint(0, rows - 1)

    # 将像素坐标转换为地理坐标
    x = xmin + col * geo_transform[1]
    y = ymax + row * geo_transform[5]

    # 将点添加到列表中
    random_points.append(Point(x, y))

# 创建GeoDataFrame
gdf = gpd.GeoDataFrame(geometry=random_points)

# 设置CRS为TIFF影像的坐标参考系（这里使用EPSG:4326作为示例，实际情况可能需要调整）
gdf.set_crs('EPSG:4326', inplace=True)

# 保存为Shapefile
gdf.to_file('random_points_from_tif.shp')

print(f"成功生成了{num_points}个随机点，并保存为'random_points_from_tif.shp'文件。")
