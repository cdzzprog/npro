import os
from osgeo import ogr, gdal
from pathlib import Path

def extract_ndvi_to_points(vector_path, raster_path, field_name='NDVI'):
    # 打开矢量数据
    vector_ds = ogr.Open(vector_path, 1)  # 1表示可写
    vector_layer = vector_ds.GetLayer()
    
    # 检查是否存在NDVI字段，不存在则创建
    layer_defn = vector_layer.GetLayerDefn()
    field_exists = vector_layer.GetFieldIndex(field_name) != -1
    
    if not field_exists:
        field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
        vector_layer.CreateField(field_defn)
    
    # 打开栅格数据
    raster_ds = gdal.Open(raster_path)
    gt = raster_ds.GetGeoTransform()
    band = raster_ds.GetRasterBand(1)
    
    # 遍历所有点要素
    for feature in vector_layer:
        geom = feature.GetGeometryRef()
        x, y = geom.GetX(), geom.GetY()
        
        # 计算栅格行列号
        px = int((x - gt[0]) / gt[1])
        py = int((y - gt[3]) / gt[5])
        
        # 检查是否越界
        if 0 <= px < band.XSize and 0 <= py < band.YSize:
            # 读取NDVI值
            ndvi = band.ReadAsArray(px, py, 1, 1)[0][0]
            
            # 设置字段值
            feature.SetField(field_name, float(ndvi))
            vector_layer.SetFeature(feature)
    
    # 关闭数据集
    vector_ds = None
    raster_ds = None

point_dir = r'.\Data\firePoints\shp\grassland\fire'
lucc_dir = r'.\Data\normlization\VIIRS5KM'
point_list = [str(file) for file in Path(point_dir).glob('*.shp')]
lucc_list = [str(file) for file in Path(lucc_dir).glob('*.tif')]
field = 'Viirs'

for i in range(0, len(point_list)):
    f = os.path.splitext(os.path.basename(point_list[i]))[0]
    lucc_path = os.path.join(lucc_dir, f + '.tif')
    print(f"正在处理{f}年数据")
    extract_ndvi_to_points(point_list[i], lucc_path, field)
