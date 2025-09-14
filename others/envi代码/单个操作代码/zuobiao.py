import os
from osgeo import gdal

# 输入和输出文件路径
input_file = r'E:\BaiduNetdiskDownload\UTM新疆DEM10M\repj新疆WGS84.tif'  # 替换为你的输入文件路径
output_file = r'E:\BaiduNetdiskDownload\UTM\xingjiang.tif'  # 替换为你的输出文件路径

# 打开输入文件
dataset = gdal.Open(input_file)

# 检查文件是否成功打开
if dataset is None:
    print("无法打开输入文件")
else:
    # 设置目标坐标系
    target_srs = 'EPSG:4326'  # 替换为44N的EPSG代码

    # 执行坐标转换
    gdal.Warp(output_file, dataset, dstSRS=target_srs)

    print("坐标转换完成，输出文件为:", output_file)

# 清理资源
dataset = None
