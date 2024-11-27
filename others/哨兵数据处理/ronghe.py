from osgeo import gdal
import numpy as np

def convert_to_tiff(filename, output_path):
    # 打开栅格数据集
    root_ds = gdal.Open(filename)
    ds_list = root_ds.GetSubDatasets()  # 获取子数据集
    # print(df_list)  # 查看各个子集对应的波段详解
    visual_ds = gdal.Open(ds_list[0][0])  # 打开第一个数据子集的路径，ds_list[i][0]表示不同的子集，也就对应不同分辨率的波段。
    visual_arr = visual_ds.ReadAsArray()  # 将数据集中的数据读取为ndarray

    # 创建.tif文件
    band_count = visual_ds.RasterCount  # 波段数
    xsize = visual_ds.RasterXSize  # 列数
    ysize = visual_ds.RasterYSize  # 行数
    out_tif_name = output_path + "/" + filename.split(".SAFE")[0].split("/")[-1] + ".tif"  # 输出文件名
    driver = gdal.GetDriverByName("GTiff")  # 创建输出文件
    out_tif = driver.Create(out_tif_name, xsize, ysize, band_count, gdal.GDT_Float32)  # 创建输出文件
    out_tif.SetProjection(visual_ds.GetProjection())  # 设置投影坐标
    out_tif.SetGeoTransform(visual_ds.GetGeoTransform())  # 设置仿射变换参数

    for index, band in enumerate(visual_arr):  # 遍历每个波段
        band = np.array([band])  # 将每个波段的数据转换为ndarray
        for i in range(len(band[:])):
            # 数据写出
            out_tif.GetRasterBand(index + 1).WriteArray(band[i])

    out_tif.FlushCache()  # 最终将数据写入硬盘
    out_tif = None  # 注意必须关闭tif文件

# 调用示例
path = r'E:/sentinel2/ludingafter/S2A_MSIL2A_20220911T034551_N0400_R104_T47RQM_20220911T084454.SAFE/'
output_path = r'E:/sentinel2/ludingafter/S2A_MSIL2A_20220911T034551_N0400_R104_T47RQM_20220911T084454.SAFE/data'
filename = path + 'MTD_MSIL2A.xml'
convert_to_tiff(filename, output_path)  # 调用转换函数
print("转换完成")
