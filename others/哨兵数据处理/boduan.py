from osgeo import gdal

def convert_to_tiff(input_file, output_file):
    ifile_name = 'MTD_MSIL2A.xml'  # 构造 MTD 文件名
    #  ofile_name = input_file.split(".SAFE")[0].split("/")[-1] + "_1.tif"  # 输出文件名
    ofile_name = input_file.split(".SAFE")[0].split("/")[-1] + ".tif"  # 输出文件名
    
    root_ds = gdal.Open(input_file+ifile_name)  # 打开 Sentinel-2 数据集
    ds_list = root_ds.GetSubDatasets()  # 获取子数据集列表
    # ds_list[0]、ds_list[1]和ds_list[2]分别对应10m、20m和60m分辨率的波段
    vrt_ds = gdal.BuildVRT('merged.vrt', ds_list[0][0])  # 创建VRT文件
    gdal.Translate(output_file+ofile_name, vrt_ds, options=['-co', 'COMPRESS=NONE'])  # 将VRT文件转换为GeoTIFF文件，不压缩数据
    # 清理 - 关闭数据集
    vrt_ds = None
    root_ds = None

# 示例调用
input_file_path = r'E:/sentinel2/dujiangyan/S2B_MSIL2A_20220807T034539_N0400_R104_T48RUV_20220807T061819.SAFE/'
output_file_path = r'E:/sentinel2/dujiangyan/S2B_MSIL2A_20220807T034539_N0400_R104_T48RUV_20220807T061819.SAFE/'
convert_to_tiff(input_file_path, output_file_path)
