import os
import argparse
from osgeo import gdal
from tqdm import tqdm
 
 
def crop_and_tile_tif(input_tif, output_dir, tile_size=(512, 512), keep_geo='True', suffix='.tif'):
    """
    使用GDAL裁剪并分块读取和写入GeoTIFF图像，并保存为小块。
    :param input_tif: 输入的tif文件路径或包含tif文件的目录
    :param output_dir: 保存切块的文件夹路径
    :param tile_size: 切块大小（宽度, 高度），默认 512x512
    :param keep_geo: 是否保留地理坐标信息，默认为True
    :param suffix: 文件后缀，默认是 .tif
    """
 
    def process_file(file_path):
        """裁剪单个TIF文件"""
        dataset = gdal.Open(file_path)
        if dataset is None:
            raise Exception(f"\033[91m无法打开输入文件: {file_path}\033[0m")
 
        # 获取图像尺寸
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        num_bands = dataset.RasterCount
        geo_transform = dataset.GetGeoTransform() if keep_geo=='True'  else None
        projection = dataset.GetProjection() if keep_geo=='True'  else None
        name_last = '.tif' if keep_geo=='True' else '.jpg'
        type = dataset.GetRasterBand(1).DataType
 
        # 计算行列的步长
        tile_width, tile_height = tile_size
        for i in tqdm(range(0, width, tile_width)):
            for j in range(0, height, tile_height):
                # 计算裁剪区域的宽度和高度
                w = min(tile_width, width - i)
                h = min(tile_height, height - j)
 
                # 创建输出文件名
                # output_tif = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_tile_{i}_{j}{name_last}")
                output_tif = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_{int(i/512)}_{int(j/512)}{name_last}")
 
                # 创建输出数据集
                driver = gdal.GetDriverByName('GTiff')
                out_dataset = driver.Create(output_tif, w, h, num_bands, type)
 
                # 设置地理坐标信息
                if keep_geo and geo_transform is not None:
                    new_geo_transform = list(geo_transform)
                    new_geo_transform[0] += i * geo_transform[1]
                    new_geo_transform[3] += j * geo_transform[5]
                    out_dataset.SetGeoTransform(new_geo_transform)
                    out_dataset.SetProjection(projection)
 
                # 逐波段读取并写入数据
                for band in range(1, num_bands + 1):
                    in_band = dataset.GetRasterBand(band)
                    out_band = out_dataset.GetRasterBand(band)
                    data = in_band.ReadAsArray(i, j, w, h)
                    out_band.WriteArray(data)
 
                # 释放输出数据集
                out_dataset.FlushCache()
                del out_dataset
 
        # 释放输入数据集
        del dataset
        print(f"\033[92m文件 {file_path} 裁剪完成，切块已保存。\033[0m")
 
    # 判断输入是文件还是目录
    if os.path.isdir(input_tif):
        # 遍历目录，处理指定后缀的文件
        for root, _, files in os.walk(input_tif):
            for file in files:
                if file.endswith(suffix):
                    file_path = os.path.join(root, file)
                    process_file(file_path)
    elif os.path.isfile(input_tif) and input_tif.endswith(suffix):
        # 单个文件处理
        process_file(input_tif)
    else:
        raise Exception(f"\033[91m输入路径不是有效的文件或目录，或者不包含指定后缀的文件: {input_tif}\033[0m")
 
 
def main():
    parser = argparse.ArgumentParser(description='使用GDAL裁剪并分块GeoTIFF文件。')
 
    # 添加命令行参数
    parser.add_argument('--input_tif', default=r'E:\sentinel2\weiyi\S2B_MSIL2A_20190726T060639_N9999_R134_T42SWH_20230512T114325.SAFE\S2B_MSIL2A_20190726T060639_N9999_R134_T42SWH_20230512T114325.tif', type=str, help='输入的tif文件路径或包含tif文件的目录')
    parser.add_argument('--output_dir', default=r'E:\sentinel2\Tarim\Sentinel-test', type=str, help='输出切块文件的目录')
    parser.add_argument('--tile_size', type=int, nargs=2, default=[256, 256], help='切块大小 (宽度, 高度)，默认为 512x512')
    parser.add_argument('--keep_geo', type=str, default='True', help='是否保留地理坐标信息，默认不保留')
    parser.add_argument('--suffix', type=str, default='.tif', help='文件后缀格式，默认为.tif')
 
    args = parser.parse_args()
 
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
 
    # 调用裁剪函数
    crop_and_tile_tif(args.input_tif, args.output_dir, tuple(args.tile_size), args.keep_geo, args.suffix)
 
 
if __name__ == '__main__':
    main()