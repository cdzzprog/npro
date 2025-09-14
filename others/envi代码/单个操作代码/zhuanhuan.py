# import os
# import rasterio
# from rasterio.warp import reproject, Resampling
# from pyproj import CRS

# def convert_tif_projection(input_tif, output_tif, src_crs, dst_crs):
#     """
#     将输入的TIFF图像从源投影转换为目标投影，并保留图像数据。
#     :param input_tif: 输入的TIFF文件路径
#     :param output_tif: 输出转换后的TIFF文件路径
#     :param src_crs: 源投影（例如，EPSG:32645）
#     :param dst_crs: 目标投影（例如，EPSG:32643）
#     """
#     with rasterio.open(input_tif) as src:
#         # 获取原始图像数据（读取所有波段）
#         image_data = src.read([1, 2, 3, 4])  # 读取红、蓝、绿、近红波段
#         transform = src.transform
#         src_meta = src.meta

#         # 获取原图的nodata值
#         src_nodata = src.nodata
        
#         # 更新输出文件的metadata
#         metadata = src_meta.copy()
#         metadata.update({
#             'crs': dst_crs,  # 更新目标CRS
#             'transform': transform,  # 仿射变换保持不变，后面会更新
#             'nodata': src_nodata  # 确保目标图像有合适的nodata值
#         })
        
#         # 设置目标的投影和仿射变换
#         dst_transform, width, height = rasterio.warp.calculate_default_transform(
#             src.crs, dst_crs, src.width, src.height, *src.bounds
#         )

#         # 更新metadata中的仿射变换和尺寸
#         metadata.update({
#             'crs': dst_crs,
#             'transform': dst_transform,
#             'width': width,
#             'height': height
#         })

#         # 创建目标TIFF文件并执行重投影
#         with rasterio.open(output_tif, 'w', **metadata) as dst:
#             for i in range(1, 5):  # 处理四个波段
#                 # 对每个波段进行重投影
#                 reproject(
#                     source=image_data[i - 1],  # 获取每个波段的数据
#                     destination=rasterio.band(dst, i),  # 写入对应波段
#                     src_transform=transform,
#                     src_crs=src.crs,
#                     dst_transform=dst_transform,
#                     dst_crs=dst_crs,
#                     resampling=Resampling.nearest  # 使用最近邻插值
#                 )

#     print(f"转换完成：{output_tif}")


# def batch_convert_tif_in_directory(input_dir, output_dir, src_crs, dst_crs):
#     """
#     批量转换目录中的所有TIFF图像文件
#     :param input_dir: 输入文件夹路径，包含待转换的TIFF图像
#     :param output_dir: 输出文件夹路径，用于保存转换后的TIFF图像
#     :param src_crs: 源投影（例如，EPSG:32645）
#     :param dst_crs: 目标投影（例如，EPSG:32643）
#     """
#     # 如果输出目录不存在，则创建
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # 遍历输入目录中的所有文件
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith('.tif'):
#             input_tif = os.path.join(input_dir, filename)
#             output_tif = os.path.join(output_dir, filename)
            
#             # 执行投影转换
#             convert_tif_projection(input_tif, output_tif, src_crs, dst_crs)


# # 设置源投影 (T45) 和目标投影 (T43)
# src_crs = CRS.from_epsg(32645)  # 例如，T45为 UTM Zone 45N
# dst_crs = CRS.from_epsg(32643)  # 目标投影为 UTM Zone 43N

# input_dir = r'C:\Users\龙儿璨\Desktop\湿地制图\11.9'  # 输入文件夹路径
# output_dir = r'C:\Users\龙儿璨\Desktop\湿地制图\11.9\T43'  # 输出文件夹路径

# # 执行批量转换
# batch_convert_tif_in_directory(input_dir, output_dir, src_crs, dst_crs)
import os
import rasterio
from pyproj import Proj
import shutil

def convert_tif_projection(input_tif, output_tif, src_proj, dst_proj):
    """
    将输入的TIFF图像从T45坐标系转换为T43坐标系
    :param input_tif: 输入的TIFF文件路径
    :param output_tif: 输出转换后的TIFF文件路径
    :param src_proj: 源投影（T45，EPSG代码或投影字符串）
    :param dst_proj: 目标投影（T43，EPSG代码或投影字符串）
    """
    # 打开输入TIFF文件
    with rasterio.open(input_tif) as src:
        # 获取输入文件的投影、仿射变换和坐标参考系
        transform = src.transform
        src_crs = src.crs
        
        # 创建目标投影
        src_proj = Proj(src_crs)  # 使用源投影的CRS
        dst_proj = Proj(dst_proj)  # 使用目标投影的CRS

        # 创建输出文件的metadata
        metadata = src.meta
        metadata.update({
            'crs': dst_proj.srs,  # 更新为目标投影
            'transform': transform  # 更新仿射变换
        })
        
        # 读取输入图像的四个波段（红、蓝、绿、近红）
        image_data = src.read([1, 2, 3, 4])  # 读取1, 2, 3, 4波段（假设为红、蓝、绿、近红）
        
        # 创建输出文件并保存
        with rasterio.open(output_tif, 'w', **metadata) as dst:
            for i in range(4):  # 处理四个波段
                dst.write(image_data[i], i + 1)  # 写入转换后的数据

    print(f"转换完成：{output_tif}")


def batch_convert_tif_in_directory(input_dir, output_dir, src_proj, dst_proj):
    """
    批量转换目录中的所有TIFF图像文件
    :param input_dir: 输入文件夹路径，包含待转换的TIFF图像
    :param output_dir: 输出文件夹路径，用于保存转换后的TIFF图像
    :param src_proj: 源投影（T45，EPSG代码或投影字符串）
    :param dst_proj: 目标投影（T43，EPSG代码或投影字符串）
    """
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.tif'):
            input_tif = os.path.join(input_dir, filename)
            output_tif = os.path.join(output_dir, filename)
            
            # 执行投影转换
            convert_tif_projection(input_tif, output_tif, src_proj, dst_proj)


# 设置源投影 (T45) 和目标投影 (T43)
src_proj = 'EPSG:32645'  # 例如，T45为 UTM Zone 45N（假设为 EPSG:32645）
dst_proj = 'EPSG:32643'  # 目标投影为 UTM Zone 43N（假设为 EPSG:32643）

input_dir = r'C:\Users\龙儿璨\Desktop\湿地制图\11.9'  # 输入文件夹路径
output_dir = r'C:\Users\龙儿璨\Desktop\湿地制图\11.9\T43'  # 输出文件夹路径

# 执行批量转换
batch_convert_tif_in_directory(input_dir, output_dir, src_proj, dst_proj)
