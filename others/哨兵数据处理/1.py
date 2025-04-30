# import geopandas as gpd
# import rasterio
# from rasterio.mask import mask
# import os
# import numpy as np

# # 输入文件路径
# tif_path = r'E:\sentinel2\insar000re.tif'
# shp_path = r'E:\sentinel2\label00.shp'

# # 输出文件夹
# output_dir = r'E:\data\insar'

# # 读取矢量数据
# shp = gpd.read_file(shp_path)

# # 打开栅格数据（TIF文件）
# with rasterio.open(tif_path) as src:
#     # 获取栅格的基本信息
#     transform = src.transform
#     width = src.width
#     height = src.height
#     crs = src.crs

#     # 遍历每个面（Polygon）
#     for index, row in shp.iterrows():
#         # 获取当前面（Polygon）的几何信息
#         geometry = [row['geometry']]

#         # 使用mask函数裁剪栅格数据
#         out_image, out_transform = mask(src, geometry, crop=True)

#         # 计算裁剪后的图像尺寸
#         out_width = out_image.shape[2]
#         out_height = out_image.shape[1]

#         # 确保每个裁剪块为256x256（如果不够256x256，可以补充背景填充）
#         block_size = 256
#         for i in range(0, out_width, block_size):
#             for j in range(0, out_height, block_size):
#                 # 定义裁剪的块（256x256）
#                 window = rasterio.windows.Window(i, j, block_size, block_size)

#                 # 截取图像块
#                 cropped_image = out_image[:, j:j + block_size, i:i + block_size]

#                 # 保存块到文件
#                 output_filename = f"{output_dir}/block_{index}_{i}_{j}.tif"
#                 kwargs = src.meta.copy()
#                 kwargs.update({
#                     'driver': 'GTiff',
#                     'count': 1,
#                     'dtype': 'float32',
#                     'height': block_size,
#                     'width': block_size,
#                     'crs': crs,
#                     'transform': out_transform
#                 })

#                 with rasterio.open(output_filename, 'w', **kwargs) as dst:
#                     dst.write(cropped_image)

#                 print(f"Saved {output_filename}")



#单波段
# import geopandas as gpd
# import rasterio
# from rasterio.mask import mask
# import os
# import numpy as np
# from shapely.geometry import box

# # 输入文件路径
# tif_path = r'E:\sentinel2\pamieryou\2022.7.22\S2A_MSIL2A_20220722T055651_N0400_R091_T43SCC_20220722T092103.SAFE\S2A_MSIL2A_20220722T055651_N0400_R091_T43SCC_20220722T092103.tif'
# shp_path = r'E:\sentinel2\label0.shp'

# # 输出文件夹
# output_dir = r'E:\data\1'

# # 读取矢量数据（shp）
# shp = gpd.read_file(shp_path)

# # 打开栅格数据（tif文件）
# with rasterio.open(tif_path) as src:
#     # 获取栅格数据的边界
#     raster_bounds = src.bounds  # (minx, miny, maxx, maxy)

#     # 计算栅格数据的边界框
#     raster_box = box(raster_bounds[0], raster_bounds[1], raster_bounds[2], raster_bounds[3])

#     # 将矢量数据裁剪为与栅格图像相交的部分
#     shp = shp[shp.intersects(raster_box)]  # 只保留与栅格相交的部分
#     shp = shp.to_crs(src.crs)  # 确保矢量数据的CRS与栅格数据一致

#     # 遍历每个多边形进行裁剪
#     for index, row in shp.iterrows():
#         # 获取当前面（Polygon）的几何信息
#         geometry = [row['geometry']]

#         # 使用mask函数裁剪栅格数据
#         out_image, out_transform = mask(src, geometry, crop=True)

#         # 计算裁剪后的图像尺寸
#         out_width = out_image.shape[2]
#         out_height = out_image.shape[1]

#         # 确保每个裁剪块为256x256（如果不够256x256，可以补充背景填充）
#         block_size = 256
#         for i in range(0, out_width, block_size):
#             for j in range(0, out_height, block_size):
#                 # 定义裁剪的块（256x256）
#                 window = rasterio.windows.Window(i, j, block_size, block_size)

#                 # 截取图像块
#                 cropped_image = out_image[:, j:j + block_size, i:i + block_size]

#                 # 保存块到文件
#                 output_filename = f"{output_dir}/block_{index}_{i}_{j}.tif"
#                 kwargs = src.meta.copy()
#                 kwargs.update({
#                     'driver': 'GTiff',
#                     'count': 1,
#                     'dtype': 'float32',
#                     'height': block_size,
#                     'width': block_size,
#                     'crs': src.crs,
#                     'transform': out_transform
#                 })

#                 with rasterio.open(output_filename, 'w', **kwargs) as dst:
#                     dst.write(cropped_image)

#                 print(f"Saved {output_filename}")



# import geopandas as gpd
# import rasterio
# from rasterio.mask import mask
# import os
# from shapely.geometry import box

# # 输入文件路径
# tif_path = r'E:\sentinel2\pamieryou\2022.7.22\S2A_MSIL2A_20220722T055651_N0400_R091_T43SCC_20220722T092103.SAFE\S2A_MSIL2A_20220722T055651_N0400_R091_T43SCC_20220722T092103.tif'
# shp_path = r'E:\sentinel2\label0.shp'

# # 输出文件夹
# output_dir = r'E:\data\1'

# # 读取矢量数据（shp）
# shp = gpd.read_file(shp_path)

# # 打开栅格数据（tif文件）
# with rasterio.open(tif_path) as src:
#     # 获取栅格数据的边界
#     raster_bounds = src.bounds  # (minx, miny, maxx, maxy)

#     # 计算栅格数据的边界框
#     raster_box = box(raster_bounds[0], raster_bounds[1], raster_bounds[2], raster_bounds[3])

#     # 将矢量数据裁剪为与栅格图像相交的部分
#     shp = shp[shp.intersects(raster_box)]  # 只保留与栅格相交的部分
#     shp = shp.to_crs(src.crs)  # 确保矢量数据的CRS与栅格数据一致

#     # 遍历每个多边形进行裁剪
#     for index, row in shp.iterrows():
#         # 获取当前面（Polygon）的几何信息
#         geometry = [row['geometry']]

#         # 使用mask函数裁剪栅格数据
#         out_image, out_transform = mask(src, geometry, crop=True)

#         # 计算裁剪后的图像尺寸
#         out_width = out_image.shape[2]
#         out_height = out_image.shape[1]

#         # 确保每个裁剪块为256x256（如果不够256x256，可以补充背景填充）
#         block_size = 256
#         for i in range(0, out_width, block_size):
#             for j in range(0, out_height, block_size):
#                 # 定义裁剪的块（256x256）
#                 window = rasterio.windows.Window(i, j, block_size, block_size)

#                 # 截取图像块
#                 cropped_image = out_image[:, j:j + block_size, i:i + block_size]

#                 # 保存块到文件
#                 output_filename = f"{output_dir}/block_{index}_{i}_{j}.tif"
#                 kwargs = src.meta.copy()

#                 # 如果是多波段图像，设置正确的波段数
#                 kwargs.update({
#                     'driver': 'GTiff',
#                     'count': out_image.shape[0],  # 波段数
#                     'dtype': out_image.dtype,
#                     'height': block_size,
#                     'width': block_size,
#                     'crs': src.crs,
#                     'transform': out_transform
#                 })

#                 with rasterio.open(output_filename, 'w', **kwargs) as dst:
#                     # 如果是多波段图像，逐个波段写入
#                     for band in range(1, out_image.shape[0] + 1):
#                         dst.write(cropped_image[band - 1], band)

#                 print(f"Saved {output_filename}")


import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import numpy as np

# 输入栅格文件路径
raster1_path = r'E:\data001\test\cnd\cnd.tif'
raster2_path = r'E:\data001\test\insar\insar.tif'

# 打开第一个栅格（参考栅格）
with rasterio.open(raster1_path) as src1:
    # 获取第一个栅格的CRS和边界信息
    crs1 = src1.crs
    bounds1 = src1.bounds

    # 打开第二个栅格
    with rasterio.open(raster2_path) as src2:
        # 获取第二个栅格的CRS和边界信息
        crs2 = src2.crs
        bounds2 = src2.bounds

        # 如果坐标系不同，则需要进行重投影
        if crs1 != crs2:
            # 创建一个新的空栅格，用于存储重投影后的数据
            transform, width, height = rasterio.warp.calculate_default_transform(
                crs2, crs1, src2.width, src2.height, *bounds2
            )
            kwargs = src2.meta.copy()
            kwargs.update({
                'crs': crs1,
                'transform': transform,
                'width': width,
                'height': height
            })

            # 重投影第二个栅格
            reproject(
                source=rasterio.band(src2, 1),
                destination=np.empty((height, width), dtype=np.float32),
                src_transform=src2.transform,
                src_crs=crs2,
                dst_transform=transform,
                dst_crs=crs1,
                resampling=Resampling.nearest
            )

            # 保存重投影后的栅格
            with rasterio.open('reprojected_raster2.tif', 'w', **kwargs) as dst:
                dst.write(np.empty((height, width), dtype=np.float32), 1)

            # 更新路径为新重投影后的栅格文件
            raster2_path = 'reprojected_raster2.tif'
        
        # 重新打开重投影后的栅格
        with rasterio.open(raster2_path) as src2_reprojected:
            # 计算交集区域
            intersection_bounds = (
                max(bounds1[0], bounds2[0]),  # minx
                max(bounds1[1], bounds2[1]),  # miny
                min(bounds1[2], bounds2[2]),  # maxx
                min(bounds1[3], bounds2[3])   # maxy
            )

            # 创建一个裁剪窗口
            window = rasterio.windows.from_bounds(*intersection_bounds, transform=src1.transform)

            # 对两个栅格进行裁剪
            with rasterio.open(raster1_path) as src1:
                out_image1, _ = mask(src1, [box(*intersection_bounds)], crop=True)

            with rasterio.open(raster2_path) as src2:
                out_image2, _ = mask(src2, [box(*intersection_bounds)], crop=True)

            # 输出交集区域的栅格数据（可以保存为新的文件）
            output_filename1 = 'output_raster1_intersection.tif'
            output_filename2 = 'output_raster2_intersection.tif'

            kwargs1 = src1.meta.copy()
            kwargs2 = src2.meta.copy()

            kwargs1.update({
                'driver': 'GTiff',
                'height': out_image1.shape[1],
                'width': out_image1.shape[2],
                'crs': crs1,
                'transform': src1.transform
            })

            kwargs2.update({
                'driver': 'GTiff',
                'height': out_image2.shape[1],
                'width': out_image2.shape[2],
                'crs': crs1,
                'transform': src1.transform
            })

            with rasterio.open(output_filename1, 'w', **kwargs1) as dst1:
                dst1.write(out_image1)

            with rasterio.open(output_filename2, 'w', **kwargs2) as dst2:
                dst2.write(out_image2)

            print(f"Intersection of raster1 saved to {output_filename1}")
            print(f"Intersection of raster2 saved to {output_filename2}")
