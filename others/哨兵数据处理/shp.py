# coding=utf-8
# import os
# import arcpy
# from arcpy import env
 
# # 设置工作空间为包含 shp 文件的目录
# env.workspace = r"E:\data001\test\shp"
 
# # 设置保存转化后的影像路径
# save_path = r"E:\data001\test\label"
 
# # 判断保存路径是否存在，如果不存在则创建
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
 
# # 获取工作空间中的所有 shp 文件
# shp_files = arcpy.ListFeatureClasses("trainw84megeclipfanhebing.shp")
 
# # 设置像元大小
# cell_size = 10.0
 
# # 遍历每个 shp 文件
# for shp_file in shp_files:
#     # 构建输出的 tif 文件路径，使用原始 shp 文件名字加上 .tif 后缀
#     output_tif = os.path.join(save_path, "{}.tif".format(os.path.splitext(shp_file)[0]))
 
#     # 获取 shp 文件的空间参考信息
#     spatial_ref = arcpy.Describe(shp_file).spatialReference
 
#     # 进行 FeatureToRaster 转换
#     arcpy.FeatureToRaster_conversion(shp_file, "label", output_tif, cell_size)
 
#     # 更新生成的栅格的空间参考
#     arcpy.DefineProjection_management(output_tif, spatial_ref)
 
#     print("已经完成 {} 的转换。".format(shp_file))



#####强制匹配
import os
import arcpy
from arcpy import env

# 设置工作空间为包含 shp 文件的目录
env.workspace = r"E:\data001\test\labelshp"

# 设置保存转换后的影像路径
save_path = r"E:\data001\test\labelssss"

# 判断保存路径是否存在，如果不存在则创建
if not os.path.exists(save_path):
    os.mkdir(save_path)

# 获取工作空间中的所有 shp 文件
shp_files = arcpy.ListFeatureClasses("*.shp")

# 设置输入的参考 tiff 文件路径
input_tif = r"E:\data001\test\cnd\cnd10train.tif"  # 输入 tiff 文件路径

# 获取输入 tif 的空间参考、范围和像元大小
tif_desc = arcpy.Describe(input_tif)
tif_spatial_ref = tif_desc.spatialReference
tif_extent = tif_desc.extent
tif_cell_size = tif_desc.meanCellHeight  # 或者 tif_desc.meanCellWidth，根据需求使用

# 打印输入 tif 的行数和列数
print("输入 tiff 的列数: {}".format(tif_desc.width))
print("输入 tiff 的行数: {}".format(tif_desc.height))

# 遍历每个 shp 文件
for shp_file in shp_files:
    # 构建输出的 tif 文件路径，使用原始 shp 文件名字加上 .tif 后缀
    output_tif = os.path.join(save_path, "{}.tif".format(os.path.splitext(shp_file)[0]))

    # 获取 shp 文件的空间参考信息
    spatial_ref = arcpy.Describe(shp_file).spatialReference

    # 确保 shp 文件和输入 tif 文件的空间参考一致
    if spatial_ref.name != tif_spatial_ref.name:
        print("空间参考不一致: {} 和 {} 的空间参考不同".format(shp_file, input_tif))
        continue  # 跳过这个文件，进行下一个

    # 确保使用相同的像元大小
    arcpy.env.cellSize = tif_cell_size

    # 设置输出栅格的空间范围与输入 tif 一致
    arcpy.env.extent = tif_extent

    # 使用 FeatureToRaster 转换矢量数据为栅格
    arcpy.FeatureToRaster_conversion(shp_file, "label", output_tif, tif_cell_size)

    # 更新生成的栅格的空间参考
    arcpy.DefineProjection_management(output_tif, tif_spatial_ref)

    # 打印输出的 tiff 文件行列数
    output_desc = arcpy.Describe(output_tif)
    print("输出 {} 对应 tiff 的列数: {}".format(shp_file, output_desc.width))
    print("输出 {} 对应 tiff 的行数: {}".format(shp_file, output_desc.height))

    # 如果需要，可以检查生成的栅格是否与输入栅格完全一致
    if output_desc.width == tif_desc.width and output_desc.height == tif_desc.height:
        print("转换成功: {} 的输出与输入 tiff 匹配".format(shp_file))
    else:
        print("转换失败: {} 的输出与输入 tiff 不匹配".format(shp_file))
