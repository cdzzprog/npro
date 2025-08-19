# -*- coding: utf-8 -*-
import arcpy
import os
import csv
import codecs  # 引入 codecs 模块

# 输入和输出文件夹
input_folder = r"E:\sar\vector"
output_folder = r"E:\sar\csv"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置工作空间
arcpy.env.workspace = input_folder
shp_files = arcpy.ListFeatureClasses()

# 处理每个 .shp 文件
for shp in shp_files:
    # 获取完整路径
    shp_path = os.path.join(input_folder, shp)

    # 构造 CSV 输出文件名
    csv_filename = os.path.splitext(shp)[0] + ".csv"
    csv_path = os.path.join(output_folder, csv_filename)

    # 获取所有字段名称
    fields = [field.name for field in arcpy.ListFields(shp_path)]

    # 确保 ORIG_FID 字段存在
    if "FID" not in fields:
        print("Skipping {}: No 'FID' field found.".format(shp))
        continue

    # 获取 ORIG_FID 字段在字段列表中的索引
    orig_fid_index = fields.index("FID")

    # 使用 codecs 打开文件并指定 UTF-8 编码
    with codecs.open(csv_path, "w", "utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # 写入 CSV 头部
        writer.writerow(fields)

        # 读取所有字段的值，并按 ORIG_FID 排序
        with arcpy.da.SearchCursor(shp_path, fields) as cursor:
            sorted_rows = sorted(cursor, key=lambda x: x[orig_fid_index])  # 按 ORIG_FID 升序排序
            for row in sorted_rows:
                writer.writerow(row)

    print("Saved:", csv_path)

print("All shapefiles processed successfully!")
