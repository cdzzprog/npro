import rasterio
import numpy as np
import os

def split_raster(input_raster_path, output_folder, tile_size=512):
    # 打开 Sentinel-2 数据
    with rasterio.open(input_raster_path) as src:
        # 获取数据的元数据
        profile = src.profile
        width = src.width
        height = src.height
        bands = src.count
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_folder, exist_ok=True)
        
        # 计算生成的块数
        num_tiles_x = (width // tile_size) + (1 if width % tile_size > 0 else 0)
        num_tiles_y = (height // tile_size) + (1 if height % tile_size > 0 else 0)

        # 循环遍历并将影像分割成小块
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # 计算每个块的坐标范围
                x_start = i * tile_size
                x_end = min((i + 1) * tile_size, width)
                y_start = j * tile_size
                y_end = min((j + 1) * tile_size, height)

                # 从原始图像中切割出相应的区域
                window = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
                tile_data = src.read(window=window)  # 获取该窗口的影像数据

                # 创建输出文件名
                tile_name = f"tile_{i}_{j}.tif"
                tile_path = os.path.join(output_folder, tile_name)

                # 将切割的区域保存为新图像
                profile.update({
                    'height': tile_data.shape[1],
                    'width': tile_data.shape[2],
                    'transform': rasterio.windows.transform(window, src.transform)
                })

                with rasterio.open(tile_path, 'w', **profile) as dst:
                    dst.write(tile_data)

                print(f"Saved {tile_name} to {output_folder}")

# 输入 Sentinel-2 图像路径和输出文件夹
input_raster_path = r'E:\sentinel2\pamier\22.10.5\S2B_MSIL2A_20221005T055709_N0400_R091_T43SBC_20221005T084816.SAFE\S2B_MSIL2A_20221005T055709_N0400_R091_T43SBC_20221005T084816.tif'  # 替换为你的文件路径
output_folder = r'E:\sentinel2\pamier\22.10.5\seg'  # 替换为你想要保存切割块的文件夹

# 调用分割函数
split_raster(input_raster_path, output_folder, tile_size=512)
