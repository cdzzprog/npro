import rasterio
from rasterio.windows import Window
import os

def clip_dem_by_optical_image(dem_path, optical_tif_dir, output_dir):
    """
    使用光学影像的边界裁剪 DEM 图像
    :param dem_path: 输入的 DEM 文件路径 (.tif)
    :param optical_tif_dir: 存放光学影像的小块目录路径，假设这些是 .tif 文件
    :param output_dir: 输出裁剪后的 DEM 文件保存目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取 DEM 图像
    with rasterio.open(dem_path) as dem_src:
        dem_transform = dem_src.transform
        dem_crs = dem_src.crs
        dem_width = dem_src.width
        dem_height = dem_src.height
        dem_bounds = dem_src.bounds

        # 遍历光学影像目录中的每个 TIFF 文件
        for optical_tif_file in os.listdir(optical_tif_dir):
            optical_tif_path = os.path.join(optical_tif_dir, optical_tif_file)

            if optical_tif_path.endswith('.tif'):
                with rasterio.open(optical_tif_path) as optical_src:
                    # 获取光学影像的边界
                    optical_bounds = optical_src.bounds
                    optical_transform = optical_src.transform
                    
                    # 检查 DEM 图像和光学影像的坐标系是否一致
                    if optical_src.crs != dem_crs:
                        raise ValueError(f"光学影像和 DEM 图像的坐标系不一致！光学影像：{optical_src.crs}, DEM 图像：{dem_crs}")
                    
                    # 计算光学影像裁剪区域在 DEM 图像中的像素坐标（行列号）
                    row_min, col_min = ~dem_transform * (optical_bounds[0], optical_bounds[3])
                    row_max, col_max = ~dem_transform * (optical_bounds[2], optical_bounds[1])

                    # 确保索引在 DEM 图像的有效范围内
                    row_min, col_min = max(0, int(row_min)), max(0, int(col_min))
                    row_max, col_max = min(dem_height, int(row_max)), min(dem_width, int(col_max))

                    # 读取 DEM 图像的裁剪区域
                    window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
                    dem_data = dem_src.read(1, window=window)

                    # 更新裁剪后的 DEM 的 transform
                    new_transform = rasterio.Affine(
                        dem_transform.a, dem_transform.b, optical_bounds[0],
                        dem_transform.d, dem_transform.e, optical_bounds[3]
                    )

                    # 保存裁剪后的 DEM 图像
                    output_file = os.path.join(output_dir, f"clipped_{os.path.basename(optical_tif_file)}")
                    metadata = dem_src.meta.copy()
                    metadata.update({
                        'driver': 'GTiff',
                        'count': 1,
                        'dtype': dem_data.dtype,
                        'width': col_max - col_min,
                        'height': row_max - row_min,
                        'crs': dem_crs,
                        'transform': new_transform
                    })

                    with rasterio.open(output_file, 'w', **metadata) as out_raster:
                        out_raster.write(dem_data, 1)

                print(f"裁剪后的 DEM 已保存至: {output_file}")

# 示例用法
dem_path =r'E:\BaiduNetdiskDownload\UTM新疆DEM10M\repj新疆WGS84.tif'  # DEM 文件路径
optical_tif_dir = r'E:\湿地制图\JSSF\img'  # 光学影像切割块的目录路径
output_dir = r'E:\湿地制图\JSSF\img\dem'  # 输出裁剪后的 DEM 文件的目录路径

clip_dem_by_optical_image(dem_path, optical_tif_dir, output_dir)
