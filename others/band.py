import rasterio

# 打开栅格文件
with rasterio.open(r'C:\Users\龙儿璨\Desktop\湿地制图\朱杭\img\WetlandXJ_202308_T43_PW_7.tif') as src:
    # 获取波段数量
    band_count = src.count
    print(f"Number of bands: {band_count}")
    
    # 获取每个波段的详细信息
    for i in range(1, band_count + 1):
        band = src.read(i)
        print(f"Band {i} stats:")
        print(f"  Min: {band.min()}, Max: {band.max()}")
        print(f"  Shape: {band.shape}")
        print(f"  Data type: {band.dtype}")
