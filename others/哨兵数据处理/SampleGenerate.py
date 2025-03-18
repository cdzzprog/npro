from osgeo import gdal
from osgeo import ogr
from osgeo import gdalconst
import cv2
import numpy as np
import math
import os
from PIL import Image
import random
import shutil

def searchImage(Path = '', type = '.tif'):
  imageFiles = []           #创建队列
  for file in os.listdir(Path):
    if os.path.splitext(file)[1] == type:  # 查找.tif文件
      file = file.split('.')[0]
      imageFiles.append(file)
  return imageFiles

def clipSample(image, size, output, name):
    image = cv2.imread(image, -1)
    ww = image.shape[1]
    hh = image.shape[0]
    # bb = image.shape[2]
    nww = math.ceil(ww / size)
    nhh = math.ceil(hh / size)
    for i in range(nww):
        for j in range(nhh):
            sx = i*size
            sy = j*size
            ex = sx + size
            ey = sy + size
            if ex > ww:
                sx = ww - size - 1
                ex = sx + size
            if ey > hh:
                sy = hh - size - 1
                ey = sy + size

            blockdata = image[sy:ey, sx:ex]
            save = output + '\\' + name + '_' + str(i) + '_' + str(j) + '.tif'
            cv2.imwrite(save, blockdata)

def bigMapClip(image, size, output, name):
    # 读取大尺寸哨兵图像
    data = gdal.Open(image, gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    proj=data.GetProjection()
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    data_arr = data.ReadAsArray()
    band_count = data.RasterCount  # 波段数
    nww = math.ceil(x_res / size)
    nhh = math.ceil(y_res / size)
    type = data.GetRasterBand(1).DataType
    for i in range(nww):
        for j in range(nhh):
            sx = i*size
            sy = j*size
            ex = sx + size
            ey = sy + size
            if ex > x_res:
                sx = x_res - size - 1
                ex = sx + size
            if ey > y_res:
                sy = y_res - size - 1
                ey = sy + size
            driver = gdal.GetDriverByName("GTiff")
            save = output + '\\' + name + '_' + str(i) + '_' + str(j) + '.tif'
            out_tif = driver.Create(save, size, size, 4, type)
            out_tif.SetGeoTransform(geo_transform)
            out_tif.SetProjection(proj)
            for n in range(band_count):
                out_tif.GetRasterBand(n + 1).WriteArray(data_arr[n,sy:ey, sx:ex])
            # out_tif.GetRasterBand(1).WriteArray(data_arr[3,sy:ey, sx:ex])
            # out_tif.GetRasterBand(2).WriteArray(data_arr[1,sy:ey, sx:ex])
            # out_tif.GetRasterBand(3).WriteArray(data_arr[2,sy:ey, sx:ex])
            out_tif.FlushCache()  # 最终将数据写入硬盘
            out_tif = None  # 注意必须关闭tif文件
    print('裁块完毕！')

def shp2Raster(shp,templatePic,output,field,nodata):
    """
    shp:字符串，一个矢量，从0开始计数，整数
    templatePic:字符串，模板栅格，一个tif，地理变换信息从这里读，栅格大小与该栅格一致
    output:字符串，输出栅格，一个tif
    field:字符串，栅格值的字段
    nodata:整型或浮点型，矢量空白区转换后的值
    """
    ndsm = templatePic
    data = gdal.Open(ndsm, gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    proj=data.GetProjection()
    #source_layer = data.GetLayer()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    mb_v = ogr.Open(shp)
    mb_l = mb_v.GetLayer()
    pixel_width = geo_transform[1]
    #输出影像为16位整型
    target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Int16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(proj)
    band = target_ds.GetRasterBand(1)
    NoData_value = nodata
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE=%s"%field,'ALL_TOUCHED=TRUE'])
    target_ds = None

def merge2CDlabel():
    # 两时相新增消失建筑标签合并 
    t1label = r'D:\work\code\Dataset\ChangeDetection\Suzhou\label\T1_Label.tif'
    t2label = r'D:\work\code\Dataset\ChangeDetection\Suzhou\label\T2_Label.tif'
    output = r'D:\work\code\Dataset\ChangeDetection\Suzhou\label\label.tif'
    t1label = cv2.imread(t1label, -1)
    t2label = cv2.imread(t2label, -1)
    label = t1label + t2label
    label = ((label > 0) * 255).astype(np.uint8)
    cv2.imwrite(output, label)

def sampleAug(t1dir, t2dir, labeldir, output):
    nameList = searchImage(t1dir)
    t1images = [os.path.join(t1dir, name + ".tif") for name in nameList]
    t2images = [os.path.join(t2dir, name + ".tif") for name in nameList]
    labelFiles = [os.path.join(labeldir, name + ".tif") for name in nameList]
    count = len(t1images)
    for i in range(0,count):
        # t1img = cv2.imread(t1images[i], -1)
        # t2img = cv2.imread(t2images[i], -1)
        # label = cv2.imread(labelFiles[i], -1)
        t1img = Image.open(t1images[i])
        t2img = Image.open(t2images[i])
        label = Image.open(labelFiles[i])
        # 水平翻转
        vert_t1img = flip_image(t1img, mode='vertical')
        vert_t2img = flip_image(t2img, mode='vertical')
        vert_label = flip_image(label, mode='vertical')
        # 垂直翻转
        hori_t1img = flip_image(t1img, mode='horizontal')
        hori_t2img = flip_image(t2img, mode='horizontal')
        hori_label = flip_image(label, mode='horizontal')
        # 对角线翻转
        diag_t1img = flip_image(vert_t1img, mode='horizontal')
        diag_t2img = flip_image(vert_t2img, mode='horizontal')
        diag_label = flip_image(vert_label, mode='horizontal')
        save_T1 = output + '/IMG_T1/' + nameList[i] + '_vert.tif'
        save_T2 = output + '/IMG_T2/' + nameList[i] + '_vert.tif'
        save_Label = output + '/LABEL/' + nameList[i] + '_vert.tif'
        vert_t1img.save(save_T1)
        vert_t2img.save(save_T2)
        vert_label.save(save_Label)
        save_T1 = output + '/IMG_T1/' + nameList[i] + '_hori.tif'
        save_T2 = output + '/IMG_T2/' + nameList[i] + '_hori.tif'
        save_Label = output + '/LABEL/' + nameList[i] + '_hori.tif'
        hori_t1img.save(save_T1)
        hori_t2img.save(save_T2)
        hori_label.save(save_Label)
        save_T1 = output + '/IMG_T1/' + nameList[i] + '_diag.tif'
        save_T2 = output + '/IMG_T2/' + nameList[i] + '_diag.tif'
        save_Label = output + '/LABEL/' + nameList[i] + '_diag.tif'
        diag_t1img.save(save_T1)
        diag_t2img.save(save_T2)
        diag_label.save(save_Label)

def randomDivid(input, output, num):
    t1dir = input + '/IMG_T1/'
    t2dir = input + '/IMG_T2/'
    labeldir = input + '/LABEL/'
    nameList = searchImage(t1dir)
    t1images = [os.path.join(t1dir, name + ".tif") for name in nameList]
    t2images = [os.path.join(t2dir, name + ".tif") for name in nameList]
    labelFiles = [os.path.join(labeldir, name + ".tif") for name in nameList]
    count = len(nameList)
    random_idx = random.sample(range(0, count), num)
    for i in range(num):
        t1image = t1images[random_idx[i]]
        t2image = t2images[random_idx[i]]
        labelFile = labelFiles[random_idx[i]]
        saveT1 = output + '/IMG_T1/' + nameList[random_idx[i]] + '.tif'
        saveT2 = output + '/IMG_T2/' + nameList[random_idx[i]] + '.tif'
        saveLabel = output + '/LABEL/' + nameList[random_idx[i]] + '.tif'
        shutil.move(t1image, saveT1)
        shutil.move(t2image, saveT2)
        shutil.move(labelFile, saveLabel)
        

def flip_image(image, mode='horizontal'):
    """
    翻转图像
    :param image: 原始图像
    :param mode: 翻转模式 ('horizontal' 或 'vertical')
    :return: 翻转后的图像
    """
    if mode == 'horizontal':
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == 'vertical':
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("mode should be 'horizontal' or 'vertical'")
    return flipped_image

def labelAug(labeldir, output):
    nameList = searchImage(labeldir)
    labelFiles = [os.path.join(labeldir, name + ".tif") for name in nameList]
    count = len(labelFiles)
    for i in range(0,count):
        label = Image.open(labelFiles[i])
        # 水平翻转
        vert_label = flip_image(label, mode='vertical')
        # 垂直翻转
        hori_label = flip_image(label, mode='horizontal')
        # 对角线翻转
        diag_label = flip_image(vert_label, mode='horizontal')
        save_Label = output + nameList[i] + '_vert.tif'
        vert_label.save(save_Label)
        save_Label = output + nameList[i] + '_hori.tif'
        hori_label.save(save_Label)
        save_Label = output + nameList[i] + '_diag.tif'
        diag_label.save(save_Label)

def findsamesample(refdir, labeldir, output):
    nameList = searchImage(refdir)
    labelFiles = [os.path.join(labeldir, name + ".tif") for name in nameList]
    saveFiles = [os.path.join(output, name + ".tif") for name in nameList]
    count = len(labelFiles)
    for i in range(0,count):
        shutil.copy(labelFiles[i], saveFiles[i])
        
def main():
    # 矢量转栅格
    # shp = r'D:\work\Dataset\ChangeDetection\Suzhou\shp\T1_seg.shp'
    # templatePic = r'D:\work\Dataset\ChangeDetection\Suzhou\image\T1.tif'
    # # output = r'D:\work\code\Dataset\ChangeDetection\Suzhou\label\T2_seg.tif'
    # output = r'D:\work\Dataset\ChangeDetection\Suzhou\label\T1_label.tif'
    # field = 'test'
    # nodata = 0
    # shp2Raster(shp,templatePic,output,field,nodata)
    # merge2CDlabel()
    image = r'D:\work\Dataset\Tarim\Sentinel-2\tif\11_November.tif'
    output = r'D:\work\Dataset\Tarim\Sentinel-2\tif_512\11_November'
    name = 'Tarim'
    # clipSample(image, 256, output, name)
    bigMapClip(image, 512, output, name)
    # t1dir = 'D:\\work\\code\\Dataset\\ChangeDetection\\Suzhou\\sample\\train\\IMG_T1\\'
    # t2dir = 'D:\\work\\code\\Dataset\\ChangeDetection\\Suzhou\\sample\\train\\IMG_T2\\'
    # labeldir = 'D:\\work\\code\\Dataset\\ChangeDetection\\Suzhou\\sample\\train\\LABEL\\'
    # output = 'D:\\work\\code\\Dataset\\ChangeDetection\\Suzhou\\sample\\allSample\\'
    # sampleAug(t1dir, t2dir, labeldir, output)
    # input = 'D:\\work\\code\\Dataset\\ChangeDetection\\Suzhou\\sample\\allSample\\aug\\'
    # output = 'D:\\work\\code\\Dataset\\ChangeDetection\\Suzhou\\sample\\val\\'
    # num = 80
    # randomDivid(input, output, num)
    # labeldir = 'D:\\work\\code\\Dataset\\ChangeDetection\\Suzhou\\sample\\allSample\\IMG_T2\\'
    # output = 'D:\\work\\code\\Dataset\\ChangeDetection\\Suzhou\\sample\\allSample\\aug\\IMG_T2\\'
    # labelAug(labeldir, output)
    # refdir = 'D:\\work\\code\\Dataset\\ChangeDetection\\Suzhou\\sample\\test\\LABEL\\'
    # labeldir = 'D:\\work\\code\\Dataset\\ChangeDetection\\Suzhou\\sample\\allSample\\aug\\LABELGT\\'
    # output = 'D:\\work\\code\\Dataset\\ChangeDetection\\Suzhou\\sample\\test\\LABELGT\\'
    # findsamesample(refdir, labeldir, output)

if __name__ == "__main__":
    main()