## author: luo xin, creat: 2021.6.18, modify: 2021.7.14
# modify 2022.7.28
import numpy as np
from osgeo import gdal,gdalconst
from osgeo import osr
import os

from time import time
# Stop GDAL printing both warnings and errors to STDERR
gdal.PushErrorHandler('CPLQuietErrorHandler')

# Make GDAL raise python exceptions for errors (warnings won't raise an exception)
gdal.UseExceptions()

### tiff image reading
def readTiff(path_in):
    RS_Data=gdal.Open(path_in)
    im_col = RS_Data.RasterXSize  # 
    im_row = RS_Data.RasterYSize  # 
    im_bands =RS_Data.RasterCount  # 
    im_geotrans = RS_Data.GetGeoTransform()  # 
    im_proj = RS_Data.GetProjection()  # 
    img_array = RS_Data.ReadAsArray(0, 0, im_col, im_row).astype(np.float64)  #
    left = im_geotrans[0]
    up = im_geotrans[3]
    right = left + im_geotrans[1] * im_col + im_geotrans[2] * im_row
    bottom = up + im_geotrans[5] * im_row + im_geotrans[4] * im_col
    extent = (left, right, bottom, up)
    espg_code = osr.SpatialReference(wkt=im_proj).GetAttrValue('AUTHORITY',1)

    img_info = {'geoextent': extent, 'geotrans':im_geotrans,
                'geosrs': espg_code, 'row': im_row, 'col': im_col,
                    'bands': im_bands}

    if im_bands > 1:
        img_array = np.transpose(img_array, (1, 2, 0)) # 
        return img_array, img_info 
    else:
        return img_array, img_info


def writeTiff(im_data, im_geotrans, path_out,
              proj='''GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],
              PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'''):

    im_data = np.squeeze(im_data)
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_data = np.transpose(im_data, (2, 0, 1))
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands,(im_height, im_width) = 1,im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path_out, im_width, im_height, im_bands, datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(proj)

    if im_bands > 1:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset
    else:
        dataset.GetRasterBand(1).WriteArray(im_data)
        del dataset

def SynthesisBands(dstlist,dst_name,band_num):
    image_dst_list = np.loadtxt(dstlist, 'str')
    dataset_init = gdal.Open(image_dst_list[0])
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(dst_name, dataset_init.RasterXSize, dataset_init.RasterYSize, band_num,gdal.GDT_Float32)#
    out_ds.SetProjection(dataset_init.GetProjection())
    out_ds.SetGeoTransform(dataset_init.GetGeoTransform())
    flags=0
    for i in range(len(image_dst_list)):
        dataset = gdal.Open(image_dst_list[i])
        print(dataset.RasterCount)
        for j in range(dataset.RasterCount):
            band_temp = dataset.GetRasterBand(1+j)
            flags+=1

            out_ds.GetRasterBand(flags).WriteArray(band_temp.ReadAsArray())
    del out_ds
    print("the bands systhesised to one tif.")

def mean_normalization(source_name,dst_name):
    source_data=gdal.Open(source_name)

    gtiff_driver = gdal.GetDriverByName('GTiff')

    out_ds = gtiff_driver.Create(dst_name, source_data.RasterXSize, source_data.RasterYSize, source_data.RasterCount,
                                 gdal.GDT_Float32)
    out_ds.SetProjection(source_data.GetProjection())
    out_ds.SetGeoTransform(source_data.GetGeoTransform())
    print(source_data.RasterCount)
    for i in range(source_data.RasterCount):
        band_temp = source_data.GetRasterBand(1 + i)

        band_temp = band_temp.ReadAsArray()
        band_temp[band_temp<0]=0
        band_mean = np.mean(band_temp)
        print(band_mean)
        band_temp = band_temp / band_mean

        out_ds.GetRasterBand(1+i).WriteArray(band_temp)
    del out_ds
    print('mean normalization done')

def mean_normalization_for_dir(source_dir,dst_dir,prefix,norm_para,):

    for dir,ds,fs in os.walk(source_dir):
        for file_name in fs:

            source_name=dir+file_name
            source_data=gdal.Open(source_name)

            gtiff_driver = gdal.GetDriverByName('GTiff')
            dst_name=dst_dir+prefix+'_'+file_name

            out_ds = gtiff_driver.Create(dst_name, source_data.RasterXSize, source_data.RasterYSize, source_data.RasterCount,
                                 gdal.GDT_Float32)  #
            out_ds.SetProjection(source_data.GetProjection())
            out_ds.SetGeoTransform(source_data.GetGeoTransform())
            # print(source_data.RasterCount)
            for i in range(source_data.RasterCount):
                band_temp = source_data.GetRasterBand(1 + i)

                band_temp = band_temp.ReadAsArray()
                band_temp[band_temp<0]=0
                band_temp = band_temp / norm_para[i]

                out_ds.GetRasterBand(1+i).WriteArray(band_temp)
            del out_ds
    print('dir mean normalization done')


def CalculateMean(source_name):

    dataset = gdal.Open(source_name)
    print(dataset.RasterCount)
    mean_list=[]
    for j in range(dataset.RasterCount):
        band_temp = dataset.GetRasterBand(1+j)
        band_temp = band_temp.ReadAsArray()
        band_temp[band_temp < 0] = 0
        band_mean = np.mean(band_temp)
        mean_list.append(band_mean)

    print(mean_list)


def getImageList(image_dir,out_file_list_name):
    fresult_file = open(out_file_list_name, 'w')
    for root, dirs, files in os.walk(image_dir):
        if files:
            for file in files:
                prefix=file.split('.')[-1]
                if prefix=='tif':
                    print((root + '\\' + file))
                    fresult_file.write((root + '\\' + file + '\n'))
    print("image list created.")
    return fresult_file



def mosaic(dir,dst_name):
    os.chdir(dir)
    tifs = os.listdir()
    vrt = gdal.BuildVRT('all.vrt', tifs)
    gdal.Translate(dst_name, vrt)
    vrt=None

def fishnet_clip(source_file,target_dir,target_file_prefix,patch_size,proj,dtype=np.float32):
    img_array,img_info=readTiff(source_file)
    print(img_info,img_array.shape)
    geotrans=img_info['geotrans']

    num = 0
    s_time=time()

    row_start=0

    start_x,start_y=geotrans[0],geotrans[3]
    next_x,next_y=start_x,start_y + geotrans[5] * patch_size + geotrans[4] * patch_size

    img_d=len(img_array.shape)

    for i in range(img_info['row'] // patch_size):
        for j in range(img_info['col'] // patch_size):

            if i==0 and j==0:
                if img_d==3:
                    sub_image = img_array[0: patch_size, 0: patch_size, :].astype(dtype)
                elif img_d==2:
                    sub_image = img_array[0: patch_size, 0: patch_size].astype(dtype)
                writeTiff(sub_image, geotrans, target_dir + target_file_prefix + '{}.tif'.format(num), proj)
                num+=1
                row_start=0
                continue


            if row_start==0:
                start_x,start_y=start_x + geotrans[1] * patch_size + geotrans[2] * patch_size,start_y
                geotrans=(start_x,img_info['geotrans'][1],img_info['geotrans'][2],start_y,img_info['geotrans'][4],img_info['geotrans'][5])
            if row_start==1:
                start_x,start_y=next_x,next_y
                geotrans = (start_x, img_info['geotrans'][1],img_info['geotrans'][2], start_y, img_info['geotrans'][4],img_info['geotrans'][5])
                next_x,next_y=start_x,start_y + geotrans[5] * patch_size + geotrans[4] * patch_size
            if img_d == 3:
                sub_image = img_array[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size,:].astype(dtype)
            elif img_d == 2:
                sub_image = img_array[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size].astype(dtype)

            writeTiff(sub_image,geotrans,target_dir+target_file_prefix+'{}.tif'.format(num),proj)
            num += 1
            row_start=0
        row_start = 1

    e_time=time()
    t_time=e_time-s_time
    print("processing time {} s".format(num,t_time))



if __name__=='__main__':

    proj ='''PROJCS["WGS 84 / UTM zone 51N",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",123],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",0],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["EPSG","32651"]]'''

    SynthesisBands('[band list.txt]','[output.tif]',band_num=14)

    mean_normalization('[input.tif]','[output.tif]')

    fishnet_clip('[input.tif]','[save dir]','image_',128,proj)


    # mosaic('[save dir]','[output.tif]')