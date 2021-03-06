# coding=gbk
'''
author: LiYang
Date: 20191108
Location: zzq BeiJing
Desctiption: SPEI pre-process
'''
import os
import gzip
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import to_raster
from netCDF4 import Dataset
import requests
import analysis
# from pyhdf.SD import SD
import re
import requests
from __init__ import *

# exit()
# this_root = 'D:/project05/'



def check_zip(path):
    '''
    :param path: *.zip
    :return: True 下载
    :return: False 不下载
    '''
    if not os.path.isfile(path):
        return True
    # ZipFile = zipfile.ZipFile
    f_gzip = gzip.GzipFile(path, "rb")
    print(path)
    print(f_gzip.read())
    f_gzip.close()



class pre_poecess_daily:
    def __init__(self):
        # 0 下载数据
        # self.download()
        # 1 *.gz解压数据
        # unzip()
        # 2 生成tif
        # 2.1 nc to tif
        # self.nc_to_tif()
        # nc_to_tif('SWE_avg')
        # 3 定义投影
        # arcpy_func.define_swe_projection()
        # 4 转换成84
        # arcpy_func.re_projection_swe()
        # 5 split_year
        # self.split_annual()
        # 6 转换为per_pix
        # fdir = data_root + 'GLOBSWE/reproj/swe_annual/'
        # for folder in os.listdir(fdir):
        #     print(folder)
        #     print('\n')
        #     fdir_i = data_root + 'GLOBSWE/reproj/swe_annual/{}/'.format(folder)
        #     outf = data_root + 'GLOBSWE/per_pix_annual/'+folder
        #     self.data_transform(fdir_i,outf)
        # 7 计算 thaw融雪日
        # self.get_thaw()
        # self.insert_per_pix()
        # 6 计算 anomaly
        self.cal_anomaly()
        pass


    def download(self):
        # 1 get download list
        url = 'https://www.globsnow.info/swe/archive_v3.0/L3A_daily_SWE/NetCDF4/'
        request = requests.request('GET',url)
        body = request.text
        # exit()
        # fw = open(f, 'w')
        p = re.findall('<a href=".*?">', body)
        urllist = []
        params = []
        for pi in p:
            if '.nc' in pi:
                url_i = pi.split('"')[1]
                urllist.append(url_i)
                params.append((url,url_i))
        # for i in tqdm(urllist):
        #     self.kernel_download([url,i])
        MULTIPROCESS(self.kernel_download,params).run(10,'t',desc='downloading...')

    def kernel_download(self,params):
        url_father,url_i = params
        url = url_father + url_i
        name = '{}'.format(url_i)
        out_dir = data_root + 'GLOBSWE/download/'
        f = out_dir + name
        T.mk_dir(out_dir, force=True)
        # check_zip(f)
        # if check_zip(f):
        #     print f
        if not os.path.isfile(f):
            # print(f)
            try:
                # print(1/0)
                self.downloadFILE(url, f)
            except:
                fw = open(f + '.error','wb')
                fw.close()
                print(f,'error')
                # time.sleep(1)


    def downloadFILE(self,url, name):
        resp = requests.get(url=url, stream=True)
        # stream=True的作用是仅让响应头被下载，连接保持打开状态，
        # content_size = int(resp.headers['Content-Length'])/1024        #确定整个安装包的大小
        con = resp.content
        fw = open(name, "wb")
        fw.write(con)

    def unzip(self):
        fdir = this_root + 'GLOBSWE/download/'
        outdir = this_root + 'GLOBSWE/nc/'
        analysis.Tools().mk_dir(outdir)
        for f in os.listdir(fdir):
            if f.endswith('.gz'):
                # print f
                try:
                    f_gzip = gzip.GzipFile(fdir + f, "rb")
                    content = f_gzip.read()
                    with open(outdir + f.split('.')[0] + '.nc', 'wb') as fw:
                        fw.write(content)
                except:
                    pass

    def nc_to_tif(self,data='swe'):
        # data = 'SWE_avg'
        # data = 'SWE_max'
        # 可行
        ncdir = data_root + 'GLOBSWE/download/'
        out_dir = data_root + 'GLOBSWE/nc_to_tif/' + data + '/'
        analysis.Tools().mk_dir(out_dir, force=True)
        for f in tqdm(os.listdir(ncdir)):
            # print f
            nc = ncdir + f
            ncin = Dataset(nc, 'r')
            x = np.array(ncin['x'])
            y = np.array(ncin['y'])
            swe_avg = np.array(ncin[data])
            swe_avg = np.array(swe_avg)
            longitude_start, latitude_start, pixelWidth, pixelHeight = x[0], y[0], x[1] - x[0], y[1] - y[0]
            fname = f.split('_')[0] + '.tif'
            # print(fname)
            to_raster.array2raster_polar(out_dir + fname, longitude_start, latitude_start, pixelWidth, pixelHeight,
                                         swe_avg, ndv=-1)
            # plt.imshow(swe_avg)
            # plt.show()
            # exit()
            pass

    def hdf_to_tif(self,data='swe_average'):
        # data = 'swe_maximum'
        # data = 'swe_average'

        # 获取nc空间坐标信息
        nc = r'D:\project05\GLOBSWE\nc\198203.nc'
        ncin = Dataset(nc, 'r')
        x = np.array(ncin['x'])
        y = np.array(ncin['y'])
        longitude_start, latitude_start, pixelWidth, pixelHeight = x[0], y[0], x[1] - x[0], y[1] - y[0]
        hdf_dir = this_root + 'GLOBSWE/download/'
        out_dir = this_root + 'GLOBSWE/hdf_to_tif/' + data + '/'
        analysis.Tools().mk_dir(out_dir, force=True)
        for f in os.listdir(hdf_dir):
            if f.endswith('.hdf'):
                # print f
                fname = hdf_dir + f
                hdf = SD(fname)
                # print hdf.datasets()
                # swe = np.array(hdf.select(data)['fakeDim2'])
                swe = hdf.select(data)[:]
                # swe = np.ma.masked_where(swe == -1,swe)
                # plt.imshow(swe)
                # plt.show()
                # attr = hdf.attributes(full=1)
                # attNames = attr.keys()
                # attNames.sort()
                # print attNames
                # print attr['Spatial Resolution ']
                out_fname = f.split('_')[-2] + '.tif'
                to_raster.array2raster_polar(out_dir + out_fname, longitude_start, latitude_start, pixelWidth,
                                             pixelHeight, swe, ndv=-1)

        pass


    def split_annual(self):
        fdir = data_root + 'GLOBSWE/reproj/' + 'swe' + '/'
        outdir = data_root + 'GLOBSWE/reproj/' + 'swe_annual' + '/'
        T.mk_dir(outdir)

        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.tif'):
                year = f[:4]
                outdir_i = outdir + year + '/'
                T.mk_dir(outdir_i)
                shutil.copy(fdir + f,outdir_i + f)
        pass

    def proj_trans1(self):
        # 出现花纹

        grid = []
        for i in range(721):
            temp = []
            for j in range(1442):
                temp.append(-999999)
            grid.append(temp)
        grid = np.array(grid)
        # plt.imshow(grid)
        # plt.show()

        nc = r'D:\project05\GLOBSWE\download\198412\198412.nc'
        ncin = Dataset(nc, 'r')

        lat = np.array(ncin['lat'])
        lon = np.array(ncin['lon'])

        swe_avg = np.array(ncin['SWE_avg'])
        x_list = []
        y_list = []
        val_list = []
        for i in tqdm(range(len(swe_avg))):
            for j in range(len(swe_avg[0])):
                lon_ = lon[i][j] + 180
                lat_ = lat[i][j] + 90
                val = swe_avg[i][j]

                if lon_ < -999 and lat_ < -999:
                    continue
                # print lon_,lat_,val
                # x = int(round(lon_ / 0.25,0))
                # y = int(round(lat_ / 0.25,0))
                x_list.append(lon_)
                y_list.append(lat_)
                val_list.append(val)
                # print x,y
                # try:
                # grid[y][x] = val
                # except:
                #     pass
                # pass
        # print len(x_list)
        # print len(set(x_list))
        exit()
        grid = np.ma.masked_where(grid < 0, grid)
        grid = grid[::-1]
        plt.imshow(grid)
        plt.show()

    def data_transform(self,fdir,outf):
        date_list = []
        date_start = outf.split('/')[-1]
        date_start = int(date_start)
        date_end = date_start + 1

        date_start_obj = datetime.datetime(date_start,1,1)
        date_end_obj = datetime.datetime(date_end,1,1)
        delta_day = date_end_obj - date_start_obj
        delta_day = delta_day.days
        for d in range(delta_day):
            date_delta = datetime.timedelta(d)
            date = date_start_obj + date_delta
            year, mon, day = date.year, date.month, date.day
            date_str = '{:d}{:02d}{:02d}'.format(year, mon, day)
            date_list.append(date_str)
        template_f = os.path.join(fdir, os.listdir(fdir)[0])
        template_arr = to_raster.raster2array(template_f)[0]
        void_arr = np.ones_like(template_arr) * -999999
        all_array = []
        for d in tqdm(date_list, 'loading...'):
            fname = fdir + d + '.tif'
            # print(fname)
            if os.path.isfile(fname):
                array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fname)
                all_array.append(array)
            else:
                all_array.append(void_arr)
        # print len(all_array)
        all_array = np.array(all_array)
        row = len(all_array[0])
        col = len(all_array[0][0])

        void_dic = {}
        void_dic_list = []
        for r in range(row):
            for c in range(col):
                void_dic[(r, c)] = []
                void_dic_list.append((r, c))

        for r in tqdm(range(row), desc='transforming...'):
            for c in range(col):
                for arr in all_array:
                    val = arr[r][c]
                    void_dic[(r, c)].append(val)
        for pix in tqdm(void_dic,desc='list to np.array...'):
            void_dic[pix] = np.array(void_dic[pix])
        print('saving...')
        np.save(outf, void_dic)

    def insert_nan_vals(self,vals):
        # input [1,2,3,4,10,11,12,1,2,3,4]
        # output [1,2,3,4,0,0,0,0,0,10,11,12,1,2,3,4,0,0,0,0,0]
        for i in range(238 // 7):
            for j in range(5):
                vals.insert(4 + i * 12, 0)
        return vals

    def insert_per_pix(self):
        mode = 'SWE_avg'
        # mode = 'SWE_max'
        fdir = this_root+'GLOBSWE/per_pix/'+mode+'/'
        out_dir = this_root+'GLOBSWE/per_pix/'+mode+'_408/'
        analysis.Tools().mk_dir(out_dir)
        # spatial_dic = {}

        for f in tqdm(os.listdir(fdir)):
            dic = dict(np.load(fdir+f).item())
            data_dic = {}
            for pix in dic:
                val = dic[pix]
                if val[0] > 0:
                    val = self.insert_nan_vals(val)
                    data_dic[pix] = val

                    # plt.plot(val)
                    # print len(val)
                    # # print f
                    # plt.show()
                else:
                    data_dic[pix] = [-999999]*408
        # arr = analysis.DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.show()
            np.save(out_dir+f,data_dic)

    def kernel_get_thaw(self,params):

        f,outdir,fdir,window = params

        year = f.split('.')[0]
        outf = outdir + year + '.tif'
        thaw_date_dic = {}
        dic = T.load_npy(fdir + f)
        for pix in dic:
            val = dic[pix]
            val = np.array(val)
            val_origin = copy.copy(val)
            val[val < 1] = np.nan
            if np.isnan(np.nanmean(val)):
                continue
            thaw_date = -999999
            for i in range(len(val)):
                if i + window >= len(val):
                    break
                picked_vals = val[i:i + window]
                is_nan_list = list(np.isnan(picked_vals))
                True_count = is_nan_list.count(True)
                if True_count == window:
                    thaw_date = i
                    break

            thaw_date_dic[pix] = thaw_date
            # if thaw_date == 0:
            #     # plt.scatter(range(len(val)),val,c='r')
            #     print(len(val_origin))
            #     val_origin[val_origin<-1] = np.nan
            #     plt.scatter(range(len(val_origin)),val_origin,c='b')
            #     # plt.plot(range(len(interp_val)),interp_val)
            #     plt.show()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(thaw_date_dic)
        DIC_and_TIF().arr_to_tif(arr, outf)

        pass

    def get_thaw(self):
        window = 60 # days
        fdir = data_root + 'GLOBSWE/per_pix_annual/'
        outdir = data_root + 'GLOBSWE/thaw_tif/'
        T.mk_dir(outdir)
        params = []
        for f in os.listdir(fdir):
            params.append((f,outdir,fdir,window))
        MULTIPROCESS(self.kernel_get_thaw,params).run(process=5)
        pass

    def cal_anomaly(self):
        fdir = data_root + 'GLOBSWE/thaw_tif/'
        outdir_anomaly = data_root + 'GLOBSWE/thaw_tif_anomaly/'
        outdir_std_anomaly = data_root + 'GLOBSWE/thaw_tif_std_anomaly/'
        T.mk_dir(outdir_anomaly)
        T.mk_dir(outdir_std_anomaly)
        arrs = []
        for f in sorted(os.listdir(fdir)):
            arr = to_raster.raster2array(fdir+f)[0]
            arrs.append(arr)

        mean_dic = {}
        std_dic = {}
        for i in tqdm(range(len(arrs[0]))):
            for j in range(len(arrs[0][0])):
                pix = (i,j)
                val_list = []
                for k in range(len(arrs)):
                    val = arrs[k][i][j]
                    if val <= 0:
                        continue
                    val_list.append(val)
                if len(val_list) == 0:
                    continue
                mean = np.nanmean(val_list)
                std = np.nanstd(val_list)
                mean_dic[pix] = mean
                std_dic[pix] = std
        mean_arr = DIC_and_TIF().pix_dic_to_spatial_arr(mean_dic)
        std_arr = DIC_and_TIF().pix_dic_to_spatial_arr(std_dic)
        base = 1982
        flag = 0
        for arr in arrs:
            fname = '{}.tif'.format(base+flag)
            print(fname)
            arr[arr <= 0] = np.nan
            anomaly_arr = arr - mean_arr
            std_anomaly_arr = anomaly_arr / std_arr
            DIC_and_TIF().arr_to_tif(anomaly_arr,outdir_anomaly+fname)
            DIC_and_TIF().arr_to_tif(std_anomaly_arr,outdir_std_anomaly+fname)
            flag += 1

        pass


class pre_poecess_monthly:
    def __init__(self):
        # 0 下载数据
        # download()
        # 1 *.gz解压数据
        # unzip()
        # 2 生成tif
        # 2.1 nc to tif
        # nc_to_tif('SWE_max')
        # nc_to_tif('SWE_avg')
        # 2.2 hdf to tif
        # hdf_to_tif(data='swe_average')
        # hdf_to_tif(data='swe_maximum')
        # 3 定义投影
        # arcpy_func.define_swe_projection()
        # 4 转换成84
        # arcpy_func.re_projection_swe()
        # 5 转换为per_pix
        # self.data_transform()
        # self.insert_per_pix()
        # 6 计算 anomaly
        # analysis.Pre_Process().cal_anomaly()
        pass


    def download(self):
        year_list = []
        for y in range(1982,2016):
            year_list.append(str(y))
        date_list = []
        for y in year_list:
            for mon in range(1,13):
                mon = '%02d'%mon
                date = y+mon
                date_list.append(date)
        params = []
        for i in date_list:
            y = i[:4]
            date = i
            params.append([y,date])
        for i in params:
            self.kernel_download(i)
        # analysis.MUTIPROCESS(kernel_download,params).run(10,'t',desc='downloading...')

    def kernel_download(self,params):
        year, date = params
        # url = 'http://www.globsnow.info/swe/archive_v2.0/'
        url = 'http://www.globsnow.info/swe/archive_v2.0/{}/L3B_monthly_SWE/' \
              'GlobSnow_SWE_L3B_monthly_{}_v2.0.nc.gz'.format(year, date)
        # print url
        name = '{}.gz'.format(date)
        out_dir = this_root + 'GLOBSWE/download/'
        f = out_dir + name

        analysis.Tools().mk_dir(out_dir, force=True)
        # check_zip(f)
        # if check_zip(f):
        #     print f
        if not os.path.isfile(f):
            # print f
            # print url
            self.downloadFILE(url, f)

    def downloadFILE(self,url, name):
        resp = requests.get(url=url, stream=True)
        # stream=True的作用是仅让响应头被下载，连接保持打开状态，
        # content_size = int(resp.headers['Content-Length'])/1024        #确定整个安装包的大小
        with open(name, "wb") as f:
            # for data in tqdm(iterable=resp.iter_content(1024),total=content_size,unit='k',desc=name):
            # for data in resp.iter_content(1024):
            f.write(resp.content)

    def unzip(self):
        fdir = this_root + 'GLOBSWE/download/'
        outdir = this_root + 'GLOBSWE/nc/'
        analysis.Tools().mk_dir(outdir)
        for f in os.listdir(fdir):
            if f.endswith('.gz'):
                # print f
                try:
                    f_gzip = gzip.GzipFile(fdir + f, "rb")
                    content = f_gzip.read()
                    with open(outdir + f.split('.')[0] + '.nc', 'wb') as fw:
                        fw.write(content)
                except:
                    pass

    def nc_to_tif(self,data='SWE_avg'):
        # data = 'SWE_avg'
        # data = 'SWE_max'
        # 可行
        ncdir = this_root + 'GLOBSWE/nc/'
        out_dir = this_root + 'GLOBSWE/nc_to_tif/' + data + '/'
        analysis.Tools().mk_dir(out_dir, force=True)
        for f in os.listdir(ncdir):
            # print f
            nc = ncdir + f
            ncin = Dataset(nc, 'r')
            x = np.array(ncin['x'])
            y = np.array(ncin['y'])
            swe_avg = np.array(ncin[data])
            swe_avg = np.array(swe_avg)
            longitude_start, latitude_start, pixelWidth, pixelHeight = x[0], y[0], x[1] - x[0], y[1] - y[0]
            fname = f.split('.')[0] + '.tif'
            to_raster.array2raster_polar(out_dir + fname, longitude_start, latitude_start, pixelWidth, pixelHeight,
                                         swe_avg, ndv=-1)
            # plt.imshow(swe_avg)
            # plt.show()
            # exit()
            pass

    def hdf_to_tif(self,data='swe_average'):
        # data = 'swe_maximum'
        # data = 'swe_average'

        # 获取nc空间坐标信息
        nc = r'D:\project05\GLOBSWE\nc\198203.nc'
        ncin = Dataset(nc, 'r')
        x = np.array(ncin['x'])
        y = np.array(ncin['y'])
        longitude_start, latitude_start, pixelWidth, pixelHeight = x[0], y[0], x[1] - x[0], y[1] - y[0]
        hdf_dir = this_root + 'GLOBSWE/download/'
        out_dir = this_root + 'GLOBSWE/hdf_to_tif/' + data + '/'
        analysis.Tools().mk_dir(out_dir, force=True)
        for f in os.listdir(hdf_dir):
            if f.endswith('.hdf'):
                # print f
                fname = hdf_dir + f
                hdf = SD(fname)
                # print hdf.datasets()
                # swe = np.array(hdf.select(data)['fakeDim2'])
                swe = hdf.select(data)[:]
                # swe = np.ma.masked_where(swe == -1,swe)
                # plt.imshow(swe)
                # plt.show()
                # attr = hdf.attributes(full=1)
                # attNames = attr.keys()
                # attNames.sort()
                # print attNames
                # print attr['Spatial Resolution ']
                out_fname = f.split('_')[-2] + '.tif'
                to_raster.array2raster_polar(out_dir + out_fname, longitude_start, latitude_start, pixelWidth,
                                             pixelHeight, swe, ndv=-1)

        pass

    def proj_trans1(self):
        # 出现花纹

        grid = []
        for i in range(721):
            temp = []
            for j in range(1442):
                temp.append(-999999)
            grid.append(temp)
        grid = np.array(grid)
        # plt.imshow(grid)
        # plt.show()

        nc = r'D:\project05\GLOBSWE\download\198412\198412.nc'
        ncin = Dataset(nc, 'r')

        lat = np.array(ncin['lat'])
        lon = np.array(ncin['lon'])

        swe_avg = np.array(ncin['SWE_avg'])
        x_list = []
        y_list = []
        val_list = []
        for i in tqdm(range(len(swe_avg))):
            for j in range(len(swe_avg[0])):
                lon_ = lon[i][j] + 180
                lat_ = lat[i][j] + 90
                val = swe_avg[i][j]

                if lon_ < -999 and lat_ < -999:
                    continue
                # print lon_,lat_,val
                # x = int(round(lon_ / 0.25,0))
                # y = int(round(lat_ / 0.25,0))
                x_list.append(lon_)
                y_list.append(lat_)
                val_list.append(val)
                # print x,y
                # try:
                # grid[y][x] = val
                # except:
                #     pass
                # pass
        # print len(x_list)
        # print len(set(x_list))
        exit()
        grid = np.ma.masked_where(grid < 0, grid)
        grid = grid[::-1]
        plt.imshow(grid)
        plt.show()

    def data_transform(self):
        # 不可并行，内存不足
        mode = 'SWE_max'
        # mode = 'SWE_avg'
        fdir = this_root+'GLOBSWE/tif/'+mode+'/'
        outdir = this_root+'GLOBSWE/per_pix/'+mode+'/'
        analysis.Tools().mk_dir(outdir,force=True)
        # 将空间图转换为数组
        # per_pix_data
        flist = os.listdir(fdir)
        date_list = []
        for y in range(1982,2016):
            for mon in range(1,13):
                if mon in range(5,10):
                    continue
                date_list.append('{}{}'.format(y,'%02d'%mon))
        # for i in date_list:
        #     print i
        # exit()
        all_array = []
        for d in tqdm(date_list, 'loading...'):

            for f in flist:
                # if not d in f:
                #     continue
                if f.endswith('.tif'):
                    if f.split('.')[0] == d:
                        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir + f)
                        all_array.append(array)
        # print len(all_array)
        # exit()
        row = len(all_array[0])
        col = len(all_array[0][0])

        void_dic = {}
        void_dic_list = []
        for r in range(row):
            for c in range(col):
                void_dic['%03d.%03d' % (r, c)] = []
                void_dic_list.append('%03d.%03d' % (r, c))

        # print(len(void_dic))
        # exit()
        for r in tqdm(range(row), 'transforming...'):
            for c in range(col):
                for arr in all_array:
                    val = arr[r][c]
                    void_dic['%03d.%03d' % (r, c)].append(val)

        # for i in void_dic_list:
        #     print(i)
        # exit()
        flag = 0
        temp_dic = {}
        for key in tqdm(void_dic_list, 'saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            temp_dic[key] = void_dic[key]
            if flag % 10000 == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

    def insert_nan_vals(self,vals):
        # input [1,2,3,4,10,11,12,1,2,3,4]
        # output [1,2,3,4,0,0,0,0,0,10,11,12,1,2,3,4,0,0,0,0,0]
        for i in range(238 / 7):
            for j in range(5):
                vals.insert(4 + i * 12, 0)
        return vals

    def insert_per_pix(self):
        mode = 'SWE_avg'
        # mode = 'SWE_max'
        fdir = this_root+'GLOBSWE/per_pix/'+mode+'/'
        out_dir = this_root+'GLOBSWE/per_pix/'+mode+'_408/'
        analysis.Tools().mk_dir(out_dir)
        # spatial_dic = {}

        for f in tqdm(os.listdir(fdir)):
            dic = dict(np.load(fdir+f).item())
            data_dic = {}
            for pix in dic:
                val = dic[pix]
                if val[0] > 0:
                    val = self.insert_nan_vals(val)
                    data_dic[pix] = val

                    # plt.plot(val)
                    # print len(val)
                    # # print f
                    # plt.show()
                else:
                    data_dic[pix] = [-999999]*408
        # arr = analysis.DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.show()
            np.save(out_dir+f,data_dic)

def main():

    pre_poecess_daily()
    pass

if __name__ == '__main__':
    main()