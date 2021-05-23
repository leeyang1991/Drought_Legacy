# coding=utf-8

# from __init__ import *
from Main_flow_csif_legacy_2002 import *

class CSIF:
    def __init__(self):
        self.this_data_root = data_root + 'CSIF/'
        pass


    def run(self):
        # self.nc_to_tif()
        # self.monthly_compose()
        # self.plot_monthly_clear()
        self.per_pix()
        # self.clean_per_pix()
        self.cal_per_pix_anomaly()
        # self.detrend()
        # self.check_per_pix()
        pass

    def nc_to_tif(self):
        outdir = self.this_data_root + 'tif/clear/'
        T.mk_dir(outdir,force=True)
        fdir = self.this_data_root + 'nc/clear/'
        for fi in os.listdir(fdir):
            print(fi)
            f = fdir + fi
            year = fi.split('.')[-2]
            ncin = Dataset(f, 'r')
            # print(ncin.variables)
            # exit()
            lat = ncin['lat'][::-1]
            lon = ncin['lon']
            pixelWidth = lon[1] - lon[0]
            pixelHeight = lat[1] - lat[0]
            longitude_start = lon[0]
            latitude_start = lat[0]
            time = ncin.variables['doy']

            start = datetime.datetime(int(year), 1, 1)
            # print(start)
            flag = 0
            for i in tqdm(range(len(time))):
                # print(i)
                flag += 1
                # print(time[i])
                date = start + datetime.timedelta(days=int(time[i])-1)
                year = str(date.year)
                # exit()
                month = '%02d' % date.month
                day = '%02d'%date.day
                date_str = year + month + day
                # if not date_str[:4] in valid_year:
                #     continue
                # print(date_str)
                # exit()
                arr = ncin.variables['clear_inst_sif'][i][::-1]
                arr = np.array(arr)
                # print(arr)
                # grid = arr < 99999
                # arr[np.logical_not(grid)] = -999999
                newRasterfn = outdir + date_str + '.tif'
                # to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # grid = np.ma.masked_where(grid>1000,grid)
                DIC_and_TIF().arr_to_tif(arr,newRasterfn)
                # plt.imshow(arr,'RdBu')
                # plt.colorbar()
                # plt.show()
                # nc_dic[date_str] = arr
                # exit()

    def detrend(self):
        fdir = self.this_data_root + 'per_pix_anomaly/'
        outdir = self.this_data_root + 'per_pix_anomaly_detrend/'
        Pre_Process().detrend(fdir,outdir)
        pass


    def monthly_compose(self):

        fdir = self.this_data_root + 'tif/clear/'
        outdir = self.this_data_root + 'tif/monthly_clear/'
        T.mk_dir(outdir)
        mon_dic = {}
        for y in range(2001,2017):
            for m in range(1,13):
                date = '{}{:02d}'.format(y,m)
                mon_dic[date] = []
        for f in os.listdir(fdir):
            year_m = f[:6]
            f_path = fdir + f
            mon_dic[year_m].append(f_path)
        for date in tqdm(mon_dic):
            # arr_sum = 0.
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            for f_path in mon_dic[date]:
                arr = to_raster.raster2array(f_path)[0]
                dic = DIC_and_TIF().spatial_arr_to_dic(arr)
                for pix in dic:
                    val = dic[pix]
                    if val < -999:
                        continue
                    spatial_dic[pix].append(val)
                # arr_sum += arr
            arr_mean = DIC_and_TIF().pix_dic_to_spatial_arr_mean(spatial_dic)
            T.mask_999999_arr(arr_mean)
            outf = outdir + date + '.tif'
            DIC_and_TIF().arr_to_tif(arr_mean,outf)
            #
            # plt.imshow(arr_mean)
            # plt.title(date)
            # plt.show()


        pass

    def plot_monthly_clear(self):
        fdir = self.this_data_root + 'tif/monthly_clear/'
        for f in os.listdir(fdir):
            arr = to_raster.raster2array(fdir + f)[0]
            T.mask_999999_arr(arr)
            plt.imshow(arr,vmin=0,vmax=0.7)
            plt.title(f)
            plt.colorbar()
            plt.show()
        pass

    def per_pix(self):
        fdir = self.this_data_root + 'tif/monthly_clear/'
        outdir = self.this_data_root + 'per_pix/'
        Pre_Process().data_transform(fdir,outdir)


    def clean_per_pix(self):
        fdir = self.this_data_root + 'per_pix/'
        outdir = self.this_data_root + 'per_pix_clean/'
        Pre_Process().clean_per_pix(fdir,outdir)
        pass

    def check_per_pix(self):
        fdir = self.this_data_root + 'per_pix_anomaly/'
        dic = T.load_npy_dir(fdir,condition='015')
        for pix in dic:
            print(pix)
            vals = dic[pix]
            vals = np.array(vals)
            if vals[0] < -999:
                continue
            # T.mask_999999_arr(vals)
            plt.plot(vals)
            plt.show()
        pass

    def cal_per_pix_anomaly(self):
        fdir = self.this_data_root + 'per_pix_clean/'
        outdir = self.this_data_root + 'per_pix_anomaly/'
        Pre_Process().cal_anomaly(fdir,outdir)

        pass


class SPEI_preprocess:

    def __init__(self):
        self.this_data_root = data_root + 'SPEI/'
        pass

    def run(self):
        # self.nc_to_tif()
        self.tif_to_perpix()
        # self.clean_spei()
        pass

    def nc_to_tif(self):
        outdir = self.this_data_root + 'tif/'
        T.mk_dir(outdir)
        f = self.this_data_root + 'spei03.nc'
        ncin = Dataset(f, 'r')
        lat = ncin['lat'][::-1]
        lon = ncin['lon']
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]
        longitude_start = lon[0]
        latitude_start = lat[0]

        time = ncin.variables['time']

        # print(time)
        # exit()
        # time_bounds = ncin.variables['time_bounds']
        # print(time_bounds)
        start = datetime.datetime(1900, 1, 1)
        # print(start)
        # exit()
        # a = start + datetime.timedelta(days=5459)
        # print(a)
        # print(len(time_bounds))
        # print(len(time))
        # for i in time:
        #     print(i)
        # exit()
        # nc_dic = {}
        flag = 0

        for i in tqdm(range(len(time))):
            flag += 1
            # print(time[i])
            date = start + datetime.timedelta(days=int(time[i]))
            # print(date)
            # exit()
            year = str(date.year)
            if 2001<=int(year)<=2016:
                month = '%02d' % date.month
                # day = '%02d'%date.day
                date_str = year + month
                # if not date_str[:4] in valid_year:
                #     continue
                # print(date_str)
                arr = ncin.variables['spei'][i][::-1]
                arr = np.array(arr)
                grid = arr < 99999
                arr[np.logical_not(grid)] = -999999
                newRasterfn = outdir + date_str + '.tif'
                to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # grid = np.ma.masked_where(grid>1000,grid)
                # plt.imshow(arr,'RdBu',vmin=-3,vmax=3)
                # plt.colorbar()
                # plt.show()
                # nc_dic[date_str] = arr
                # exit()

        pass

    def tif_to_perpix(self):
        # fdir = self.this_data_root + 'tif/'
        fdir = self.this_data_root + 'tif_for_modis/'
        # outdir = self.this_data_root + 'per_pix/'
        outdir = self.this_data_root + 'per_pix_for_modis/'
        Pre_Process().data_transform(fdir,outdir)

        pass

    def clean_spei(self):
        fdir = self.this_data_root + 'per_pix/'
        outdir = self.this_data_root + 'per_pix_clean/'
        Pre_Process().clean_per_pix(fdir,outdir)
        pass

class TWS_Water_Gap:
    '''
    data download from
    https://doi.pangaea.de/10.1594/PANGAEA.918447
    '''
    def __init__(self):

        pass

    def run(self):
        # self.nc_to_tif()
        # self.data_transform()
        # self.trend()
        # self.anomaly()
        self.anomaly_trend()
        pass

    def nc_to_tif(self):
        outdir = data_root + 'TWS/water_gap/tif/'
        T.mk_dir(outdir)
        nc = data_root + 'TWS/water_gap/watergap_22d_WFDEI-GPCC_histsoc_tws_monthly_1901_2016.nc4'
        ncin = Dataset(nc, 'r')
        lat = ncin['lat'][::-1]
        lon = ncin['lon']
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[0] - lat[1]
        longitude_start = lon[0]
        latitude_start = lat[-1]

        time = ncin.variables['time']
        # print ncin.variables
        # exit()
        # print(time)
        # exit()
        # time_bounds = ncin.variables['time_bounds']
        # print(time_bounds)
        start = datetime.datetime(1901, 1, 1)
        # a = start + datetime.timedelta(days=5459)
        # print(a)
        # print(len(time_bounds))
        # print(len(time))
        # for i in time:
        #     print(i)
        # exit()
        # nc_dic = {}
        flag = 0
        valid_date_list = []
        for yyyy in range(2001,2017):
            for mm in range(1,13):
                date = '{}{:02d}'.format(yyyy,mm)
                valid_date_list.append(date)
        for i in tqdm(range(len(time))):
            flag += 1
            # print(time[i])
            # date = start + datetime.timedelta(months=int(time[i]))
            date = start + relativedelta.relativedelta(months=int(time[i]))
            year = str(date.year)
            month = '%02d' % date.month
            # day = '%02d'%date.day
            date_str = year + month
            if not date_str in valid_date_list:
                continue
            # if not date_str[:4] in valid_year:
            #     continue
            # print(date_str)
            # exit()
            arr = ncin.variables['tws'][i]
            arr = np.array(arr)
            # arr[arr < -999] = -999999
            arr[arr>999999] = -999999
            newRasterfn = outdir + date_str + '.tif'
            to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
            # grid = np.ma.masked_where(grid>1000,grid)
            # plt.imshow(arr,'RdBu',vmin=0,vmax=1000)
            # plt.colorbar()
            # plt.show()
            # nc_dic[date_str] = arr
            # exit()
        pass

    def data_transform(self):
        fdir = data_root + 'TWS/water_gap/tif/'
        outdir = data_root + 'TWS/water_gap/per_pix/'
        Pre_Process().data_transform(fdir,outdir)

        pass



    def anomaly(self):
        fdir = data_root + 'TWS/water_gap/per_pix/'
        outdir = data_root + 'TWS/water_gap/per_pix_anomaly/'
        Pre_Process().cal_anomaly(fdir,outdir)

        pass

    def trend(self):
        outdir = temp_results_dir + 'TWS_WATER_Gap/'
        T.mk_dir(outdir)
        outf = outdir + 'trend.tif'
        dff = Main_flow_Prepare().this_class_arr + 'prepare/data_frame_threshold_0.df'
        df = T.load_df(dff)
        T.print_head_n(df)
        # exit()
        fdir = data_root + 'TWS/water_gap/per_pix/'
        dic = {}
        for f in tqdm(os.listdir(fdir),desc='loading...'):
            dic_i = T.load_npy(fdir + f)
            dic.update(dic_i)

        spatial_dic = {}
        pixs = set()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            if pix in pixs:
                continue
            pixs.add(pix)

            # gs_mons = row.gs_mons
            tws = dic[pix]

            # gs_indx = []
            # for indx in range(len(tws)):
            #     mon = indx % 12 + 1
            #     if mon in gs_mons:
            #         gs_indx.append(indx)
            # gs_tws = T.pick_vals_from_1darray(tws,gs_indx)
            tws_annual = []
            # gs_tws_reshape = tws.reshape((-1,len(gs_mons)))
            gs_tws_reshape = tws.reshape((-1,12))
            plt.imshow(gs_tws_reshape)
            plt.colorbar()
            plt.show()
            for y in gs_tws_reshape:
                tws_annual.append(np.mean(y))
            k,b = T.linear_regression_1d(list(range(len(tws_annual))),tws_annual)
            spatial_dic[pix] = k
            # plt.plot(tws_annual)
            # plt.imshow(gs_tws_reshape)
            # plt.show()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().arr_to_tif(arr,outf)
        # plt.imshow(arr)
        # plt.show()


    def anomaly_trend(self):
        outdir = temp_results_dir + 'TWS_WATER_Gap/'
        T.mk_dir(outdir)
        outf = outdir + 'anomaly_trend.tif'
        dff = Main_flow_Prepare().this_class_arr + 'prepare/data_frame_threshold_0.df'
        df = T.load_df(dff)
        T.print_head_n(df)
        # exit()
        fdir = data_root + 'TWS/water_gap/per_pix_anomaly/'
        dic = {}
        for f in tqdm(os.listdir(fdir),desc='loading...'):
            dic_i = T.load_npy(fdir + f)
            dic.update(dic_i)

        spatial_dic = {}
        pixs = set()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            if pix in pixs:
                continue
            pixs.add(pix)

            # gs_mons = row.gs_mons
            tws = dic[pix]
            tws = np.array(tws)
            # gs_indx = []
            # for indx in range(len(tws)):
            #     mon = indx % 12 + 1
            #     if mon in gs_mons:
            #         gs_indx.append(indx)
            # gs_tws = T.pick_vals_from_1darray(tws,gs_indx)
            tws_annual = []
            # gs_tws_reshape = tws.reshape((-1,len(gs_mons)))
            gs_tws_reshape = tws.reshape((-1,12))
            # plt.imshow(gs_tws_reshape)
            # plt.colorbar()
            # plt.show()
            for y in gs_tws_reshape:
                tws_annual.append(np.mean(y))
            k,b = T.linear_regression_1d(list(range(len(tws_annual))),tws_annual)
            spatial_dic[pix] = k
            # plt.plot(tws_annual)
            # plt.imshow(gs_tws_reshape)
            # plt.show()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().arr_to_tif(arr,outf)
        # plt.imshow(arr)
        # plt.show()




        pass


class GRACE:

    def __init__(self):
        self.this_data_root = data_root + 'TWS/GRACE/'

        pass

    def run(self):
        self.nc_to_tif()
        # self.per_pix()
        # self.clean_per_pix()
        # self.per_pix_anomaly()

        # self.check_per_pix()
        pass


    def nc_to_tif(self):
        outdir = self.this_data_root + 'tif/'
        T.mk_dir(outdir,force=True)
        # f = self.this_data_root + 'GRCTellus.JPL.200204_202011.GLO.RL06M.MSCNv02.nc'
        f = self.this_data_root + 'GRCTellus.JPL.200204_202102.GLO.RL06M.MSCNv02CRI.nc'
        ncin = Dataset(f, 'r')
        lat = ncin['lat'][::-1]
        lon = ncin['lon']
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]
        longitude_start = -180.
        latitude_start = 90.

        time_series = ncin.variables['time']
        # print(ncin.variables)
        # print(time)
        # exit()
        # time_bounds = ncin.variables['time_bounds']
        # print(time_bounds)
        start = datetime.datetime(2002, 1, 1)
        # print(start)
        # exit()
        # a = start + datetime.timedelta(days=5459)
        # print(a)
        # print(len(time_bounds))
        # print(len(time))
        # for i in time:
        #     print(i)
        # exit()
        # nc_dic = {}
        flag = 0

        for i in tqdm(range(len(time_series))):
            flag += 1
            # print(time_series[i])
            # continue
            # exit()
            date = start + datetime.timedelta(days=int(time_series[i]))
            # print(date)
            # continue
            # exit()
            year = str(date.year)
            if 2001<=int(year)<=2016:
                month = '%02d' % date.month
                # day = '%02d'%date.day
                date_str = year + month
                # if not date_str[:4] in valid_year:
                #     continue
                # print(date_str)
                arr = ncin.variables['lwe_thickness'][i][::-1]
                arr = np.array(arr)
                arr_new = []
                for i in arr:
                    temp1 = i[int(len(i) / 2):]
                    temp2 = i[:int(len(i) / 2)]
                    add_temp = list(temp1) + list(temp2)
                    arr_new.append(add_temp)
                arr = np.array(arr_new)
                grid = arr < 99999
                arr[np.logical_not(grid)] = -999999
                newRasterfn = outdir + date_str + '.tif'
                to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # grid = np.ma.masked_where(grid>1000,grid)
                # plt.imshow(arr,'RdBu',vmin=-30,vmax=30)
                # plt.colorbar()
                # plt.show()
                # nc_dic[date_str] = arr
                # exit()

        pass

    def per_pix(self):
        fdir = self.this_data_root + 'tif/'
        outdir = self.this_data_root + 'per_pix/'
        f_list = []
        for y in range(2001,2017):
            for m in range(1,13):
                date = '{}{:02d}.tif'.format(y,m)
                # print(date)
                f_list.append(date)
        # print(len(f_list))
        Pre_Process().data_transform_with_date_list(fdir,outdir,f_list)

        pass


    def clean_per_pix(self):
        fdir = self.this_data_root+'per_pix/'
        outdir = self.this_data_root+'per_pix_clean/'

        Pre_Process().clean_per_pix(fdir,outdir)

        pass

    def check_per_pix(self):
        fdir = self.this_data_root + 'per_pix/'
        # fdir = self.this_data_root + 'per_pix_anomaly/'
        for f in os.listdir(fdir):
            if not '015' in f:
                continue
            dic = T.load_npy(fdir + f)
            for pix in dic:
                vals = dic[pix]
                vals = np.array(vals)
                T.mask_999999_arr(vals)
                plt.plot(vals)
                plt.title(str(pix))
                plt.show()


        pass

    def per_pix_anomaly(self):
        fdir = self.this_data_root + 'per_pix/'
        outdir = self.this_data_root + 'per_pix_anomaly/'
        Pre_Process().cal_anomaly(fdir,outdir)

        pass


class NDVI:

    def __init__(self):
        self.this_class_root = data_root + 'NDVI/'
        pass

    def run(self):
        # self.data_transform()
        # self.anomaly()
        # self.data_clean()
        # self.detrend_anomaly()
        self.trans_168_to_180()
        # self.check_detrend()
        pass


    def data_transform(self):
        fdir = self.this_class_root + 'MOD13C1.NDVI_MVC/'
        outdir = self.this_class_root + 'per_pix/'
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir,outdir)




    def anomaly(self):
        fdir = self.this_class_root + 'per_pix/'
        outdir = self.this_class_root + 'per_pix_anomaly/'
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

    def data_clean(self):
        fdir = self.this_class_root + 'per_pix_anomaly/'
        outdir = self.this_class_root + 'per_pix_anomaly_clean/'
        Pre_Process().clean_per_pix(fdir,outdir)
        pass

    def detrend_anomaly(self):
        fdir = self.this_class_root + 'per_pix_anomaly_clean/'
        outdir = self.this_class_root + 'per_pix_anomaly_clean_detrend/'
        T.mk_dir(outdir)
        for f in tqdm(os.listdir(fdir)):
            dic_i = T.load_npy(fdir + f)
            detrend_dic_i = T.detrend_dic(dic_i)
            np.save(outdir + f,detrend_dic_i)

        pass

    def trans_168_to_180(self):

        SIF_dir = data_root + 'NDVI/per_pix_anomaly_clean_detrend/'
        outdir = data_root + 'NDVI/per_pix_anomaly_clean_detrend_180/'
        T.mk_dir(outdir)
        sif_dic = T.load_npy_dir(SIF_dir)
        sif_dic_new = {}

        for pix in sif_dic:
            vals = sif_dic[pix]
            inserted_vals = [np.nan] * 12
            vals_new = np.insert(vals, 0, inserted_vals)
            sif_dic_new[pix] = vals_new
        np.save(outdir + 'per_pix_dic',sif_dic_new)

    def check_detrend(self):
        # fdir = self.this_class_root + 'per_pix_anomaly_detrend/'
        fdir = self.this_class_root + 'per_pix_anomaly_clean_detrend/'
        dic = T.load_npy_dir(fdir)

        spatial_dic = {}
        for pix in dic:
            val_len = len(dic[pix])
            spatial_dic[pix] = val_len

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr,cmap='jet')
        plt.colorbar()
        plt.show()

        pass


class Climate:

    def __init__(self):

        pass

    def run(self):
        # self.climate_408_to_180()
        self.anomaly()
        pass


    def climate_408_to_180(self):
        fdir = data_root + 'Climate_408/'
        outdir = data_root + 'Climate_180/'
        T.mk_dir(outdir)
        for var in os.listdir(fdir):
            if var.startswith('.'):
                continue
            print(var)
            outdir_i = outdir + var + '/per_pix/'
            T.mk_dir(outdir_i,force=True)
            fdir_i = os.path.join(fdir,var,'per_pix_clean')
            dic_i = T.load_npy_dir(fdir_i)
            # DIC_and_TIF().per_pix_animate(fdir_i)
            dic_i_new = {}
            for pix in dic_i:
                val = dic_i[pix]
                val_new = val[-180:]
                val_new = np.array(val_new)
                dic_i_new[pix] = val_new
            np.save(outdir_i+'per_pix_dic.npy',dic_i_new)


    def anomaly(self):
        fdir = data_root + 'Climate_180/'

        for var in os.listdir(fdir):
            if var.startswith('.'):
                continue
            print(var)
            fdir_i = data_root + 'Climate_180/{}/per_pix/'.format(var)
            outdir = data_root + 'Climate_180/{}/per_pix_anomaly/'.format(var)
            T.mk_dir(outdir)
            Pre_Process().cal_anomaly(fdir_i,outdir)
        pass



class SM:

    def __init__(self):

        pass

    def run(self):
        # self.sm_408_to_180()
        # self.anomaly()
        self.check_anomaly()
        pass

    def sm_408_to_180(self):
        fdir = data_root + 'SM/per_pix_clean/'
        outdir = data_root + 'SM/per_pix_clean_180/'
        T.mk_dir(outdir, force=True)
        dic = T.load_npy_dir(fdir)
        # DIC_and_TIF().per_pix_animate(fdir_i)
        dic_i_new = {}
        for pix in tqdm(dic):
            val = dic[pix]
            val_new = val[-180:]
            val_new = np.array(val_new)
            dic_i_new[pix] = val_new
        np.save(outdir + 'per_pix_dic.npy', dic_i_new)
        pass

    def anomaly(self):
        fdir = data_root + 'SM/per_pix_clean_180/'
        outdir = data_root + 'SM/per_pix_clean_anomaly_180/'
        Pre_Process().cal_anomaly(fdir,outdir)


    def check_anomaly(self):
        fdir = data_root + 'SM/per_pix_clean_anomaly_180/'
        DIC_and_TIF().per_pix_animate(fdir)



class Total_Nitrogen:

    def __init__(self):

        pass


    def run(self):
        # self.mosaic_tiles()
        # self.mosaic_all()
        # self.copy_tiles_to_one_folder()
        pass

    def mosaic_tiles_i(self, params):
        in_dir, out_tif = params
        # forked from https://www.neonscience.org/merge-lidar-geotiff-py
        # GDAL mosaic
        anaconda_python_path = r'python3 '
        # gdal_script = r'/Library/Python/3.8/site-packages/GDAL-3.2.2-py3.8-macosx-10.14.6-x86_64.egg/EGG-INFO/scripts/gdal_merge.py'
        gdal_script = r'/usr/local/Cellar/gdal/3.2.2/bin/gdal_merge.py' # homebrew directory
        files_to_mosaic = glob.glob('{}/*.tif'.format(in_dir))
        files_string = " ".join(files_to_mosaic)
        # print(files_string)
        command = "{} {} -o {} -of gtiff ".format(anaconda_python_path, gdal_script, out_tif) + files_string
        print(command)
        os.system(command)


    def mosaic_tiles(self):
        fdir = data_root + 'Soilgrids/tiles/'
        outdir = data_root + 'Soilgrids/mosaic_step1/'
        T.mk_dir(outdir)
        params = []
        for folder in tqdm(os.listdir(fdir)):
            params.append([fdir + folder,outdir + folder + '.tif'])
            # self.mosaic_tiles_i()
        MULTIPROCESS(self.mosaic_tiles_i,params).run()

    def mosaic_all(self):
        fdir = data_root + 'Soilgrids/mosaic_step1/'
        outdir = data_root + 'Soilgrids/mosaic_step2/'
        T.mk_dir(outdir)
        self.mosaic_tiles_i([fdir,outdir+'Nitrogen_0_5_cm.tif'])

        pass


class Terra_climate:

    def __init__(self):

        pass

    def run(self):
        # self.nc_to_tif_pet()
        # self.nc_to_tif_precip()
        self.resample()
        pass

    def Warp_test(self):
        tif = '/Users/wenzhang/project/Drought_legacy_new/data//landcover/glc2000_v1_1.tif'
        outtif = tif + 'gdal.tif'
        dataset = gdal.Open(tif)
        gdal.Warp(outtif, dataset, xRes=0.5, yRes=0.5, srcSRS='EPSG:4326',dstSRS='EPSG:4326')
        pass

    def nc_to_tif_pet(self):
        outdir = '/Users/wenzhang/project/drought_legacy_new/data/CWD/PET_terra/tif/'
        T.mk_dir(outdir,force=True)
        fdir = '/Users/wenzhang/project/drought_legacy_new/data/CWD/PET_terra/nc/'
        for fi in os.listdir(fdir):
            print(fi)
            f = fdir + fi
            year = fi.split('.')[-2].split('_')[-1]
            # print(year)
            # exit()
            ncin = Dataset(f, 'r')
            # print(ncin.variables)
            # exit()
            lat = ncin['lat']
            lon = ncin['lon']
            pixelWidth = lon[1] - lon[0]
            pixelHeight = lat[1] - lat[0]
            longitude_start = lon[0]
            latitude_start = lat[0]
            time = ncin.variables['time']

            start = datetime.datetime(1900, 1, 1)
            # print(time)
            # for t in time:
            #     print(t)
            # exit()
            flag = 0
            for i in tqdm(range(len(time))):
                # print(i)
                flag += 1
                # print(time[i])
                date = start + datetime.timedelta(days=int(time[i]))
                year = str(date.year)
                # exit()
                month = '%02d' % date.month
                day = '%02d'%date.day
                date_str = year + month
                # print(date_str)
                # exit()
                # if not date_str[:4] in valid_year:
                #     continue
                # print(date_str)
                # exit()
                arr = ncin.variables['pet'][i]
                arr = np.array(arr)
                # print(arr)
                # grid = arr < 99999
                # arr[np.logical_not(grid)] = -999999
                newRasterfn = outdir + date_str + '.tif'
                to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # grid = np.ma.masked_where(grid>1000,grid)
                # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
                # plt.imshow(arr,'RdBu')
                # plt.colorbar()
                # plt.show()
                # nc_dic[date_str] = arr
                # exit()
    def nc_to_tif_precip(self):
        outdir = '/Users/wenzhang/project/drought_legacy_new/data/CWD/Precip_terra/tif/'
        T.mk_dir(outdir,force=True)
        fdir = '/Users/wenzhang/project/drought_legacy_new/data/CWD/Precip_terra/nc/'
        for fi in os.listdir(fdir):
            print(fi)
            if fi.startswith('.'):
                continue
            f = fdir + fi
            year = fi.split('.')[-2].split('_')[-1]
            # print(year)
            # exit()
            ncin = Dataset(f, 'r')
            # print(ncin.variables)
            # exit()
            lat = ncin['lat']
            lon = ncin['lon']
            pixelWidth = lon[1] - lon[0]
            pixelHeight = lat[1] - lat[0]
            longitude_start = lon[0]
            latitude_start = lat[0]
            time = ncin.variables['time']

            start = datetime.datetime(1900, 1, 1)
            # print(time)
            # for t in time:
            #     print(t)
            # exit()
            flag = 0
            for i in tqdm(range(len(time))):
                # print(i)
                flag += 1
                # print(time[i])
                date = start + datetime.timedelta(days=int(time[i]))
                year = str(date.year)
                # exit()
                month = '%02d' % date.month
                day = '%02d'%date.day
                date_str = year + month
                # print(date_str)
                # exit()
                # if not date_str[:4] in valid_year:
                #     continue
                # print(date_str)
                # exit()
                arr = ncin.variables['ppt'][i]
                arr = np.array(arr)
                # print(arr)
                # grid = arr < 99999
                # arr[np.logical_not(grid)] = -999999
                newRasterfn = outdir + date_str + '.tif'
                to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # grid = np.ma.masked_where(grid>1000,grid)
                # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
                # plt.imshow(arr,'RdBu')
                # plt.colorbar()
                # plt.show()
                # nc_dic[date_str] = arr
                # exit()


    def resample(self):
        # fdir = data_root + 'CWD/PET_terra/tif/'
        # outdir = data_root + 'CWD/PET_terra/tif_005/'

        fdir = data_root + 'CWD/Precip_terra/tif/'
        outdir = data_root + 'CWD/Precip_terra/tif_005/'
        T.mk_dir(outdir)
        for f in tqdm(os.listdir(fdir)):
            # print(f)
            tif = fdir + f
            outtif = outdir + f
            dataset = gdal.Open(tif)
            gdal.Warp(outtif, dataset, xRes=0.05, yRes=0.05, srcSRS='EPSG:4326', dstSRS='EPSG:4326')
            # exit()


class CSIF_005:
    def __init__(self):

        pass

    def run(self):
        # self.nc_to_tif()
        # self.per_pix()
        # self.anomaly()
        self.detrend()
        pass

    def nc_to_tif(self):
        fdir = data_root + 'CSIF005/nc/'
        outdir = data_root + 'CSIF005/tif/'
        T.mk_dir(outdir)
        for fi in tqdm(os.listdir(fdir)):

            # print(fi)
            if fi.startswith('.'):
                continue
            f = fdir + fi
            # print(year)
            # exit()
            ncin = Dataset(f, 'r')
            # print(ncin.variables)
            # exit()
            lat = ncin['lat']
            lon = ncin['lon']
            pixelWidth = lon[1] - lon[0]
            pixelHeight = lat[1] - lat[0]
            longitude_start = lon[0]
            latitude_start = lat[0]

            date_str = fi.split('.')[-2].split('_')[-1]
            # print(date_str)
            # exit()
            arr = ncin.variables['SIF_740_daily_corr']
            arr = np.array(arr)
            # print(arr)
            # grid = arr < 99999
            # arr[np.logical_not(grid)] = -999999
            newRasterfn = outdir + date_str + '.tif'
            to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
            # grid = np.ma.masked_where(grid>1000,grid)
            # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
            # plt.imshow(arr,'RdBu')
            # plt.colorbar()
            # plt.show()
            # nc_dic[date_str] = arr
            # exit()
        pass

    def per_pix(self):
        fdir = data_root + 'CSIF005/tif/'
        outdir = data_root + 'CSIF005/per_pix/'
        T.mk_dir(outdir)
        valid_spatial_dic_f = Landcover().forest_spatial_dic_f
        valid_spatial_dic = T.load_npy(valid_spatial_dic_f)
        template_tif = Global_vars().tif_template_7200_3600
        template_arr = to_raster.raster2array(template_tif)[0]
        row = len(template_arr)
        col = len(template_arr[0])
        arr_list = []
        for f in tqdm(sorted(os.listdir(fdir)),desc='loading data'):
            arr = to_raster.raster2array(fdir + f)[0]
            arr_list.append(arr)
        spatial_dic = {}
        for pix in valid_spatial_dic:
            spatial_dic[pix] = []
        for r in tqdm(tqdm(range(row)),desc='transforming...'):
            for c in range(col):
                pix = (r,c)
                if not pix in valid_spatial_dic:
                    continue
                for i in range(len(arr_list)):
                    val = arr_list[i][r][c]
                    spatial_dic[pix].append(val)

        flag = 0
        temp_dic = {}
        for key in tqdm(spatial_dic, 'saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            arr = spatial_dic[key]
            arr = np.array(arr)
            temp_dic[key] = arr
            if flag % 10000 == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)


    def anomaly(self):
        fdir = data_root + 'CSIF005/per_pix/'
        outdir = data_root + 'CSIF005/per_pix_anomaly/'
        Pre_Process().cal_anomaly(fdir, outdir)
        pass



    def kernel_detrend(self,params):
        fdir,f,outdir = params
        dic = T.load_npy(fdir + f)
        dic_detrend = T.detrend_dic(dic)
        T.save_npy(dic_detrend, outdir + f)
        pass

    def detrend(self):
        fdir = data_root + 'CSIF005/per_pix_anomaly/'
        outdir = data_root + 'CSIF005/per_pix_anomaly_detrend/'
        T.mk_dir(outdir)
        params = []
        for f in tqdm(os.listdir(fdir)):
            params.append([fdir,f,outdir])
        MULTIPROCESS(self.kernel_detrend,params).run()


        pass

class Landcover:

    def __init__(self):
        self.forest_pix_f = data_root + 'landcover/forests_pix.npy'
        self.forest_spatial_dic_f = data_root + 'landcover/gen_spatial_dic.npy'
        pass

    def run(self):
        # self.unify_raster()
        # self.forests_pix()
        # self.gen_spatial_dic()
        self.check_pix()
        pass

    def unify_raster(self):
        tif = data_root + 'landcover/glc2000_v1_1_resample.tif'
        outtif = data_root + 'landcover/glc2000_v1_1_resample_7200_3600.tif'
        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif)
        array[array < 0] = np.nan
        print(np.shape(array))
        print(originY,pixelHeight)
        # exit()
        top_line_num = abs((90 - originY) / pixelHeight)
        bottom_line_num = abs((90 + originY + pixelHeight * len(array)) / pixelHeight)
        top_line_num = int(round(top_line_num, 0))
        # print(top_line_num)
        # exit()
        bottom_line_num = int(round(bottom_line_num, 0))
        nan_array_insert = np.ones_like(array[0]) * -999999
        # nan_array_insert = np.ones_like(array[0])
        top_array_insert = []
        for i in range(top_line_num):
            top_array_insert.append(nan_array_insert)
        bottom_array_insert = []
        for i in range(bottom_line_num):
            bottom_array_insert.append(nan_array_insert)
        bottom_array_insert = np.array(bottom_array_insert)
        if len(top_array_insert) != 0:
            arr_temp = np.insert(array, 0, top_array_insert, 0)
        else:
            arr_temp = array
        if len(bottom_array_insert) != 0:
            array_unify = np.vstack((arr_temp, bottom_array_insert))
        else:
            array_unify = arr_temp

        # plt.imshow(array_unify)
        # plt.show()
        newRasterfn = outtif
        longitude_start, latitude_start = originX,originY
        to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array_unify)
        pass


    def forests_pix(self):
        tif = data_root + 'landcover/glc2000_v1_1_resample_7200_3600.tif'
        outf = data_root + 'landcover/forests_pix'
        dic = DIC_and_TIF(tif).spatial_tif_to_dic(tif)
        forest_val = [1,2,3,4,5]
        forest_pix_dic = {}
        for v in forest_val:
            forest_pix_dic[v] = []
        for pix in tqdm(dic):
            r,c = pix
            if r > 1800:
                continue
            if np.isnan(dic[pix]):
                continue
            val = int(dic[pix])
            if val in forest_val:
                forest_pix_dic[val].append(pix)
        np.save(outf,forest_pix_dic)

        pass


    def gen_spatial_dic(self):
        valid_dic = T.load_npy(self.forest_pix_f)
        outf = data_root + 'landcover/gen_spatial_dic'
        spatial_dic = {}
        for val in tqdm(valid_dic):
            pixs = valid_dic[val]
            for pix in pixs:
                spatial_dic[pix] = val
        np.save(outf,spatial_dic)


    def check_pix(self):
        # f = data_root + 'landcover/forests_pix.npy'
        f = data_root + 'landcover/gen_spatial_dic.npy'
        dic = T.load_npy(f)
        # spatial_dic = {}
        # for val in tqdm(dic):
        #     pixs = dic[val]
        #     for pix in pixs:
        #         # print(pix)
        #         spatial_dic[pix] = 1
        arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(dic)
        plt.imshow(arr)
        plt.show()
        pass

class CWD:
    def __init__(self):

        pass

    def run(self):
        # self.p_minus_pet()
        # self.per_pix()
        self.anomaly()
        pass

    def p_minus_pet(self):

        P_dir = data_root + 'CWD/Precip_terra/tif_005/'
        PET_dir = data_root + 'CWD/PET_terra/tif_005/'
        outdir = data_root + 'CWD/CWD/tif_005/'
        T.mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(P_dir)):
            p_arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(P_dir + f)
            pet_arr = to_raster.raster2array(PET_dir + f)[0]
            p_arr[p_arr<-99]=np.nan
            pet_arr[pet_arr<-99]=np.nan
            p_arr[p_arr > 32000] = np.nan
            pet_arr[pet_arr > 32000] = np.nan
            cwd = p_arr - pet_arr
            nan_matrix = np.isnan(cwd)
            cwd[nan_matrix] = -999999
            newRasterfn = outdir + f
            longitude_start, latitude_start = originX,originY
            to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,cwd,ndv = -999999)
            # exit()
            # plt.imshow(cwd,vmin=-100,vmax=100,cmap='jet')
            # plt.figure()
            # plt.imshow(p_arr)
            # plt.show()

    def per_pix(self):
        fdir = data_root + 'CWD/CWD/tif_005/'
        outdir = data_root + 'CWD/CWD/per_pix/'
        T.mk_dir(outdir)
        valid_spatial_dic_f = Landcover().forest_spatial_dic_f
        valid_spatial_dic = T.load_npy(valid_spatial_dic_f)
        template_tif = Global_vars().tif_template_7200_3600
        template_arr = to_raster.raster2array(template_tif)[0]
        row = len(template_arr)
        col = len(template_arr[0])
        arr_list = []
        for f in tqdm(sorted(os.listdir(fdir)),desc='loading data'):
            arr = to_raster.raster2array(fdir + f)[0]
            arr_list.append(arr)
        spatial_dic = {}
        for pix in valid_spatial_dic:
            spatial_dic[pix] = []
        for r in tqdm(tqdm(range(row)),desc='transforming...'):
            for c in range(col):
                pix = (r,c)
                if not pix in valid_spatial_dic:
                    continue
                for i in range(len(arr_list)):
                    val = arr_list[i][r][c]
                    spatial_dic[pix].append(val)

        flag = 0
        temp_dic = {}
        for key in tqdm(spatial_dic, 'saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            arr = spatial_dic[key]
            arr = np.array(arr)
            temp_dic[key] = arr
            if flag % 10000 == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)


    def anomaly(self):
        fdir = data_root + 'CWD/CWD/per_pix/'
        outdir = data_root + 'CWD/CWD/per_pix_anomaly/'
        Pre_Process().cal_anomaly(fdir,outdir)
        # DIC_and_TIF(Global_vars().tif_template_7200_3600).per_pix_animate(fdir,condition='005')
        # for f in os.listdir(fdir):
        #     dic = T.load_npy(fdir + f)
        pass

class SPEI12:

    def __init__(self):

        pass

    def run(self):
        # self.nc_to_tif()
        # self.resample_005()
        # self.per_pix()
        self.per_pix_05()
        pass

    def nc_to_tif(self):
        outdir = data_root + 'SPEI12/tif/'
        T.mk_dir(outdir)
        f = data_root + 'SPEI12/nc/spei12.nc'
        ncin = Dataset(f, 'r')
        lat = ncin['lat'][::-1]
        lon = ncin['lon']
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]
        longitude_start = lon[0]
        latitude_start = lat[0]

        time = ncin.variables['time']

        # print(time)
        # exit()
        # time_bounds = ncin.variables['time_bounds']
        # print(time_bounds)
        start = datetime.datetime(1900, 1, 1)
        # print(start)
        # exit()
        # a = start + datetime.timedelta(days=5459)
        # print(a)
        # print(len(time_bounds))
        # print(len(time))
        # for i in time:
        #     print(i)
        # exit()
        # nc_dic = {}
        flag = 0

        for i in tqdm(range(len(time))):
            flag += 1
            # print(time[i])
            date = start + datetime.timedelta(days=int(time[i]))
            # print(date)
            # exit()
            year = str(date.year)
            if 2001<=int(year)<=2019:
                month = '%02d' % date.month
                # day = '%02d'%date.day
                date_str = year + month
                # if not date_str[:4] in valid_year:
                #     continue
                # print(date_str)
                arr = ncin.variables['spei'][i][::-1]
                arr = np.array(arr)
                grid = arr < 99999
                arr[np.logical_not(grid)] = -999999
                newRasterfn = outdir + date_str + '.tif'
                to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # grid = np.ma.masked_where(grid>1000,grid)
                # plt.imshow(arr,'RdBu',vmin=-3,vmax=3)
                # plt.colorbar()
                # plt.show()
                # nc_dic[date_str] = arr
                # exit()


    def resample_005(self):
        fdir = data_root + 'SPEI12/tif/'
        outdir = data_root + 'SPEI12/tif_005/'
        T.mk_dir(outdir)
        for f in tqdm(os.listdir(fdir)):
            if f.startswith('.'):
                continue
            if not f.endswith('.tif'):
                continue
            tif = fdir + f
            outtif = outdir + f
            dataset = gdal.Open(tif)
            gdal.Warp(outtif, dataset, xRes=0.05, yRes=0.05, srcSRS='EPSG:4326', dstSRS='EPSG:4326')
            # arr = to_raster.raster2array(outtif)[0]
            # arr[arr<-999]=np.nan
            # plt.imshow(arr,vmin=-2,vmax=2)
            # plt.show()
            # exit()
        pass


    def per_pix(self):
        fdir = data_root + 'SPEI12/tif_005/'
        outdir = data_root + 'SPEI12/per_pix/'
        T.mk_dir(outdir)
        valid_spatial_dic_f = Landcover().forest_spatial_dic_f
        valid_spatial_dic = T.load_npy(valid_spatial_dic_f)
        template_tif = Global_vars().tif_template_7200_3600
        template_arr = to_raster.raster2array(template_tif)[0]
        row = len(template_arr)
        col = len(template_arr[0])
        arr_list = []
        for f in tqdm(sorted(os.listdir(fdir)),desc='loading data'):
            arr = to_raster.raster2array(fdir + f)[0]
            arr_list.append(arr)
        spatial_dic = {}
        for pix in valid_spatial_dic:
            spatial_dic[pix] = []
        for r in tqdm(tqdm(range(row)),desc='transforming...'):
            for c in range(col):
                pix = (r,c)
                if not pix in valid_spatial_dic:
                    continue
                for i in range(len(arr_list)):
                    val = arr_list[i][r][c]
                    spatial_dic[pix].append(val)

        flag = 0
        temp_dic = {}
        for key in tqdm(spatial_dic, 'saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            arr = spatial_dic[key]
            arr = np.array(arr)
            temp_dic[key] = arr
            if flag % 10000 == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

    def per_pix_05(self):
        fdir = data_root + 'SPEI12/tif/'
        outdir = data_root + 'SPEI12/per_pix_05/'
        T.mk_dir(outdir)
        template_arr = to_raster.raster2array(DIC_and_TIF().tif_template)[0]
        row = len(template_arr)
        col = len(template_arr[0])
        arr_list = []
        for f in tqdm(sorted(os.listdir(fdir)),desc='loading data'):
            if not f.endswith('.tif'):
                continue
            arr = to_raster.raster2array(fdir + f)[0]
            arr_list.append(arr)
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for r in tqdm(tqdm(range(row)),desc='transforming...'):
            for c in range(col):
                pix = (r,c)
                for i in range(len(arr_list)):
                    val = arr_list[i][r][c]
                    spatial_dic[pix].append(val)

        flag = 0
        temp_dic = {}
        for key in tqdm(spatial_dic, 'saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            arr = spatial_dic[key]
            arr = np.array(arr)
            temp_dic[key] = arr
            if flag % 10000 == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

def main():
    # CSIF().run()
    # SPEI_preprocess().run()
    # TWS_Water_Gap().run()
    # GRACE().run()
    # NDVI().run()
    # Climate().run()
    # SM().run()
    # Total_Nitrogen().run()
    # Terra_climate().run()
    # CSIF_005().run()
    # Landcover().run()
    # CWD().run()
    SPEI12().run()
    pass


if __name__ == '__main__':
    main()