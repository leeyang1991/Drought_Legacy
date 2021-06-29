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



class Soilgrids:
    '''
    High IO required
    Higher is better
    '''
    def __init__(self):
        pass


    def run(self):

        father_dir = '/Volumes/SSD/soil_test/'

        # step 1: check all tif
        # params = []
        # for fdir in os.listdir(father_dir):
        #     if fdir.startswith('.'):
        #         continue
        #     params.append(father_dir + fdir + '/')
        # MULTIPROCESS(self.check_tifs,params).run(process=18)

        # step 2: re-download invalid tifs
        # params = []
        # for fdir in os.listdir(father_dir):
        #     if fdir.startswith('.'):
        #         continue
        #     invalid_tif = father_dir + fdir + '/invalid_tif.txt'
        #     tiles_dir = father_dir + fdir + '/tifs/'
        #     params.append([invalid_tif, tiles_dir])
        # MULTIPROCESS(self.download_invalid_tiles,params).run()

        # step 3: mosaic tiles to tifs
        #           and mosaic tifs to global
        # for fdir in sorted(os.listdir(father_dir)):
        #     print(fdir)
        #     if fdir.startswith('.'):
        #         continue
        #     fdir = father_dir + fdir + '/'
        #     step1_dir = fdir + 'step1/'
        #     step2_dir = fdir + 'step2/'
        #     self.mosaic_step1(fdir,step1_dir)
        #     self.mosaic_step2(step1_dir,step2_dir)

        # step 4: re-projection and resample
        # for fdir in tqdm(sorted(os.listdir(father_dir))):
        #     if fdir.startswith('.'):
        #         continue
        #     fdir_i = father_dir + fdir + '/'
        #     tif = fdir_i + 'step2/global_5000m.tif'
        #     res = 0.5
        #     outtif = fdir_i + 'step2/global_5000m_84_{}.tif'.format(res)
        #     self.re_projection(tif,outtif,res=res)

        # step5 unify raster
        # for fdir in tqdm(sorted(os.listdir(father_dir))):
        #     if fdir.startswith('.'):
        #         continue
        #     fdir_i = father_dir + fdir + '/'
        #     res = 0.5
        #     tif = fdir_i + 'step2/global_5000m_84_{}.tif'.format(res)
        #     outtif = fdir_i + 'step2/global_5000m_84_{}_unify.tif'.format(res)
        #     self.unify_raster(tif,outtif)
            # exit()


        # self.copy_result(father_dir)

        pass

    def kernel_check_tif(self,tif):
        try:
            arr = to_raster.raster2array(tif)[0]
            a=(len(arr))
            return True
        except:
            return False
        # if arr:
        #     return True
        # else:
        #     return False

    def check_tifs(self,fdir):
        # fdir = '/Volumes/SSD/drought_legacy_new/data/Soilgrids/tiles/'
        root_fdir = copy.copy(fdir)
        fdir = fdir + 'tifs/'
        invalid_f_list = []
        all_valid = True
        for folder in tqdm(os.listdir(fdir),desc='checking tiles '+fdir):
            if folder.startswith('.'):
                continue
            for f in os.listdir(os.path.join(fdir,folder)):
                if f.startswith('.'):
                    continue
                if not f.endswith('.tif'):
                    continue
                is_ok = self.kernel_check_tif(os.path.join(fdir,folder,f))
                if not is_ok:
                    all_valid = False
                    invalid_f_list.append(os.path.join(fdir,folder,f))
        fw = open(root_fdir+'invalid_tif.txt','w')
        for i in invalid_f_list:
            fw.write(i+'\n')
        fw.close()
        return all_valid
        pass


    def mosaic_tiles_i(self, params):
        from osgeo.utils import gdal_merge
        in_dir, out_tif = params
        if os.path.isfile(out_tif):
            return None
        in_dir = in_dir + '/'
        # forked from https://www.neonscience.org/merge-lidar-geotiff-py
        # GDAL mosaic
        anaconda_python_path = r'python3 '
        # gdal_script = r'/Library/Python/3.8/site-packages/GDAL-3.2.2-py3.8-macosx-10.14.6-x86_64.egg/EGG-INFO/scripts/gdal_merge.py'
        gdal_script = r'/usr/local/Cellar/gdal/3.2.2/bin/gdal_merge.py' # homebrew directory
        # files_to_mosaic = glob.glob('{}/*.tif'.format(in_dir))
        files_to_mosaic = []
        for f in os.listdir(in_dir):
            if f.startswith('.'):
                continue
            if not f.endswith('.tif'):
                continue
            files_to_mosaic.append(in_dir + f)
        # print(files_to_mosaic)
        # exit()
        if len(files_to_mosaic) == 0:
            return None
        files_string = " ".join(files_to_mosaic)
        command = "{} {} -o {} -of gtiff -ot Int16 ".format(anaconda_python_path, gdal_script, out_tif) + files_string
        print(command)
        os.system(command)

    def mosaic_tiles_i_step2(self, params):
        # from osgeo_utils import gdal_merge
        in_dir, out_tif, res = params
        in_dir = in_dir + '/'
        # forked from https://www.neonscience.org/merge-lidar-geotiff-py
        # GDAL mosaic
        anaconda_python_path = r'python3 '
        # gdal_script = r'/Library/Python/3.8/site-packages/GDAL-3.2.2-py3.8-macosx-10.14.6-x86_64.egg/EGG-INFO/scripts/gdal_merge.py'
        gdal_script = r'/Users/liyang/miniforge3/lib/python3.9/site-packages/osgeo_utils/gdal_merge.py' # miniforge gdal_merge script
        # files_to_mosaic = glob.glob('{}/*.tif'.format(in_dir))
        files_to_mosaic = []
        for f in os.listdir(in_dir):
            if f.startswith('.'):
                continue
            if not f.endswith('.tif'):
                continue
            files_to_mosaic.append(in_dir + f)
        # print(files_to_mosaic)
        # exit()
        files_string = " ".join(files_to_mosaic)
        command = "{} {} -o {} -of gtiff -ot Int16 -ps {} {} ".format(anaconda_python_path, gdal_script, out_tif, res, res) + files_string
        print(command)
        os.system(command)


    def mosaic_step1(self,fdir,outdir):
        # fdir = data_root + 'Soilgrids/tiles/'
        # outdir = data_root + 'Soilgrids/mosaic_step1/'
        T.mk_dir(outdir)
        params = []
        fdir = fdir + 'tifs/'
        for folder in tqdm(os.listdir(fdir)):
            if folder.startswith('.'):
                continue
            params.append([fdir + folder,outdir + folder + '.tif'])
            # self.mosaic_tiles_i([fdir + folder,outdir + folder + '.tif'])
        MULTIPROCESS(self.mosaic_tiles_i,params).run()

    def mosaic_step2(self,step1_dir,outdir):
        # fdir = data_root + 'Soilgrids/mosaic_step1/'
        # outdir = data_root + 'Soilgrids/mosaic_step2/'
        T.mk_dir(outdir)
        # res = 5000 # unit meters
        res = 250 # unit meters
        outtif = outdir+'global_{}m.tif'.format(res)
        if os.path.isfile(outtif):
            return None
        self.mosaic_tiles_i_step2([step1_dir,outtif,res])

        pass

    def download_invalid_tiles(self,params):
        # invalid_tiles_txt = '/Volumes/SSD/drought_legacy_new/data/Soilgrids/invalid_tiles.txt'
        # outdir = '/Volumes/SSD/drought_legacy_new/data/Soilgrids/tiles/'
        invalid_tiles_txt, outdir = params
        fr = open(invalid_tiles_txt,'r')
        lines = fr.readlines()
        for line in tqdm(lines,desc='re-downloading...'):
            # print(line)
            line = line.split('\n')[0]
            line_split = line.split('/')
            tif_name = line_split[-1]
            tile_folder = line_split[-2]
            p = line_split[-4].split('_')[0]
            p1 = line_split[-4]
            url = 'https://files.isric.org/soilgrids/latest/data/{}/{}/{}/{}'.format(p,p1,tile_folder,tif_name)
            # print(url)
            # exit()
            outdir_i = outdir + tile_folder + '/'
            # print(outdir_i)
            self.download_i_overwrite(url,outdir_i)
        pass


    def download_i_overwrite(self,url,outdir_i):

        # fname = url.split('/')[-1]
        # req = requests.request('GET',url)
        # content = req.content
        # fw = open(outdir_i + fname,'wb')
        # fw.write(content)
        # fw.close()

        #################
        attempt = 0
        while 1:
            try:
                fname = url.split('/')[-1]
                # if os.path.isfile(outdir_i + fname):
                #     print(outdir_i + fname,' is existed')
                #     return None
                req = requests.request('GET', url)
                content = req.content
                fw = open(outdir_i + fname, 'wb')
                fw.write(content)
                fw.close()
                return None

            except Exception as e:
                print(url, 'error sleep 5s')
                time.sleep(5)
                attempt += 1
            if attempt >= 10:
                return None

    def re_projection(self,tif,outtif,res=0.05):
        # tif = '/Users/wenzhang/Soilgrids/soc_60-100cm_mean/step2/global_5000m.tif'
        dataset = gdal.Open(tif)
        inRasterSRS = osr.SpatialReference()
        prj_info = '''PROJCS["Homolosine", 
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
        PROJECTION["Interrupted_Goode_Homolosine"], 
        UNIT["Meter",1]]'''
        inRasterSRS.ImportFromWkt(prj_info)
        # print(outRasterSRS.ExportToWkt())
        gdal.Warp(outtif, dataset, xRes=res, yRes=res, srcSRS=inRasterSRS, dstSRS='EPSG:4326')


    def unify_raster(self,tif,outtif,insert_value=0):
        # tif = data_root + 'landcover/glc2000_v1_1_resample.tif'
        # outtif = data_root + 'landcover/glc2000_v1_1_resample_7200_3600.tif'
        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif)
        # array[array < 0] = np.nan
        # print(np.shape(array))
        # print(originY,pixelHeight)
        # exit()
        top_line_num = abs((90. - originY) / pixelHeight)
        bottom_line_num = abs((90. + originY + pixelHeight * len(array)) / pixelHeight)
        top_line_num = int(round(top_line_num, 0))
        bottom_line_num = int(round(bottom_line_num, 0))
        # print(top_line_num)
        # print(bottom_line_num)
        # exit()
        nan_array_insert = np.ones_like(array[0]) * insert_value
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

        # insert value to column
        array_unify_1 = []
        for i in array_unify:
            n = 360./pixelWidth - len(i)
            n = int(n)
            i = list(i)
            for j in range(n):
                i.append(insert_value)
            array_unify_1.append(i)
        array_unify_1 = np.array(array_unify_1)
        # plt.imshow(array_unify)
        # plt.show()
        newRasterfn = outtif
        to_raster.array2raster(newRasterfn,-180,90,pixelWidth,pixelHeight,array_unify_1)
        pass

    def copy_result(self,father_dir):
        # father_dir = '/Volumes/Seagate_5T/Soilgrids/'
        outdir = '/Users/wenzhang/Desktop/soilgrids_0603/'
        T.mk_dir(outdir)
        for product in tqdm(T.list_dir(father_dir)):
            f = os.path.join(father_dir,product,'step2','global_5000m_84_0.5_unify.tif')
            shutil.copy(f,outdir + product + '.tif')

        pass


class Terra_climate:

    def __init__(self):

        pass

    def run(self):
        # self.nc_to_tif_pet()
        # self.nc_to_tif_precip()
        # self.nc_to_tif_vpd()
        # self.nc_to_tif_soil()
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

    def nc_to_tif_vpd(self):
        outdir = '/Volumes/SSD/drought_legacy_new/data/VPD/tif/'
        T.mk_dir(outdir,force=True)
        fdir = '/Volumes/SSD/drought_legacy_new/data/VPD/nc/'
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
                arr = ncin.variables['vpd'][i]
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
    def nc_to_tif_soil(self):
        outdir = '/Volumes/SSD/drought_legacy_new/terraclimate/soil/tif/'
        T.mk_dir(outdir,force=True)
        fdir = '/Volumes/SSD/drought_legacy_new/terraclimate/soil/nc/'
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
                arr = ncin.variables['soil'][i]
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

        # fdir = data_root + 'CWD/Precip_terra/tif/'
        # outdir = data_root + 'CWD/Precip_terra/tif_005/'

        # fdir = '/Volumes/SSD/drought_legacy_new/data/VPD/tif/'
        # outdir = '/Volumes/SSD/drought_legacy_new/data/VPD/tif_005/'

        fdir = '/Volumes/SSD/drought_legacy_new/terraclimate/soil/tif/'
        outdir = '/Volumes/SSD/drought_legacy_new/terraclimate/soil/tif_005/'

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


class Precip:

    def __init__(self):

        pass

    def run(self):
        self.per_pix()
        self.anomaly()
        pass


    def per_pix(self):
        fdir = data_root + 'Precip_terra/tif_005/'
        outdir = data_root + 'Precip_terra/per_pix/'
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
        fdir = data_root + 'Precip_terra/per_pix/'
        outdir = data_root + 'Precip_terra/per_pix_anomaly/'
        Pre_Process().cal_anomaly(fdir,outdir)
        # DIC_and_TIF(Global_vars().tif_template_7200_3600).per_pix_animate(fdir,condition='005')
        # for f in os.listdir(fdir):
        #     dic = T.load_npy(fdir + f)
        pass

class Soil_terra:

    def __init__(self):

        pass

    def run(self):
        # self.per_pix()
        self.anomaly()
        # self.check_per_pix()
        pass


    def per_pix(self):
        fdir = data_root + 'terraclimate/soil/tif_005/'
        outdir = data_root + 'terraclimate/soil/per_pix/'
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
        fdir = data_root + 'terraclimate/soil/per_pix/'
        outdir = data_root + 'terraclimate/soil/per_pix_anomaly/'
        Pre_Process().cal_anomaly(fdir,outdir)
        # DIC_and_TIF(Global_vars().tif_template_7200_3600).per_pix_animate(fdir,condition='005')
        # for f in os.listdir(fdir):
        #     dic = T.load_npy(fdir + f)
        pass

    def check_per_pix(self):
        fdir = data_root + 'terraclimate/soil/per_pix/'
        for f in T.list_dir(fdir):
            print(f)
            dic = T.load_npy(fdir + f)
            flag = 0
            matrix = []
            for pix in dic:
                vals = dic[pix]
                flag += 1
                if flag == len(vals):
                    break
                matrix.append(vals)
            plt.imshow(matrix)
            plt.show()

        pass
class PET_terra:

    def __init__(self):

        pass

    def run(self):
        # self.per_pix()
        self.anomaly()
        # self.check_per_pix()
        pass


    def per_pix(self):
        fdir = data_root + 'CWD/PET_terra/tif_005/'
        outdir = data_root + 'CWD/PET_terra/per_pix/'
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
        fdir = data_root + 'terraclimate/soil/per_pix/'
        outdir = data_root + 'terraclimate/soil/per_pix_anomaly/'
        Pre_Process().cal_anomaly(fdir,outdir)
        # DIC_and_TIF(Global_vars().tif_template_7200_3600).per_pix_animate(fdir,condition='005')
        # for f in os.listdir(fdir):
        #     dic = T.load_npy(fdir + f)
        pass

    def check_per_pix(self):
        fdir = data_root + 'terraclimate/soil/per_pix/'
        for f in T.list_dir(fdir):
            print(f)
            dic = T.load_npy(fdir + f)
            flag = 0
            matrix = []
            for pix in dic:
                vals = dic[pix]
                flag += 1
                if flag == len(vals):
                    break
                matrix.append(vals)
            plt.imshow(matrix)
            plt.show()

        pass

class VPD:
    def __init__(self):

        pass

    def run(self):

        self.per_pix()
        self.anomaly()

    def per_pix(self):
        fdir = data_root + 'VPD/tif_005/'
        outdir = data_root + 'VPD/per_pix/'
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
        fdir = data_root + 'VPD/per_pix/'
        outdir = data_root + 'VPD/per_pix_anomaly/'
        Pre_Process().cal_anomaly(fdir,outdir)
        # DIC_and_TIF(Global_vars().tif_template_7200_3600).per_pix_animate(fdir,condition='005')
        # for f in os.listdir(fdir):
        #     dic = T.load_npy(fdir + f)
        pass


class Water_balance:

    def __init__(self):

        pass

    def run(self):
        self.cal_water_balance()
        pass

    def cal_water_balance(self):
        gs_mons = Global_vars().gs_mons()
        precip_dir = data_root + 'Precip_terra/per_pix/'
        pet_dir = data_root + 'CWD/PET_terra/per_pix/'
        outdir = data_root + 'Water_balance/'
        T.mk_dir(outdir)
        spatial_dic = {}
        for f in tqdm(T.list_dir(precip_dir)):
            pre_dic = T.load_npy(precip_dir + f)
            pet_dic = T.load_npy(pet_dir + f)
            for pix in pre_dic:
                pre = pre_dic[pix]
                pet = pet_dic[pix]
                pre_gs = []
                pet_gs = []
                for i in range(len(pre)):
                    mon = i % 12 + 1
                    if mon in gs_mons:
                        pre_gs.append(pre[i])
                        pet_gs.append(pet[i])
                MAP = np.mean(pre_gs)
                MA_PET = np.mean(pet_gs)
                wb = MAP / MA_PET
                spatial_dic[pix] = wb
        DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_tif(spatial_dic,outdir + 'Aridity_Index.tif')

class Plant_Strategy:
    def __init__(self):
        '''
        https://github.com/yalingliu-cu/plant-strategies
        '''
        pass

    def run(self):
        # mat_f = data_root + 'plant-strategies/Zr.mat'
        # mat_f = data_root + 'plant-strategies/Rplant.mat'
        # self.mat_to_tif(mat_f)
        # self.resample()
        self.unify_raster()
        pass


    def mat_to_tif(self,mat_f):
        # mat_f = data_root + 'plant-strategies/Zr.mat'
        loc_f = data_root + 'plant-strategies/latlon.mat'
        var = mat_f.split('/')[-1].split('.')[0]
        outtif = data_root + 'plant-strategies/{}.tif'.format(var)
        mat_f_r = scipy.io.loadmat(mat_f)
        loc_f_r = scipy.io.loadmat(loc_f)
        mat = mat_f_r[var]
        latlon = loc_f_r['latlon']
        lonlist = []
        latlist = []
        val_list = []
        for i in range(len(latlon)):
            lon = latlon[i][1]
            lat = latlon[i][0]
            val = mat[i][0]
            lonlist.append(lon)
            latlist.append(lat)
            val_list.append(val)

        DIC_and_TIF().lon_lat_val_to_tif(lonlist,latlist,val_list,outtif)


    def resample(self):
        # var = 'Rplant'
        var = 'Zr'
        in_tif = data_root + 'plant-strategies/{}.tif'.format(var)
        out_tif = data_root + 'plant-strategies/{}_005.tif'.format(var)
        dataset = gdal.Open(in_tif)
        gdal.Warp(out_tif, dataset, xRes=0.05, yRes=0.05, srcSRS='EPSG:4326', dstSRS='EPSG:4326')


    def unify_raster(self):
        # var = 'Zr_005'
        var = 'Rplant_005'
        in_tif = data_root + 'plant-strategies/{}.tif'.format(var)
        out_tif = data_root + 'plant-strategies/{}_unify.tif'.format(var)
        DIC_and_TIF().unify_raster(in_tif,out_tif)



class Isohydricity:

    def __init__(self):

        pass

    def resample(self):
        tif = data_root + 'Isohydricity/tif_all_year/ISO_Hydricity.tif'
        out_tif = data_root + 'Isohydricity/tif_all_year/ISO_Hydricity_005.tif'
        # print(tif)
        # print(out_tif)
        # exit()
        DIC_and_TIF().resample_reproj(tif,out_tif,res=0.05)

        pass


class Hydraulic_Traits_RS:
    '''
    https://github.com/YanlanLiu/VOD_hydraulics
    https://figshare.com/articles/dataset/Datasets_Global_ecosystem-scale_plant_hydraulic_traits_retrieved_using_model-data_fusion/13350713/2?file=27851388
    Code and data availability
    The maps of retrieved ensemble mean and standard deviation of plant hydraulic traits are publicly available on Figshare https://doi.org/10.6084/m9.figshare.13350713.v2 (Liu et al., 2020). The source code of the used plant hydraulic model and the model–data fusion algorithm is available at https://github.com/YanlanLiu/VOD_hydraulics (Liu et al., 2020b). All the assimilation and forcing data sets used in this study are publicly available from the referenced sources, except for the microwave-based ALEXI ET, which was obtained upon request from Thomas R. Holmes and Christopher R. Hain on 28 January 2020.
    '''
    def __init__(self):

        pass


    def run(self):
        self.nc_to_tif_P50()
        pass
    def nc_to_tif_P50(self):
        outdir = data_root + 'RemoteSensing_Traits/'
        T.mk_dir(outdir)
        var = 'P50_{}'.format(50)
        f = '/Users/liyang/Desktop/MDF_P50.nc'
        ncin = Dataset(f, 'r')
        lat = ncin['lat'][::-1]
        lon = ncin['lon']
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]
        longitude_start = lon[0]
        latitude_start = lat[0]
        print(ncin.variables)
        arr = ncin.variables[var][::-1]
        arr = np.array(arr)
        # print(arr)
        # grid = arr < 99999
        # arr[np.logical_not(grid)] = -999999
        newRasterfn = outdir + var + '.tif'
        to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
        # grid = np.ma.masked_where(grid>1000,grid)
        # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
        # plt.imshow(arr,'RdBu')
        # plt.colorbar()
        # plt.show()
        # nc_dic[date_str] = arr
        # exit()


class Koppen:

    def __init__(self):
        self.Koppen_cls()
        pass

    def run(self):

        # self.koppen_to_arr()
        self.do_reclass()
        # self.plot_reclass()
        # self.plot_koppen_landuse()
        # self.cross_koppen_landuse()
        # self.gen_cross_koppen_landuse_tif()
        # self.gen_tif_colormap()
        pass


    def Koppen_cls(self):
        self.Koppen_color_dic = {
            'Af': '8d1c21',
            'Am': 'e7161a',
            'As': 'f19596',
            'Aw': 'f8c8c9',

            'BWk': 'f1ee70',
            'BWh': 'f4c520',
            'BSk': 'c7a655',
            'BSh': 'c58a19',

            'Cfa': '113118',
            'Cfb': '114f2a',
            'Cfc': '137539',

            'Csa': '6cb92c',
            'Csb': '9bc82a',
            'Csc': 'bfd62e',

            'Cwa': 'ad6421',
            'Cwb': '916425',
            'Cwc': '583d1b',

            'Dfa': '2d112f',
            'Dfb': '5a255d',
            'Dfc': '9b3e93',
            'Dfd': 'b9177d',

            'Dsa': 'bf7cb2',
            'Dsb': 'deb3d2',
            'Dsc': 'd9c5df',
            'Dsd': 'c8c8c9',

            'Dwa': 'bdafd5',
            'Dwb': '957cac',
            'Dwc': '7f57a1',
            'Dwd': '603691',

            'EF': '688cc7',
            'ET': '87cfd9',

            'nan': '000000'
        }

        self.A = 'Af Am As Aw'.split()
        self.B = 'BWk BWh BSk BSh'.split()
        self.E = ['ET','EF']
        self.Cf = 'Cfa Cfb Cfc'.split()
        self.Csw = 'Csa Csb Csc Cwa Cwb Cwc'.split()
        self.Df = 'Dfa Dfb Dfc Dfd'.split()
        self.Dsw = 'Dsa Dsb Dsc Dwa Dwb Dwc Dwd'.split() # 没有 Dsd

        # print self.A
        # print self.B
        # print self.Cf
        # print self.Cs
        # print self.Df
        # print self.Dsw
        # print self.ET





    def plot_koppen_landuse(self):
        f = RF().this_class_arr + '/cross_koppen_landuse_pix.npy'
        outtif = self.this_class_tif+'koppen_landuse_cross.tif'
        dic = Tools().load_npy(f)

        lc_list = []
        kp_list = []
        for v,k in enumerate(dic):
            lc,kp = k.split('.')
            lc_list.append(lc)
            kp_list.append(kp)
        lc_list = set(lc_list)
        kp_list = set(kp_list)
        vk_dic = {}
        flag = 0
        for lc in ['Grasslands', 'Shrublands_Savanna', 'Forest']:
            for kp in ['AW', 'AH', 'AS', 'AR', 'TH', 'TA']:
                flag += 1
                key = lc+'.'+kp
                vk_dic[key] = flag


        spatial_dic = {}
        for key in dic:
            val = vk_dic[key]
            pixs = dic[key][0]
            for pix in pixs:
                spatial_dic[pix] = val

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().arr_to_tif(arr,outtif)
        fw = open(self.this_class_tif+'koppen_landuse_cross.txt','w')
        for i in range(20):
            for key in vk_dic:
                v = vk_dic[key]
                if v == i:
                    fw.write(str(v)+'\t'+key+'\n')
        fw.close()
        # plt.imshow(arr)
        # plt.show()



    def koppen_ASCII_to_arr(self):

        f = data_root + 'Koppen/Koeppen-Geiger-ASCII.txt'
        fr = open(f)
        fr.readline()
        lines = fr.readlines()
        fr.close()
        lon_list = []
        lat_list = []
        val_list = []
        for line in lines:
            line = line.split('\n')[0]
            lat, lon, cls = line.split()
            lon_list.append(float(lon))
            lat_list.append(float(lat))
            val_list.append(cls)

        arr_ascii = DIC_and_TIF().lon_lat_ascii_to_arr(lon_list, lat_list, val_list)
        return arr_ascii


    def reclass(self,koppen_dic,reclass_type):
        # latitude_dic = Water_balance().gen_koppen_area()
        reclass_pix = []
        for cls in reclass_type:
            pixles = koppen_dic[cls]
            for pix in pixles:
                reclass_pix.append(pix)
        return reclass_pix


    def do_reclass(self):
        outf = data_root + 'Koppen/koppen_reclass_spatial_dic'
        reclass_type = {'A':self.A,
                        'B':self.B,
                        'Cf':self.Cf,
                        'Csw':self.Csw,
                        'Df':self.Df,
                        'Dsw':self.Dsw,
                        'E':self.E}
        reclass_dic = {}
        koppen_arr = self.koppen_ASCII_to_arr()
        koppen_dic = DIC_and_TIF().spatial_arr_to_dic(koppen_arr)
        for pix in koppen_dic:
            kp = koppen_dic[pix]
            success = 0
            new_class_i = None
            for kp_reclass in reclass_type:
                kp_class_origin_list = reclass_type[kp_reclass]
                if kp in kp_class_origin_list:
                    new_class_i = kp_reclass
                    success = 1
                    break
            if success == 1:
                new_class = new_class_i
            else:
                new_class = None
            reclass_dic[pix] = new_class
        #### check ####
        # val_dic = {}
        # flag = 0
        # for i in reclass_type:
        #     flag += 1
        #     val_dic[i] = flag
        # val_dic[None] = 0
        # spatial_dic = {}
        # for pix in reclass_dic:
        #     kp = reclass_dic[pix]
        #     val = val_dic[kp]
        #     spatial_dic[pix] = val
        #
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()
        T.save_npy(reclass_dic,outf)
        return reclass_dic


    def plot_reclass(self):
        reclass_dic = self.do_reclass()
        vk_dic = {}
        for i, k in enumerate(reclass_dic):
            vk_dic[k] = i
        void_dic = DIC_and_TIF().void_spatial_dic()
        for i in vk_dic:
            pixels = reclass_dic[i]
            val = vk_dic[i]
            for pix in pixels:
                void_dic[pix] = val
        for pix in void_dic:
            if void_dic[pix] == []:
                void_dic[pix] = np.nan
        DIC_and_TIF().plot_back_ground_arr()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(void_dic)
        DIC_and_TIF().arr_to_tif(arr,self.this_class_tif+'koppen_reclass.tif')
        fw = open(self.this_class_tif+'koppen_reclass.txt','w')
        fw.write(str(vk_dic))
        fw.close()

        # palette = []
        # for i in range(len(reclass_dic)):
        #     palette.append('#' + self.Koppen_color_dic[i])
        # colors = sns.color_palette(palette)
        # cmap = mpl.colors.ListedColormap(colors)


        cmap = sns.diverging_palette(236, 0, s=99, l=50, n=len(reclass_dic), center="light")
        cmap = mpl.colors.ListedColormap(cmap)
        # color_dic = {}
        # cm = 0
        # for lc in reclass_dic:
        #     print lc, cm, cmap[cm]
        #     color_dic[lc] = cmap[cm]
        #     cm += 1

        plt.imshow(arr,cmap)
        plt.show()





    def gen_cross_koppen_landuse_tif(self):
        outtif = self.this_class_tif+'koppen_landuse.tif'
        pix_dic = self.cross_koppen_landuse()
        val_dic = {}

        regions = []
        for lc in ['Forest','Grasslands','Shrublands']:
            for kp in ['A','B','Cf','Csw','Df','Dsw','E']:
                r = lc+'.'+kp
                regions.append(r)


        for i,region in enumerate(regions):
            # print i,region
            val_dic[region] = i
        out_txt = self.this_class_tif+'koppen_landuse.csv'
        fw = open(out_txt,'w')

        landuse_dic = Landcover().glc_2000_landuse_reclass()
        koppen_dic = self.do_reclass()
        regions = []
        for lc in landuse_dic:
            for k in koppen_dic:
                region = lc+'.'+k
                regions.append(region)

        for region in regions:
            if not region in val_dic:
                continue
            val = val_dic[region]
            pix_num = len(pix_dic[region][0])
            HI_val = pix_dic[region][1]
            text = '{},{},{},{}'.format(region,val,pix_num,HI_val)
            fw.write(text+'\n')
        fw.close()
        spatial_dic = {}
        for region in pix_dic:
            pixs = pix_dic[region][0]
            # print region,len(pixs)
            for pix in pixs:
                spatial_dic[pix] = val_dic[region]

        spatial_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(spatial_arr)
        # plt.show()
        DIC_and_TIF().arr_to_tif_GDT_Byte(spatial_arr,outtif)

        pass


    def gen_tif_colormap(self):

        r = sns.color_palette("Reds_r",n_colors=7)
        g = sns.color_palette("Greens_r",n_colors=7)
        b = sns.color_palette("Blues_r",n_colors=7)
        # b = sns.light_palette("navy",n_colors=7,reverse=True)
        rgb_r = []
        rgb_g = []
        rgb_b = []
        for i in r:
            rgb_r.append(i)
        for i in g:
            rgb_g.append(i)
        for i in b:
            rgb_b.append(i)
        #     print i
        sns.palplot(r)
        sns.palplot(g)
        sns.palplot(b)
        plt.show()
        colors = rgb_g+rgb_r+rgb_b
        clr_f = self.this_class_tif+'koppen_landuse.tif.clr'
        fw = open(clr_f,'w')
        for i in range(21):
            color = colors[i]
            r,g,b = color
            r = int(r*255)
            g = int(g*255)
            b = int(b*255)
            line = '{} {} {} {}\n'.format(i,r,g,b)
            fw.write(line)
        fw.close()
        pass

def main():
    # CSIF().run()
    # SPEI_preprocess().run()
    # TWS_Water_Gap().run()
    # GRACE().run()
    # NDVI().run()
    # Climate().run()
    # SM().run()
    # Soilgrids().run()
    # Terra_climate().run()
    # CSIF_005().run()
    # Landcover().run()
    # CWD().run()
    # SPEI12().run()
    # VPD().run()
    # Precip().run()
    # Soil_terra().run()
    # PET_terra().run()
    # Water_balance().run()
    # Plant_Strategy().run()
    # step1 = '/Users/liyang/Desktop/step1/'
    # step2 = '/Users/liyang/Desktop/step2/'
    # Soilgrids().mosaic_step2(step1,step2)
    # tif = '/Users/liyang/Desktop/step2/global_250m.tif'
    # outtif = '/Users/liyang/Desktop/step2/global_250m_reproj.tif'
    # res = 0.0025
    # Soilgrids().re_projection(tif,outtif,res=res)
    # Hydraulic_Traits_RS().run()
    Koppen().run()

    pass


if __name__ == '__main__':
    main()