# coding=utf-8

from __init__ import *


class CSIF:
    def __init__(self):
        self.this_data_root = data_root + 'CSIF\\'
        pass


    def run(self):
        # self.nc_to_tif()
        # self.monthly_compose()
        # self.plot_monthly_clear()
        # self.per_pix()
        # self.clean_per_pix()
        # self.cal_per_pix_anomaly()
        # self.detrend()
        # self.check_per_pix()
        pass

    def nc_to_tif(self):
        outdir = self.this_data_root + 'tif\\clear\\'
        T.mk_dir(outdir,force=True)
        fdir = self.this_data_root + 'nc\\clear\\'
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
        fdir = self.this_data_root + 'per_pix_anomaly\\'
        outdir = self.this_data_root + 'per_pix_anomaly_detrend\\'
        Pre_Process().detrend(fdir,outdir)
        pass


    def monthly_compose(self):

        fdir = self.this_data_root + 'tif\\clear\\'
        outdir = self.this_data_root + 'tif\\monthly_clear\\'
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
        fdir = self.this_data_root + 'tif\\monthly_clear\\'
        for f in os.listdir(fdir):
            arr = to_raster.raster2array(fdir + f)[0]
            T.mask_999999_arr(arr)
            plt.imshow(arr,vmin=0,vmax=0.7)
            plt.title(f)
            plt.colorbar()
            plt.show()
        pass

    def per_pix(self):
        fdir = self.this_data_root + 'tif\\monthly_clear\\'
        outdir = self.this_data_root + 'per_pix\\'
        Pre_Process().data_transform(fdir,outdir)


    def clean_per_pix(self):
        fdir = self.this_data_root + 'per_pix\\'
        outdir = self.this_data_root + 'per_pix_clean\\'
        Pre_Process().clean_per_pix(fdir,outdir)
        pass

    def check_per_pix(self):
        fdir = self.this_data_root + 'per_pix_anomaly\\'
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
        fdir = self.this_data_root + 'per_pix_clean\\'
        outdir = self.this_data_root + 'per_pix_anomaly\\'
        Pre_Process().cal_anomaly(fdir,outdir)

        pass


class SPEI_preprocess:

    def __init__(self):
        self.this_data_root = data_root + 'SPEI\\'
        pass

    def run(self):
        # self.nc_to_tif()
        # self.tif_to_perpix()
        # self.clean_spei()
        pass

    def nc_to_tif(self):
        outdir = self.this_data_root + 'tif\\'
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
        fdir = self.this_data_root + 'tif\\'
        outdir = self.this_data_root + 'per_pix\\'
        Pre_Process().data_transform(fdir,outdir)

        pass

    def clean_spei(self):
        fdir = self.this_data_root + 'per_pix\\'
        outdir = self.this_data_root + 'per_pix_clean\\'
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
        outdir = data_root + 'TWS\\water_gap\\tif\\'
        T.mk_dir(outdir)
        nc = data_root + 'TWS\\water_gap\\watergap_22d_WFDEI-GPCC_histsoc_tws_monthly_1901_2016.nc4'
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
        fdir = data_root + 'TWS\\water_gap\\tif\\'
        outdir = data_root + 'TWS\\water_gap\\per_pix\\'
        Pre_Process().data_transform(fdir,outdir)

        pass



    def anomaly(self):
        fdir = data_root + 'TWS\\water_gap\\per_pix\\'
        outdir = data_root + 'TWS\\water_gap\\per_pix_anomaly\\'
        Pre_Process().cal_anomaly(fdir,outdir)

        pass

    def trend(self):
        outdir = temp_results_dir + 'TWS_WATER_Gap\\'
        T.mk_dir(outdir)
        outf = outdir + 'trend.tif'
        dff = Main_flow_Prepare().this_class_arr + 'prepare\\data_frame_threshold_0.df'
        df = T.load_df(dff)
        T.print_head_n(df)
        # exit()
        fdir = data_root + 'TWS\\water_gap\\per_pix\\'
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
        outdir = temp_results_dir + 'TWS_WATER_Gap\\'
        T.mk_dir(outdir)
        outf = outdir + 'anomaly_trend.tif'
        dff = Main_flow_Prepare().this_class_arr + 'prepare\\data_frame_threshold_0.df'
        df = T.load_df(dff)
        T.print_head_n(df)
        # exit()
        fdir = data_root + 'TWS\\water_gap\\per_pix_anomaly\\'
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
        self.this_data_root = data_root + 'TWS\\GRACE\\'

        pass

    def run(self):
        # self.nc_to_tif()
        # self.per_pix()
        # self.clean_per_pix()
        # self.per_pix_anomaly()

        self.check_per_pix()
        pass


    def nc_to_tif(self):
        outdir = self.this_data_root + 'tif\\'
        T.mk_dir(outdir)
        f = self.this_data_root + 'GRCTellus.JPL.200204_202011.GLO.RL06M.MSCNv02.nc'
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
        fdir = self.this_data_root + 'tif\\'
        outdir = self.this_data_root + 'per_pix\\'
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
        fdir = self.this_data_root+'per_pix\\'
        outdir = self.this_data_root+'per_pix_clean\\'

        Pre_Process().clean_per_pix(fdir,outdir)

        pass

    def check_per_pix(self):
        fdir = self.this_data_root + 'per_pix\\'
        # fdir = self.this_data_root + 'per_pix_anomaly\\'
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
        fdir = self.this_data_root + 'per_pix\\'
        outdir = self.this_data_root + 'per_pix_anomaly\\'
        Pre_Process().cal_anomaly(fdir,outdir)

        pass
def main():
    # CSIF().run()
    # SPEI_preprocess().run()
    # TWS_Water_Gap().run()
    GRACE().run()
    pass


if __name__ == '__main__':
    main()