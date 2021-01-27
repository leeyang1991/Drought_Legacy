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
        self.check_per_pix()
        pass
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
        # self.foo()

        # self.clean_spei()
        self.do_pick()
        # self.events_to_df()


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

    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        key = params[1]
        threshold = params[2]
        drought_month = []
        for i, val in enumerate(pdsi):
            if val < threshold:# SPEI
                drought_month.append(i)
            else:
                drought_month.append(-99)
        # plt.plot(drought_month)
        # plt.show()
        events = []
        event_i = []
        for ii in drought_month:
            if ii > -99:
                event_i.append(ii)
            else:
                if len(event_i) > 0:
                    events.append(event_i)
                    event_i = []
                else:
                    event_i = []

        flag = 0
        events_list = []
        # 不取两个端点
        for i in events:
            # 去除两端pdsi值小于-0.5
            if 0 in i or len(pdsi) - 1 in i:
                continue
            new_i = []
            for jj in i:
                new_i.append(jj)
            # print(new_i)
            # exit()
            flag += 1
            vals = []
            for j in new_i:
                try:
                    vals.append(pdsi[j])
                except:
                    print(j)
                    print('error')
                    print(new_i)
                    exit()
            # print(vals)

            # if 0 in new_i:
            # SPEI
            min_val = min(vals)
            if min_val < -99999:
                continue
            if min_val < threshold:
                level = 4
            # if -1 <= min_val < -.5:
            #     level = 1
            # elif -1.5 <= min_val < -1.:
            #     level = 2
            # elif -2 <= min_val < -1.5:
            #     level = 3
            # elif min_val <= -2.:
            #     level = 4
            else:
                level = 0

            events_list.append([level, new_i])
            # print(min_val)
            # plt.plot(vals)
            # plt.show()
        # for key in events_dic:
        #     # print key,events_dic[key]
        #     if 0 in events_dic[key][1]:
        #         print(events_dic[key])
        # exit()
        return events_list, key

    def pick_events(self,f, outdir):
        # 前n个月和后n个月无极端干旱事件
        fname = f.split('.')[0].split('_')[-1]
        # print(fname)
        # exit()
        n = 24.
        T.mk_dir(outdir,force=True)
        single_event_dic = {}
        dic = T.load_npy(f)
        for pix in tqdm(dic,desc='picking {}'.format(f)):
            vals = dic[pix]
            # print list(vals)
            # f = '{}_{}.txt'.format(pix[0],pix[1])
            # fw = open(f,'w')
            # fw.write(str(list(vals)))
            # fw.close()
            # pause()
            # mean = np.mean(vals)
            # std = np.std(vals)
            # threshold = mean - 2 * std
            threshold = -2.
            # threshold = np.quantile(vals, 0.05)
            event_list,key = self.kernel_find_drought_period([vals,pix,threshold])
            if len(event_list) == 0:
                continue
            events_4 = []
            for i in event_list:
                level,drought_range = i
                events_4.append(drought_range)

            single_event = []
            for i in range(len(events_4)):
                if i - 1 < 0:  # 首次事件
                    if events_4[i][0] - n < 0 or events_4[i][-1] + n >= len(vals):  # 触及两边则忽略
                        continue
                    if len(events_4) == 1:
                        single_event.append(events_4[i])
                    elif events_4[i][-1] + n <= events_4[i + 1][0]:
                        single_event.append(events_4[i])
                    continue

                # 最后一次事件
                if i + 1 >= len(events_4):
                    if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= len(vals):
                        single_event.append(events_4[i])
                    break

                # 中间事件
                if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= events_4[i + 1][0]:
                    single_event.append(events_4[i])
            # print single_event
            # sleep(0.1)
            single_event_dic[pix] = single_event
            # for evt in single_event:
            #     picked_vals = T.pick_vals_from_1darray(vals,evt)
            #     plt.scatter(evt,picked_vals,c='r')
            # plt.plot(vals)
            # plt.show()
        np.save(outdir + 'single_events_{}'.format(fname),single_event_dic)
        # spatial_dic = {}
        # for pix in single_event_dic:
        #     evt_num = len(single_event_dic[pix])
        #     if evt_num == 0:
        #         continue
        #     spatial_dic[pix] = evt_num
        # DIC_and_TIF().plot_back_ground_arr()
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()


    def do_pick(self):
        outdir = self.this_class_arr + 'drought_events\\'
        fdir = data_root + 'SPEI\\per_pix_clean\\'
        for f in os.listdir(fdir):
            self.pick_events(fdir + f,outdir)
        pass

    def events_to_df(self):
        dff = self.this_class_arr + 'events.df'
        fdir = self.this_class_arr + 'drought_events\\'
        events_dic = {}
        for f in tqdm(os.listdir(fdir)):
            dic_i = T.load_npy(fdir + f)
            events_dic.update(dic_i)
        df = pd.DataFrame()
        pix_list = []
        event_list = []
        for pix in events_dic:
            events = events_dic[pix]
            for event in events:
                pix_list.append(pix)
                event_list.append(event)
        df['pix'] = pix_list
        df['event'] = event_list
        T.save_df(df,dff)
        pass

    def clean_spei(self):
        fdir = self.this_data_root + 'per_pix\\'
        outdir = self.this_data_root + 'per_pix_clean\\'
        Pre_Process().clean_per_pix(fdir,outdir)
        pass




class Pick_drought_events:

    def __init__(self):

        pass

    def run(self):

        pass



def main():
    # CSIF().run()
    SPEI_preprocess().run()
    # Pick_drought_events().run()
    pass


if __name__ == '__main__':
    main()