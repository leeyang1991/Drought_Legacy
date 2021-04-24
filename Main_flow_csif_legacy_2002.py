# coding=utf-8

from __init__ import *
results_root_main_flow_2002 = this_root + 'results_root_main_flow_2002/'

results_root_main_flow = this_root + 'results_root_main_flow_2002/'

class Main_flow_Early_Peak_Late_Dormant:

    def __init__(self):
        self.this_class_arr = results_root_main_flow_2002 + 'arr/Main_flow_Early_Peak_Late_Dormant/'
        self.this_class_tif = results_root_main_flow_2002 + 'tif/Main_flow_Early_Peak_Late_Dormant/'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass


    def run(self):
        self.annual_phelogy()
        # self.long_term_pheology()
        pass


    def annual_phelogy(self):
        # calculate annual phenology
        # 1
        # self.tif_4_day_to_annual()
        # 2 data transform
        # self.data_transform_annual()
        # 3 smooth bi-weekly to daily
        # self.hants_smooth_annual()
        # 4 calculate phenology
        # self.early_peak_late_dormant_period_annual()
        # 5 transform daily to monthly
        # self.transform_early_peak_late_dormant_period_annual()
        # 99 check get_early_peak_late_dormant_period_long_term
        self.check_get_early_peak_late_dormant_period_long_term()
        pass


    def long_term_pheology(self):
        # calculate long term phenology
        # 1 calculate long term NDVI mean ,24 images
        # self.sif_mean_long_term()
        # 3 smooth bi-weekly to daily
        # self.hants_smooth()
        # 4 calculate phenology
        # self.early_peak_late_dormant_period_long_term()
        # 99 check phenology
        self.check_early_peak_late_dormant_period_long_term()

        pass

    def return_phenology(self):
        f = self.this_class_arr + 'early_peak_late_dormant_period_long_term/early_peak_late_dormant_period_long_term.npy'
        dic = T.load_npy(f)
        return dic

    def return_gs(self):
        f = self.this_class_arr + 'early_peak_late_dormant_period_long_term/early_peak_late_dormant_period_long_term.npy'
        dic = T.load_npy(f)
        gs_dic = {}
        for pix in dic:
            val = dic[pix]['GS_mon']
            gs_dic[pix] = val
        return gs_dic
        pass


    def check_early_peak_late_dormant_period_long_term(self):
        outtifdir = self.this_class_tif + 'early_peak_late_dormant_period_long_term/'
        T.mk_dir(outtifdir)
        f = self.this_class_arr + 'early_peak_late_dormant_period_long_term/early_peak_late_dormant_period_long_term.npy'
        dic = T.load_npy(f)

        # result = {
        #     'early_length': 'early_period',
        #     'mid_length': 'peak_period',
        #     'late_length': 'late_period',
        #     'dormant_length': 'dormant_period',
        #     'early_start': '.',
        #     'early_end': '.',
        #     'peak': '.',
        #     'late_start': '.',
        #     'late_end': '.',
        # }
        # for var in result:
        #     spatial_dic = {}
        #     for pix in dic:
        #         val = dic[pix][var]
        #         spatial_dic[pix] = val
        #     print var
        #     arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        #     DIC_and_TIF().arr_to_tif(arr,outtifdir+var+'.tif')
        #######################################################
        #######################################################
        #######################################################
        spatial_1 = {}
        spatial_2 = {}
        spatial_3 = {}
        for pix in dic:
            GS_start_mon = dic[pix]['GS_mon'][0]
            GS_end_mon = dic[pix]['GS_mon'][-1]
            GS_length_mon = len(dic[pix]['GS_mon'])
            spatial_1[pix] = GS_start_mon
            spatial_2[pix] = GS_end_mon
            spatial_3[pix] = GS_length_mon

        arr1 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_1)
        arr2 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_2)
        arr3 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_3)

        DIC_and_TIF().arr_to_tif(arr1,outtifdir + 'GS_start_mon.tif')
        DIC_and_TIF().arr_to_tif(arr2,outtifdir + 'GS_end_mon.tif')
        DIC_and_TIF().arr_to_tif(arr3,outtifdir + 'GS_length_mon.tif')


    def check_get_early_peak_late_dormant_period_long_term(self):
        fdir = self.this_class_arr + 'transform_early_peak_late_dormant_period_annual/'
        for var in os.listdir(fdir):
            # print var
            try:
                dic = T.load_npy(fdir + var)
                spatial_dic = {}
                for pix in tqdm(dic):
                    val = dic[pix]
                    if len(val) == 0:
                        continue
                    # plt.plot(val)
                    # plt.show()
                    meanarr = np.mean(val)
                    spatial_dic[pix] = meanarr
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
                plt.imshow(arr,cmap='RdGy_r')
                plt.title(var)
                plt.colorbar()
                plt.show()
            except:
                print(var,'invalid')

        pass

    def __day_to_month(self,doy):
        base = datetime.datetime(2000,1,1)
        time_delta = datetime.timedelta(int(doy))
        date = base + time_delta
        month = date.month
        day = date.day

        if day > 15:
            month = month + 1
        if month >= 12:
            month = 12
        return month


    def transform_early_peak_late_dormant_period_annual(self):
        vars_dic = {
            'early_length': 'early_period',
            'mid_length': 'peak_period',
            'late_length': 'late_period',
            'dormant_length': 'dormant_period',
            'early_start': 'early_start',
            'early_start_mon': 'self.__day_to_month(early_start)',

            'early_end': 'early_end',
            'early_end_mon': 'self.__day_to_month(early_end)',

            'peak': 'peak',
            'peak_mon': 'self.__day_to_month(peak)',

            'late_start': 'late_start',
            'late_start_mon': 'self.__day_to_month(late_start)',

            'late_end': 'late_end',
            'late_end_mon': 'self.__day_to_month(late_end)',
        }


        outdir = self.this_class_arr + 'transform_early_peak_late_dormant_period_annual/'
        T.mk_dir(outdir)
        fdir = self.this_class_arr + 'early_peak_late_dormant_period_annual/'
        sif_dic_hants_dir = self.this_class_arr + 'hants_smooth_annual/'
        # sif_dic = Tools().load_npy_dir(sif_dic_f)

        #
        for var in vars_dic:
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            for y in tqdm(range(2001,2016),desc=var):
                f = fdir + '{}.npy'.format(y)
                dic = T.load_npy(f)
                for pix in dic:
                    var_val = dic[pix][var]
                    spatial_dic[pix].append(var_val)
            np.save(outdir + var,spatial_dic)

        ############### Dormant Mons #############
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for y in tqdm(range(2001, 2016), desc='dormant_mons_list'):
            f = fdir + '{}.npy'.format(y)
            dic = T.load_npy(f)
            for pix in dic:
                sos = dic[pix]['early_start_mon']
                eos = dic[pix]['late_end_mon']
                gs = range(sos,eos+1)
                winter_mons = []
                for m in range(1,13):
                    if m in gs:
                        continue
                    winter_mons.append(m)
                spatial_dic[pix].append(winter_mons)
        np.save(outdir + 'dormant_mons', spatial_dic)

        ############### GS Mons #############
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for y in tqdm(range(2001, 2016), desc='dormant_mons_list'):
            f = fdir + '{}.npy'.format(y)
            dic = T.load_npy(f)
            for pix in dic:
                sos = dic[pix]['early_start_mon']
                eos = dic[pix]['late_end_mon']
                if sos >= eos:
                    continue
                gs_mons = range(sos, eos + 1)
                spatial_dic[pix].append(gs_mons)
        np.save(outdir + 'gs_mons', spatial_dic)

        ########  peak vals  #######
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for y in tqdm(range(2001, 2016), desc='peak_val'):
            f = fdir + '{}.npy'.format(y)
            dic = T.load_npy(f)
            sif_dic = T.load_npy(sif_dic_hants_dir + str(y) + '.npy')
            for pix in dic:
                var_val = dic[pix]['peak']
                peak_val = sif_dic[pix][var_val]
                spatial_dic[pix].append(peak_val)

        np.save(outdir + 'peak_val', spatial_dic)


        pass


    def kernel_hants_smooth_annual(self,params):

        per_pix_dir,year,outdir = params

        per_pix_dir_i = per_pix_dir + year + '/'
        outf = outdir + year
        ndvi_dic = {}
        for f in os.listdir(per_pix_dir_i):
            dic = T.load_npy(per_pix_dir_i + f)
            ndvi_dic.update(dic)
        hants_dic = {}
        for pix in tqdm(ndvi_dic,desc=year):
            # if not pix in gs_dic:
            #     continue
            vals = ndvi_dic[pix]
            smoothed_vals = self.__kernel_hants(vals)
            # print(smoothed_vals)
            if smoothed_vals[0] == None:
                continue
            # print(smoothed_vals)
            # plt.plot(smoothed_vals)
            # plt.show()
            hants_dic[pix] = np.array(smoothed_vals)
        np.save(outf, hants_dic)

        pass

    def hants_smooth_annual(self):
        outdir = self.this_class_arr + 'hants_smooth_annual/'
        T.mk_dir(outdir)
        # gs_f = Phenology_based_on_Temperature_NDVI().this_class_arr + 'growing_season_index.npy'
        # gs_dic = T.load_npy(gs_f)
        per_pix_dir = self.this_class_arr + 'data_transform_annual/'
        params = []
        for year in os.listdir(per_pix_dir):
            # print year
            params.append([per_pix_dir,year,outdir])
        MULTIPROCESS(self.kernel_hants_smooth_annual,params).run(process=6)



    def data_transform_annual(self):

        fdir = data_root + 'CSIF/tif/annual_clear/'
        outdir = self.this_class_arr + 'data_transform_annual/'
        T.mk_dir(outdir)
        for year in os.listdir(fdir):
            # print year,'\n'
            # for f in os.listdir(fdir + year):
            #     print f
            outdir_i = outdir + year + '/'
            T.mk_dir(outdir_i)
            Pre_Process().data_transform(fdir + year + '/', outdir_i)
        pass

    def early_peak_late_dormant_period_annual(self,threshold_i=0.2):
        hants_smooth_dir = self.this_class_arr + 'hants_smooth_annual/'
        # print(hants_smooth_dir)
        # print('/Users/wenzhang/project/drought_legacy/results_root_main_flow_2002/arr/Main_flow_Early_Peak_Late_Dormant/hants_smooth_annual')
        # exit()
        outdir = self.this_class_arr + 'early_peak_late_dormant_period_annual/'
        T.mk_dir(outdir)

        for f in os.listdir(hants_smooth_dir):
            outf_i = outdir + f
            year = int(f.split('.')[0])
            hants_smooth_f = hants_smooth_dir + f
            hants_dic = T.load_npy(hants_smooth_f)
            result_dic = {}
            for pix in tqdm(hants_dic,desc=str(year)):
                vals = hants_dic[pix]
                peak = np.argmax(vals)
                if peak == 0 or peak == (len(vals)-1):
                    continue
                try:
                    early_start = self.__search_left(vals, peak, threshold_i)
                    late_end = self.__search_right(vals, peak, threshold_i)
                except:
                    early_start = 60
                    late_end = 130
                    # print vals
                    plt.plot(vals)
                    plt.show()
                # method 1
                # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak)
                # method 2
                early_end, late_start = self.__median_early_late(vals,early_start,late_end,peak)

                early_period = early_end - early_start
                peak_period = late_start - early_end
                late_period = late_end - late_start
                dormant_period = 365 - (late_end - early_start)

                result = {
                    'early_length':early_period,
                    'mid_length':peak_period,
                    'late_length':late_period,
                    'dormant_length':dormant_period,
                    'early_start':early_start,
                    'early_start_mon':self.__day_to_month(early_start),

                    'early_end':early_end,
                    'early_end_mon':self.__day_to_month(early_end),

                    'peak':peak,
                    'peak_mon':self.__day_to_month(peak),

                    'late_start':late_start,
                    'late_start_mon':self.__day_to_month(late_start),

                    'late_end':late_end,
                    'late_end_mon':self.__day_to_month(late_end),
                }
                # print(result)
                # exit()
                result_dic[pix] = result
            np.save(outf_i,result_dic)

    def early_peak_late_dormant_period_long_term(self,threshold_i=0.2):
        hants_smooth_f = self.this_class_arr + 'hants_smooth/hants_smooth.npy'
        outdir = self.this_class_arr + 'early_peak_late_dormant_period_long_term/'
        T.mk_dir(outdir)
        outf = outdir + 'early_peak_late_dormant_period_long_term'
        hants_dic = T.load_npy(hants_smooth_f)
        result_dic = {}
        for pix in tqdm(hants_dic):
            vals = hants_dic[pix]
            peak = np.argmax(vals)
            try:
                # start = self.__search_left(vals, maxind, threshold_i)
                # end = self.__search_right(vals, maxind, threshold_i)
                # dormant_length = 365 - (end - start)
                # spatial_dic[pix] = dormant_length/30.

                early_start = self.__search_left(vals, peak, threshold_i)
                late_end = self.__search_right(vals, peak, threshold_i)
                # method 1
                # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak)
                # method 2
                early_end, late_start = self.__median_early_late(vals, early_start, late_end, peak)

                early_period = early_end - early_start
                peak_period = late_start - early_end
                late_period = late_end - late_start
                dormant_period = 365 - (late_end - early_start)
                GS_mon = range(self.__day_to_month(early_start),self.__day_to_month(late_end)+1)
                result = {
                    'early_length': early_period,
                    'mid_length': peak_period,
                    'late_length': late_period,
                    'dormant_length': dormant_period,
                    'early_start': self.__day_to_month(early_start),
                    'early_end': self.__day_to_month(early_end),
                    'peak': self.__day_to_month(peak),
                    'late_start': self.__day_to_month(late_start),
                    'late_end': self.__day_to_month(late_end),
                    'GS_mon': np.array(GS_mon),
                }
                result_dic[pix] = result
            except:
                pass
        np.save(outf,result_dic)

    def hants_smooth(self):
        outdir = self.this_class_arr + 'hants_smooth/'
        outf = outdir + 'hants_smooth'
        T.mk_dir(outdir)
        # gs_f = Phenology_based_on_Temperature_NDVI().this_class_arr + 'growing_season_index.npy'
        # gs_dic = T.load_npy(gs_f)
        per_pix_dir = self.this_class_arr + 'per_pix_dic_long_term/'
        ndvi_dic = {}
        for f in os.listdir(per_pix_dir):
            dic = T.load_npy(per_pix_dir+f)
            ndvi_dic.update(dic)

        hants_dic = {}
        for pix in tqdm(ndvi_dic):
            # if not pix in gs_dic:
            #     continue
            vals = ndvi_dic[pix]
            smoothed_vals = self.__kernel_hants(vals)
            hants_dic[pix] = np.array(smoothed_vals)
        np.save(outf,hants_dic)


    def tif_4_day_to_annual(self):
        fdir = data_root + 'CSIF/tif/clear/'
        outdir = data_root + 'CSIF/tif/annual_clear/'
        T.mk_dir(outdir)
        year_list = []
        for y in range(2001, 2016):
            year_ = '{}'.format(y)
            year_list.append(year_)
        for yyyy in year_list:
            print(yyyy)
            outdir_i = outdir + yyyy + '/'
            T.mk_dir(outdir_i)
            for f in os.listdir(fdir):
                if f[:4]==yyyy:
                    print(f)
                    shutil.copy(fdir+f,outdir_i+f)


    def sif_mean_long_term(self):
        fdir = data_root + 'CSIF/tif/clear/'
        outdir = self.this_class_arr + 'per_pix_dic_long_term/'
        T.mk_dir(outdir)

        arrs = []
        for f in tqdm(sorted(os.listdir(fdir))):
            arr = to_raster.raster2array(fdir+f)[0]
            arrs.append(arr)

        spatial_dic = {}
        for r in tqdm(range(len(arrs[0]))):
            for c in range(len(arrs[0][0])):
                pix = (r,c)
                time_series = []
                for arr in arrs:
                    val = arr[r][c]
                    time_series.append(val)
                time_series = np.array(time_series)
                time_series[time_series<-999]=np.nan
                time_series = Tools().interp_nan(time_series)
                if time_series[0]==None:
                    continue
                time_series_reshape = np.reshape(time_series,(len(range(2001,2017)),-1))
                time_series_mean = []
                for i in time_series_reshape:
                    mean = np.mean(i)
                    time_series_mean.append(mean)
                time_series_mean = np.array(time_series_mean)
                spatial_dic[pix] = time_series_mean
        np.save(outdir+'sif_long_term',spatial_dic)




    def data_transform(self):
        fdir = self.this_class_tif + 'tif_bi_weekly_mean/'
        outdir = self.this_class_arr + 'NDVI_bi_weekly_per_pix/'
        Pre_Process().data_transform(fdir,outdir)

    def __interp__(self, vals):

        # x_new = np.arange(min(inx), max(inx), ((max(inx) - min(inx)) / float(len(inx))) / float(zoom))

        inx = range(len(vals))
        iny = vals
        x_new = np.linspace(min(inx), max(inx), 365)
        func = interpolate.interp1d(inx, iny)
        y_new = func(x_new)

        return x_new, y_new

    def __search_left(self, vals, maxind, threshold_i):
        left_vals = vals[:maxind]
        left_min = np.min(left_vals)
        max_v = vals[maxind]
        # if left_min < 2000:
        #     left_min = 2000
        threshold = (max_v - left_min) * threshold_i + left_min

        ind = 999999
        for step in range(365):
            ind = maxind - step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __search_right(self, vals, maxind, threshold_i):
        right_vals = vals[maxind:]
        right_min = np.min(right_vals)
        max_v = vals[maxind]
        # if right_min < 2000:
        #     right_min = 2000
        threshold = (max_v - right_min) * threshold_i + right_min

        ind = 999999
        for step in range(365):
            ind = maxind + step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __kernel_hants(self,vals_bi_week):
        vals = np.array(vals_bi_week)
        std = np.std(vals)
        if std == 0:
            return [None]
        xnew, ynew = self.__interp__(vals)
        ynew = np.array([ynew])
        results = HANTS.HANTS(sample_count=365, inputs=ynew, low=-10000, high=10000,
                        fit_error_tolerance=std)
        result = results[0]

        # plt.plot(result)
        # plt.plot(range(len(ynew[0])),ynew[0])
        # plt.show()
        return result


    def __slope_early_late(self,vals,sos,eos,peak):
        # 1 slope最大和最小分别作为early late 的结束和开始
        # 问题：early late 时间太短
        slope_left = []
        for i in range(sos,peak):
            if i-1 < 0:
                slope_i = vals[1]-vals[0]
            else:
                slope_i = vals[i]-vals[i-1]
            slope_left.append(slope_i)

        slope_right = []
        for i in range(peak,eos):
            if i-1 < 0:
                slope_i = vals[1]-vals[0]
            else:
                slope_i = vals[i]-vals[i-1]
            slope_right.append(slope_i)

        max_ind = np.argmax(slope_left) + sos
        min_ind = np.argmin(slope_right) + peak

        return max_ind, min_ind


    def __median_early_late(self,vals,sos,eos,peak):
        # 2 使用sos-peak peak-eos中位数作为sos和eos的结束和开始

        median_left = int((peak-sos)/2.)
        median_right = int((eos - peak)/2)
        max_ind = median_left + sos
        min_ind = median_right + peak
        return max_ind, min_ind


class Main_Flow_Pick_drought_events:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr/SPEI_preprocess/'
        self.this_class_tif = results_root_main_flow + 'tif/SPEI_preprocess/'
        self.this_class_png = results_root_main_flow + 'png/SPEI_preprocess/'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        # self.spei_per_pix_dic()
        self.do_pick()
        pass


    def spei_per_pix_dic(self):
        fdir = data_root + 'SPEI/tif/'
        outdir = data_root + 'SPEI/per_pix_2002/'
        Pre_Process().data_transform(fdir,outdir)

        pass

    def do_pick(self):
        outdir = self.this_class_arr + 'drought_events/'
        fdir = data_root + 'SPEI/per_pix_2002/'
        for f in os.listdir(fdir):
            # if not '015' in f:
            #     continue
            self.pick_events(fdir + f,outdir)
        pass

    def pick_events(self,f, outdir):
        # 前n个月和后n个月无极端干旱事件
        fname = f.split('.')[0].split('_')[-1]
        # print(fname)
        # exit()
        n = 12
        T.mk_dir(outdir,force=True)
        single_event_dic = {}
        dic = T.load_npy(f)
        for pix in tqdm(dic,desc='picking {}'.format(f)):
            vals = dic[pix]
            print(len(vals))
            print(vals)
            # exit()
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


class Main_flow_Legacy:

    def __init__(self):
        self.this_class_arr = results_root_main_flow_2002 + 'arr/Main_flow_Legacy/'
        self.this_class_tif = results_root_main_flow_2002 + 'tif/Main_flow_Legacy/'
        self.this_class_png = results_root_main_flow_2002 + 'png/Main_flow_Legacy/'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        self.cal_leagacy()
        pass


    def cal_leagacy(self):
        # sif_dir = data_root + 'CSIF/per_pix_anomaly/'
        # sif_dic = T.load_npy_dir(sif_dir)
        # T.mk_dir(sif_dir_180)
        # sif_dic_180 = {}
        # for pix in tqdm(sif_dic):
        #     val = sif_dic[pix]
        #     val_180 = val[:180]
        #     val_180 = np.array(val_180)
        #     sif_dic_180[pix] = val_180
        # np.save(sif_dir_180 + 'per_pix_180',sif_dic_180)
        # exit()

        sif_dir_180 = data_root + 'CSIF/per_pix_anomaly_180/'
        event_dir = Main_Flow_Pick_drought_events().this_class_arr + 'drought_events/'
        event_dic = T.load_npy_dir(event_dir)
        for pix in event_dic:
            events = event_dic[pix]
            for evt in events:
                print(evt)







def main():
    # Main_flow_Early_Peak_Late_Dormant().run()

    # Main_Flow_Pick_drought_events().run()
    Main_flow_Legacy().run()
    pass



if __name__ == '__main__':
    main()
    pass