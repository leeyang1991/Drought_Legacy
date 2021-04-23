# coding=utf-8

from __init__ import *
results_root_main_flow_2002 = this_root + 'results_root_main_flow_2002/'
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
        self.hants_smooth_annual()
        # 4 calculate phenology
        # self.early_peak_late_dormant_period_annual()
        # 5 transform daily to monthly
        # self.transform_early_peak_late_dormant_period_annual()
        # 99 check get_early_peak_late_dormant_period_long_term
        # self.check_get_early_peak_late_dormant_period_long_term()
        pass


    def long_term_pheology(self):
        # calculate long term phenology
        # 1 calculate long term NDVI mean ,24 images
        # self.tif_bi_weekly_mean_long_term()
        # 2 transform to per pixel
        # self.data_transform()
        # 3 smooth bi-weekly to daily
        # self.hants_smooth()
        # 4 calculate phenology
        self.early_peak_late_dormant_period_long_term()
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
            plt.imshow(arr)
            plt.colorbar()
            plt.show()

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
        #
        for var in vars_dic:
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            for y in tqdm(range(1982,2016),desc=var):
                f = fdir + '{}.npy'.format(y)
                dic = T.load_npy(f)
                for pix in dic:
                    var_val = dic[pix][var]
                    spatial_dic[pix].append(var_val)
            np.save(outdir + var,spatial_dic)

        ############### Dormant Mons #############
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for y in tqdm(range(1982, 2016), desc='dormant_mons_list'):
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
        for y in tqdm(range(1982, 2016), desc='dormant_mons_list'):
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
        gs_f = Phenology_based_on_Temperature_NDVI().this_class_arr + 'growing_season_index.npy'
        gs_dic = T.load_npy(gs_f)
        per_pix_dir = self.this_class_arr + 'NDVI_bi_weekly_per_pix/'
        ndvi_dic = {}
        for f in os.listdir(per_pix_dir):
            dic = T.load_npy(per_pix_dir+f)
            ndvi_dic.update(dic)

        hants_dic = {}
        for pix in tqdm(ndvi_dic):
            if not pix in gs_dic:
                continue
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


    def tif_bi_weekly_mean_long_term(self):
        fdir = data_root + 'NDVI/tif_05deg_bi_weekly/'
        outdir = self.this_class_tif + 'tif_bi_weekly_mean/'
        T.mk_dir(outdir)
        date_list = []
        for m in range(1, 13):
            for d in [1, 15]:
                temp =[]
                for y in range(1982,2016):
                    temp.append('{}{:02d}{:02d}.tif'.format(y,m,d))
                date_list.append(temp)
        template_f = DIC_and_TIF().tif_template
        template_arr = to_raster.raster2array(template_f)[0]
        for mon in date_list:
            fname = mon[0].split('.')[0][-4:] + '.tif'
            # print fname
            arrs = np.zeros_like(template_arr)
            for y in mon:
                f = fdir + y
                arr = to_raster.raster2array(f)[0]
                arrs += arr
            mon_mean_arr = arrs / len(mon)
            T.mask_999999_arr(mon_mean_arr)
            DIC_and_TIF().arr_to_tif(mon_mean_arr,outdir + fname)



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
        if left_min < 2000:
            left_min = 2000
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
        if right_min < 2000:
            right_min = 2000
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


class Pick_drought_events:

    def __init__(self):

        pass


    def run(self):

        pass


class Main_flow_Legacy:

    def __init__(self):
        self.this_class_arr = results_root_main_flow_2002 + 'arr\\Main_flow_Legacy\\'
        self.this_class_tif = results_root_main_flow_2002 + 'tif\\Main_flow_Legacy\\'
        self.this_class_png = results_root_main_flow_2002 + 'png\\Main_flow_Legacy\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        for y in range(1,4):
            print(y)
            self.cal_legacy_monthly(y)
        self.plot_hist_spatial_mean()
        pass


    def load_df(self):
        dff = Main_flow_Dataframe().dff
        df = T.load_df(dff)
        T.print_head_n(df)
        return df,dff

    def cal_legacy_monthly(self,legacy_year):

        SIFdir = data_root + 'CSIF\\per_pix_anomaly_detrend\\'
        SIFdic = T.load_npy_dir(SIFdir)
        df,dff = self.load_df()
        legacy_list = []
        # rf_model_dic = self.cal_linear_reg()
        spatial_dic = DIC_and_TIF().void_spatial_dic_nan()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            event = row.event
            drought_start = event[0]
            drought_mon = drought_start % 12 + 1
            drought_mon = int(drought_mon)
            # gs_mons = list(range(1,13)) ################## Todo: Need to calculate Growing season via phenology
            gs_mons = list(range(4,11)) ################## Todo: Need to calculate Growing season via phenology
            # if not drought_mon in gs_mons:
            #     legacy_list.append(np.nan)
            #     continue
            if not pix in SIFdic:
                legacy_list.append(np.nan)
                continue

            sif = SIFdic[pix]
            sif = np.array(sif)

            gs_indx = []
            for m in range(len(sif)):
                mon = m % 12 + 1
                mon = int(mon)
                if mon in gs_mons:
                    gs_indx.append(m)

            legacy_months_start = (legacy_year - 1) * len(gs_mons)
            legacy_months_end = legacy_year * len(gs_mons)
            if drought_start + legacy_months_end >= len(sif):
                legacy_list.append(np.nan)
                continue

            legacy_duration = range(drought_start + legacy_months_start, drought_start + legacy_months_end)
            selected_indx = list(legacy_duration)
            sif_obs_selected = T.pick_vals_from_1darray(sif,selected_indx)
            # plt.plot(sif_obs_selected)
            # plt.plot(sif_pred_selected)
            # plt.show()
            legacy = sif_obs_selected

            legacy_mean = np.mean(legacy)
            # legacy_mean = np.sum(legacy)
            legacy_list.append(legacy_mean)
            # print(legacy_mean)
            # legacy_dic[pix] = legacy_mean
            # spatial_dic[pix] = 1


        df['monthly_legacy_decrease_year_{}'.format(legacy_year)] = legacy_list
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # DIC_and_TIF().plot_back_ground_arr()

        # plt.imshow(arr)
        # plt.show()
        T.save_df(df,dff)


    def plot_hist_spatial_mean(self):
        df,dff = self.load_df()
        df = df[df['lat']>30]
        df = df[df['lat']<60]
        df = df[df['gs_sif_spei_corr']>0]
        df = df[df['gs_sif_spei_corr_p']<0.05]

        legacy_list_all = []
        for year in range(1, 4):
            print(year)
            legacy_dic = DIC_and_TIF().void_spatial_dic()
            legacy_list = []
            for i,row in tqdm(df.iterrows(),total=len(df)):
                pix = row.pix
                legacy = row['monthly_legacy_decrease_year_{}'.format(year)]
                if np.isnan(legacy):
                    continue
                legacy_dic[pix].append(legacy)
                legacy_list.append(legacy)
            legacy_list_all.append(np.mean(legacy_list))

            arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(legacy_dic)
            plt.figure()
            DIC_and_TIF().plot_back_ground_arr()
            plt.imshow(arr,vmin=-1,vmax=1)
            plt.colorbar()
            plt.title('monthly_legacy_decrease_year_{}'.format(year))
        plt.figure()
        plt.bar(list(range(len(legacy_list_all))),legacy_list_all)
        plt.show()


        # T.print_head_n(df)
        # exit()
        for year in range(1,4):
            print(year)
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            for i,row in tqdm(df.iterrows(),total=len(df)):
                pix = row.pix
                legacy = row['monthly_legacy_decrease_year_{}'.format(year)]
                # hue_list.append('annual_legacy_year_{}'.format(year))
                spatial_dic[pix].append(legacy)
            hist = []
            for pix in tqdm(spatial_dic):
                vals = spatial_dic[pix]
                if len(vals)==0:
                    continue
                mean = np.nanmean(vals)
                if np.isnan(mean):
                    continue
                hist.append(mean)
                # for val in vals:
                #     if np.isnan(val):
                #         continue
                #     hist.append(val)
            hist = np.array(hist)
            n,x,_ = plt.hist(hist,range=(-2,2),bins=120,alpha=1.0,density=True,histtype=u'step',label='legacy_{}'.format(year))
            density = stats.gaussian_kde(hist)
            # plt.plot(x,density(x),label='legacy_{}'.format(year))
            # print(n)
            nn = SMOOTH().smooth_convolve(n,21)
            plt.plot(x,nn,label='legacy_{}'.format(year))
            # df_temp['annual_legacy'] = hist
            # df_temp['hue'] = hue_list

            # sns.pairplot(df_temp,hue='hue')
            # plt.xlim(-1,1)
        plt.vlines([0],0,1,linestyles='--',colors='k')
        plt.legend()
        plt.show()

        pass




def main():
    Main_flow_Early_Peak_Late_Dormant().run()
    pass



if __name__ == '__main__':
    main()
    pass