# coding=utf-8
import time

from __init__ import *
results_root_main_flow_2002 = this_root + 'results_root_main_flow_2002/'

results_root_main_flow = results_root_main_flow_2002
class Global_vars:
    def __init__(self):
        # self.growing_date_range = list(range(5,11))
        self.tif_template_7200_3600 = this_root + 'conf/tif_template_005.tif'
        pass

    def koppen_landuse(self):
        kl_list = [u'Forest.A', u'Forest.B', u'Forest.Cf', u'Forest.Csw', u'Forest.Df', u'Forest.Dsw', u'Forest.E',
         u'Grasslands.A', u'Grasslands.B', u'Grasslands.Cf', u'Grasslands.Csw', u'Grasslands.Df', u'Grasslands.Dsw',
         u'Grasslands.E', u'Shrublands.A', u'Shrublands.B', u'Shrublands.Cf', u'Shrublands.Csw', u'Shrublands.Df',
         u'Shrublands.Dsw', u'Shrublands.E']
        return kl_list

    def koppen_list(self):
        koppen_list = [ u'B', u'Cf', u'Csw', u'Df', u'Dsw', u'E',]
        return koppen_list
        pass


    def marker_dic(self):
        markers_dic = {
                       'Shrublands': "o",
                       'Forest': "X",
                       'Grasslands': "p",
                       }
        return markers_dic
    def color_dic_lc(self):
        markers_dic = {
                       'Shrublands': "b",
                       'Forest': "g",
                       'Grasslands': "r",
                       }
        return markers_dic
    def landuse_list(self):
        lc_list = [
              'Forest',
            'Shrublands',
            'Grasslands',
        ]
        return lc_list

    def line_color_dic(self):
        line_color_dic = {
            'pre': 'g',
            'early': 'r',
            'late': 'b'
        }
        return line_color_dic

    def gs_mons(self):

        gs = list(range(5,10))

        return gs

    def variables(self,n=3):

        X = [
            # 'isohydricity',
            'NDVI_pre_{}'.format(n),
            'CSIF_pre_{}'.format(n),
            'VPD_previous_{}'.format(n),
            'TMP_previous_{}'.format(n),
            'PRE_previous_{}'.format(n),
            'SM_previous_{}'.format(n),
            'TWS_previous_{}'.format(n),
        ]
        # Y = 'greenness_loss'
        Y = 'carbon_loss'
        # Y = 'legacy_1'
        # Y = 'legacy_2'
        # Y = 'legacy_3'

        return X,Y

        pass

    def clean_df(self,df):
        ndvi_valid_f = results_root_main_flow_2002 + 'arr/NDVI/NDVI_invalid_mask.npy'
        ndvi_valid_arr = np.load(ndvi_valid_f)

        spatial_dic = DIC_and_TIF().spatial_arr_to_dic(ndvi_valid_arr)
        valid_ndvi_dic = {}
        for pix in spatial_dic:
            val = spatial_dic[pix]
            if np.isnan(val):
                continue
            valid_ndvi_dic[pix]=1
        print(len(df))
        drop_index = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='Cleaning DF'):
            pix = row.pix
            if not pix in valid_ndvi_dic:
                drop_index.append(i)
        df = df.drop(df.index[drop_index])

        # print(len(df))
        # exit()
        # df = df.drop_duplicates(subset=['pix', 'delta_legacy'])
        # self.__df_to_excel(df,dff+'drop')

        # df = df[df['ratio_of_forest'] > 0.90]
        df = df[df['lat'] > 30]
        # df = df[df['lat'] < 60]
        # df = df[df['delta_legacy'] < -0]
        # df = df[df['trend_score'] > 0.2]
        # df = df[df['gs_sif_spei_corr'] > 0]

        # trend = df['trend']
        # trend_mean = np.nanmean(trend)
        # trend_std = np.nanstd(trend)
        # up = trend_mean + trend_std
        # down = trend_mean - trend_std
        # df = df[df['trend'] > down]
        # df = df[df['trend'] < up]

        # quantile = 0.4
        # delta_legacy = df['delta_legacy']
        # delta_legacy = delta_legacy.dropna()
        #
        # # print(delta_legacy)
        # q = np.quantile(delta_legacy,quantile)
        # # print(q)
        # df = df[df['delta_legacy']<q]
        # T.print_head_n(df)
        # exit()
        # spatial_dic = {}
        # for i, row in tqdm(df.iterrows(), total=len(df),desc='gen spatial_dic...'):
        #     pix = row.pix
        #     val = 1
        #     spatial_dic[pix] = val

        return df,spatial_dic

    def mask_arr_with_NDVI(self, inarr):
        ndvi_valid_f = results_root_main_flow_2002 + 'arr/NDVI/NDVI_invalid_mask.npy'
        ndvi_valid_arr = np.load(ndvi_valid_f)
        grid = np.isnan(ndvi_valid_arr)
        inarr[grid] = np.nan

        pass

    def dic_to_growing_season(self,dic):
        start_mon = self.growing_date_range[0]
        end_mon = self.growing_date_range[-1]
        growing_season_dic = {}
        for pix in tqdm(dic,desc='transfoming to growing season'):
            vals = dic[pix]
            vals_reshape = np.reshape(vals,(-1,12))
            vals_reshape_T = vals_reshape.T
            vals_reshape_T_selected = vals_reshape_T[start_mon:end_mon]
            vals_reshape_T_selected_T = vals_reshape_T_selected.T
            vals_reshape_T_selected_T_flatten = vals_reshape_T_selected_T.flatten()
            # plt.plot(vals_reshape_T_selected_T_flatten)
            # plt.imshow(vals_reshape)
            # plt.show()
            growing_season_dic[pix] = vals_reshape_T_selected_T_flatten
        return growing_season_dic

        pass

    def growing_season_indx_to_all_year_indx(self,indexs):
        new_indexs = []
        for indx in indexs:
            n_year = indx // len(self.growing_date_range)
            res_mon = indx % len(self.growing_date_range)
            real_indx = n_year * 12 + res_mon + self.growing_date_range[0]
            new_indexs.append(real_indx)
        new_indexs = tuple(new_indexs)
        return new_indexs


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
        start = time.time()
        outdir = self.this_class_arr + 'hants_smooth_annual_m1/'
        T.mk_dir(outdir)
        # gs_f = Phenology_based_on_Temperature_NDVI().this_class_arr + 'growing_season_index.npy'
        # gs_dic = T.load_npy(gs_f)
        per_pix_dir = self.this_class_arr + 'data_transform_annual/'
        params = []
        for year in os.listdir(per_pix_dir):
            # print year
            if year.startswith('.'):
                continue
            params.append([per_pix_dir,year,outdir])
            self.kernel_hants_smooth_annual([per_pix_dir,year,outdir])
        # MULTIPROCESS(self.kernel_hants_smooth_annual,params).run()
        end = time.time()
        duration = end-start
        print('duration:{} s'.format(duration))



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
        # outdir = self.this_class_arr + 'drought_events/'
        outdir = self.this_class_arr + 'drought_events_for_modis/'
        # fdir = data_root + 'SPEI/per_pix_2002/'
        fdir = data_root + 'SPEI/per_pix_for_modis/'
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
        n = 0
        T.mk_dir(outdir,force=True)
        single_event_dic = {}
        dic = T.load_npy(f)
        for pix in tqdm(dic,desc='picking {}'.format(f)):
            vals = dic[pix]
            # print(len(vals))
            # print(vals)
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
            threshold = -1.5
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

    def __drought_indx_to_gs_indx(self,indx,gs_mons,vals_len):
        void_list = [0] * vals_len
        void_list[indx] = 1
        selected_indx = []
        for i in range(len(void_list)):
            mon = i % 12 + 1
            if mon in gs_mons:
                selected_indx.append(void_list[i])
        if 1 in selected_indx:
            trans_indx = selected_indx.index(1)
            return trans_indx
        else:
            return None
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
        n = 2
        gs = list(range(5,11))
        outf = self.this_class_arr + 'legacy_dic_{}'.format(n)
        sif_dir_180 = data_root + 'CSIF/per_pix_anomaly_180/'

        sif_dic = T.load_npy_dir(sif_dir_180)
        event_dir = Main_Flow_Pick_drought_events().this_class_arr + 'drought_events/'
        event_dic = T.load_npy_dir(event_dir)
        legacy_dic = {}
        for pix in tqdm(event_dic):
            events = event_dic[pix]
            if not pix in sif_dic:
                continue
            sif = sif_dic[pix]
            legacy_dic_i = {}
            for evt in events:
                evt_start = evt[0]
                if evt_start + n * 12 >= len(sif):
                    continue
                sif_after_n_year_range = range(evt_start + (n-1) * 12,evt_start + n * 12)
                    # sif[evt_start:evt_start + n * 12]
                selected_range = []
                for indx in sif_after_n_year_range:
                    mon = indx % 12 + 1
                    if mon in gs:
                        selected_range.append(indx)
                selected_vals = T.pick_vals_from_1darray(sif,selected_range)
                legacy = np.mean(selected_vals)
                evt_start_gs = self.__drought_indx_to_gs_indx(evt_start, gs, len(sif))
                # print(evt_start,evt_start_gs)
                if evt_start_gs == None:
                    continue
                legacy_dic_i[evt_start_gs] = legacy
            legacy_dic[pix] = legacy_dic_i
        np.save(outf,legacy_dic)

class Main_flow_Carbon_loss:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr/Main_flow_Carbon_loss/'
        self.this_class_tif = results_root_main_flow + 'tif/Main_flow_Carbon_loss/'
        self.this_class_png = results_root_main_flow + 'png/Main_flow_Carbon_loss/'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        # 1 cal recovery time
        event_dic,spei_dic,sif_dic = self.load_data()
        out_dir = self.this_class_arr + 'Recovery_time_Legacy/'
        self.gen_recovery_time_legacy(event_dic,spei_dic, sif_dic,out_dir)
        pass

    def load_data(self,condition=''):
        # events_dir = results_root_main_flow + 'arr/SPEI_preprocess/drought_events/'
        # SPEI_dir = data_root + 'SPEI/per_pix_clean/'
        # SIF_dir = data_root + 'CSIF/per_pix_anomaly_detrend/'

        events_dir = Main_Flow_Pick_drought_events().this_class_arr + 'drought_events/'
        SPEI_dir = data_root + 'SPEI/per_pix_2002/'
        SIF_dir = data_root + '/CSIF/per_pix_anomaly_180/'

        event_dic = T.load_npy_dir(events_dir,condition)
        spei_dic = T.load_npy_dir(SPEI_dir,condition)
        sif_dic = T.load_npy_dir(SIF_dir,condition)

        return event_dic,spei_dic,sif_dic
        pass


    def __cal_legacy(self,ndvi_obs,recovery_range):
        selected_obs = T.pick_vals_from_1darray(ndvi_obs,recovery_range)
        diff = selected_obs
        legacy = np.sum(diff)
        return legacy
        pass

    def gen_recovery_time_legacy(self, events, spei_dic, ndvi_dic, out_dir):
        '''
        生成全球恢复期
        :param interval: SPEI_{interval}
        :return:
        '''

        # pre_dic = Main_flow_Prepare().load_X_anomaly('PRE')

        growing_date_range = list(range(5,11))
        Tools().mk_dir(out_dir, force=True)
        outf = out_dir + 'recovery_time_legacy'
        # 1 加载事件
        # interval = '%02d' % interval
        # 2 加载NDVI 与 SPEI
        recovery_time_dic = {}
        for pix in tqdm(ndvi_dic):
            if pix in events:
                ndvi = ndvi_dic[pix]
                ndvi = np.array(ndvi)
                if not pix in spei_dic:
                    continue
                spei = spei_dic[pix]
                spei = np.array(spei)
                event = events[pix]
                recovery_time_result = []
                for date_range in event:
                    # print(date_range)
                    event_start_index = T.pick_min_indx_from_1darray(spei, date_range)
                    event_start_index_trans = self.__drought_indx_to_gs_indx(event_start_index,growing_date_range,len(ndvi))
                    if event_start_index_trans == None:
                        continue
                    ndvi_gs = self.__pick_gs_vals(ndvi,growing_date_range)
                    spei_gs = self.__pick_gs_vals(spei,growing_date_range)
                    # ndvi_gs_pred = self.__pick_gs_vals(ndvi_pred,growing_date_range)
                    # print(len(ndvi_gs))
                    # print(len(spei_gs))
                    # print(len(ndvi_gs_pred))
                    date_range_new = []
                    for i in date_range:
                        i_trans = self.__drought_indx_to_gs_indx(i,growing_date_range,len(ndvi))
                        if i_trans != None:
                            date_range_new.append(i_trans)
                    # 1 挑出此次干旱事件的NDVI和SPEI值 （画图需要）
                    # spei_picked_vals = Tools().pick_vals_from_1darray(spei, date_range)
                    # 2 挑出此次干旱事件SPEI最低的索引
                    # 在当前生长季搜索

                    # 4 搜索恢复期
                    # 4.1 获取growing season NDVI的最小值
                    # 4.3 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
                    search_end_indx = 3 * len(growing_date_range)
                    recovery_range = self.search1(ndvi_gs, spei_gs, event_start_index_trans,search_end_indx)
                    # continue
                    # recovery_time, lag, recovery_start_gs, recovery_start, 'undefined'
                    ###########################################
                    ###########################################
                    ###########################################
                    if recovery_range == None:
                        continue
                    recovery_range = np.array(recovery_range)
                    date_range_new = np.array(date_range_new)
                    recovery_time = len(recovery_range)
                    legacy = self.__cal_legacy(ndvi_gs,recovery_range)
                    recovery_time_result.append({
                        'recovery_time': recovery_time,
                        'recovery_date_range': recovery_range,
                        'drought_event_date_range': date_range_new,
                        'carbon_loss': legacy,
                    })
                    #
                    # ################# plot ##################
                    # print('recovery_time',recovery_time)
                    # print('growing_date_range',growing_date_range)
                    # print('recovery_range',recovery_range)
                    # print('legacy',legacy)
                    # recovery_date_range = recovery_range
                    # recovery_ndvi = Tools().pick_vals_from_1darray(ndvi_gs, recovery_date_range)
                    #
                    # # pre_picked_vals = Tools().pick_vals_from_1darray(pre, tmp_pre_date_range)
                    # # tmp_picked_vals = Tools().pick_vals_from_1darray(tmp, tmp_pre_date_range)
                    # # if len(swe) == 0:
                    # #     continue
                    # # swe_picked_vals = Tools().pick_vals_from_1darray(swe, tmp_pre_date_range)
                    #
                    # plt.figure(figsize=(8, 6))
                    # # plt.plot(tmp_pre_date_range, pre_picked_vals, '--', c='blue', label='precipitation')
                    # # plt.plot(tmp_pre_date_range, tmp_picked_vals, '--', c='cyan', label='temperature')
                    # # plt.plot(tmp_pre_date_range, swe_picked_vals, '--', c='black', linewidth=2, label='SWE',
                    # #          zorder=99)
                    # # plt.plot(recovery_date_range, recovery_ndvi, c='g', linewidth=6, label='Recovery Period')
                    # plt.scatter(recovery_date_range, recovery_ndvi, c='g', label='Recovery Period')
                    # # plt.plot(date_range, spei_picked_vals, c='r', linewidth=6,
                    # #          label='drought Event')
                    # # plt.scatter(date_range, spei_picked_vals, c='r', zorder=99)
                    #
                    # plt.plot(range(len(ndvi_gs)), ndvi_gs, '--', c='g', zorder=99, label='ndvi')
                    # plt.plot(range(len(spei_gs)), spei_gs, '--', c='r', zorder=99, label='drought index')
                    # # plt.plot(range(len(pre)), pre, '--', c='blue', zorder=99, label='Precip')
                    # # pre_picked = T.pick_vals_from_1darray(pre,recovery_date_range)
                    # # pre_mean = np.mean(pre_picked)
                    # # plt.plot(recovery_date_range,[pre_mean]*len(recovery_date_range))
                    # plt.legend()
                    #
                    # minx = 9999
                    # maxx = -9999
                    #
                    # for ii in recovery_date_range:
                    #     if ii > maxx:
                    #         maxx = ii
                    #     if ii < minx:
                    #         minx = ii
                    #
                    # for ii in date_range_new:
                    #     if ii > maxx:
                    #         maxx = ii
                    #     if ii < minx:
                    #         minx = ii
                    #
                    # # xtick = []
                    # # for iii in np.arange(len(ndvi)):
                    # #     year = 1982 + iii / 12
                    # #     mon = iii % 12 + 1
                    # #     mon = '%02d' % mon
                    # #     xtick.append('{}.{}'.format(year, mon))
                    # # # plt.xticks(range(len(xtick))[::3], xtick[::3], rotation=90)
                    # # plt.xticks(range(len(xtick)), xtick, rotation=90)
                    # plt.grid()
                    # plt.xlim(minx - 5, maxx + 5)
                    #
                    # lon, lat, address = Tools().pix_to_address(pix)
                    # try:
                    #     plt.title('lon:{:0.2f} lat:{:0.2f} address:{}\n'.format(lon, lat, address) +
                    #               'recovery_time:'+str(recovery_time)
                    #               )
                    #
                    # except:
                    #     plt.title('lon:{:0.2f} lat:{:0.2f}\n'.format(lon, lat)+
                    #               'recovery_time:' + str(recovery_time)
                    #               )
                    # plt.show()
            #         # #################plot end ##################
                recovery_time_dic[pix] = recovery_time_result
            else:
                recovery_time_dic[pix] = []
        T.save_dict_to_binary(recovery_time_dic, outf)
        pass


    def __drought_indx_to_gs_indx(self,indx,gs_mons,vals_len):
        void_list = [0] * vals_len
        void_list[indx] = 1
        selected_indx = []
        for i in range(len(void_list)):
            mon = i % 12 + 1
            if mon in gs_mons:
                selected_indx.append(void_list[i])
        if 1 in selected_indx:
            trans_indx = selected_indx.index(1)
            return trans_indx
        else:
            return None
        pass

    def __pick_gs_vals(self,vals,gs_mons):
        picked_vals = []
        for i in range(len(vals)):
            mon = i % 12 + 1
            if mon in gs_mons:
                picked_vals.append(vals[i])
        return picked_vals


    def __split_999999(self,selected_indx):
        selected_indx_ = []
        selected_indx_s = []
        for i in selected_indx:
            if i > 9999:
                if len(selected_indx_) > 0:
                    selected_indx_s.append(selected_indx_)
                selected_indx_ = []
                continue
            else:
                selected_indx_.append(i)
        if len(selected_indx_s) == 0:
            return None
        return selected_indx_s[0]
        pass

    def search1(self, ndvi,drought_indx, event_start_index,search_end_indx):
        # print(event_start_index)
        if event_start_index+search_end_indx >= len(ndvi):
            return None
        selected_indx = []
        for i in range(event_start_index,event_start_index+search_end_indx):
            ndvi_i = ndvi[i]
            if ndvi_i < 0:
                selected_indx.append(i)
            else:
                selected_indx.append(999999)

        recovery_indx_gs = self.__split_999999(selected_indx)
        return recovery_indx_gs
        # plt.scatter(event_start_index, [0])
        # plt.plot(ndvi)
        # plt.plot(drought_indx)
        # plt.show()

        pass

class Main_flow_Greenness_loss:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr/Main_flow_Greenness_loss/'
        self.this_class_tif = results_root_main_flow + 'tif/Main_flow_Greenness_loss/'
        self.this_class_png = results_root_main_flow + 'png/Main_flow_Greenness_loss/'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        # 1 cal recovery time
        event_dic,spei_dic,sif_dic = self.load_data()
        out_dir = self.this_class_arr + 'Recovery_time_Legacy/'
        self.gen_recovery_time_legacy(event_dic,spei_dic, sif_dic,out_dir)
        pass

    def load_data(self,condition=''):
        # events_dir = results_root_main_flow + 'arr/SPEI_preprocess/drought_events/'
        # SPEI_dir = data_root + 'SPEI/per_pix_clean/'
        # SIF_dir = data_root + 'CSIF/per_pix_anomaly_detrend/'

        events_dir = Main_Flow_Pick_drought_events().this_class_arr + 'drought_events/'
        SPEI_dir = data_root + 'SPEI/per_pix_2002/'
        # SPEI_dir = data_root + 'SPEI/per_pix_for_modis/'
        # SIF_dir = data_root + '/CSIF/per_pix_anomaly_180/'
        SIF_dir = data_root + 'NDVI/per_pix_anomaly_clean_detrend/'

        event_dic = T.load_npy_dir(events_dir,condition)
        spei_dic = T.load_npy_dir(SPEI_dir,condition)
        sif_dic = T.load_npy_dir(SIF_dir,condition)

        sif_dic_new = {}

        for pix in sif_dic:
            vals = sif_dic[pix]
            inserted_vals = [np.nan] * 12
            vals_new = np.insert(vals,0,inserted_vals)
            sif_dic_new[pix] = vals_new

        return event_dic,spei_dic,sif_dic_new
        pass


    def __cal_legacy(self,ndvi_obs,recovery_range):
        selected_obs = T.pick_vals_from_1darray(ndvi_obs,recovery_range)
        diff = selected_obs
        legacy = np.sum(diff)
        return legacy
        pass

    def gen_recovery_time_legacy(self, events, spei_dic, ndvi_dic, out_dir):
        '''
        生成全球恢复期
        :param interval: SPEI_{interval}
        :return:
        '''

        # pre_dic = Main_flow_Prepare().load_X_anomaly('PRE')

        growing_date_range = list(range(5,11))
        Tools().mk_dir(out_dir, force=True)
        outf = out_dir + 'recovery_time_legacy'
        # 1 加载事件
        # interval = '%02d' % interval
        # 2 加载NDVI 与 SPEI
        recovery_time_dic = {}

        for pix in tqdm(ndvi_dic):

            if pix in events:
                ndvi = ndvi_dic[pix]
                ndvi = np.array(ndvi)
                if not pix in spei_dic:
                    continue
                spei = spei_dic[pix]
                spei = np.array(spei)
                event = events[pix]
                recovery_time_result = []
                for date_range in event:
                    # print(date_range)
                    event_start_index = T.pick_min_indx_from_1darray(spei, date_range)
                    event_start_index_trans = self.__drought_indx_to_gs_indx(event_start_index,growing_date_range,len(ndvi))
                    if event_start_index_trans == None:
                        continue
                    ndvi_gs = self.__pick_gs_vals(ndvi,growing_date_range)
                    spei_gs = self.__pick_gs_vals(spei,growing_date_range)
                    # ndvi_gs_pred = self.__pick_gs_vals(ndvi_pred,growing_date_range)
                    # print(len(ndvi_gs))
                    # print(len(spei_gs))
                    # print(len(ndvi_gs_pred))
                    date_range_new = []
                    for i in date_range:
                        i_trans = self.__drought_indx_to_gs_indx(i,growing_date_range,len(ndvi))
                        if i_trans != None:
                            date_range_new.append(i_trans)
                    # 1 挑出此次干旱事件的NDVI和SPEI值 （画图需要）
                    # spei_picked_vals = Tools().pick_vals_from_1darray(spei, date_range)
                    # 2 挑出此次干旱事件SPEI最低的索引
                    # 在当前生长季搜索

                    # 4 搜索恢复期
                    # 4.1 获取growing season NDVI的最小值
                    # 4.3 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
                    search_end_indx = 3 * len(growing_date_range)
                    recovery_range = self.search1(ndvi_gs, spei_gs, event_start_index_trans,search_end_indx)
                    # continue
                    # recovery_time, lag, recovery_start_gs, recovery_start, 'undefined'
                    ###########################################
                    ###########################################
                    ###########################################
                    if recovery_range == None:
                        continue
                    recovery_range = np.array(recovery_range)
                    date_range_new = np.array(date_range_new)

                    # exit()
                    recovery_time = len(recovery_range)
                    legacy = self.__cal_legacy(ndvi_gs,recovery_range)

                    # print(recovery_range)
                    # recovery_range = recovery_range + len(growing_date_range)
                    # date_range_new = date_range_new + len(growing_date_range)
                    # print(recovery_range)
                    # print(date_range_new)
                    # print('*******')

                    recovery_time_result.append({
                        'recovery_time': recovery_time,
                        'recovery_date_range': recovery_range,
                        'drought_event_date_range': date_range_new,
                        'greenness_loss': legacy,
                    })
                    #
                    # ################# plot ##################
                    # print('recovery_time',recovery_time)
                    # print('growing_date_range',growing_date_range)
                    # print('recovery_range',recovery_range)
                    # print('legacy',legacy)
                    # recovery_date_range = recovery_range
                    # recovery_ndvi = Tools().pick_vals_from_1darray(ndvi_gs, recovery_date_range)
                    #
                    # # pre_picked_vals = Tools().pick_vals_from_1darray(pre, tmp_pre_date_range)
                    # # tmp_picked_vals = Tools().pick_vals_from_1darray(tmp, tmp_pre_date_range)
                    # # if len(swe) == 0:
                    # #     continue
                    # # swe_picked_vals = Tools().pick_vals_from_1darray(swe, tmp_pre_date_range)
                    #
                    # plt.figure(figsize=(8, 6))
                    # # plt.plot(tmp_pre_date_range, pre_picked_vals, '--', c='blue', label='precipitation')
                    # # plt.plot(tmp_pre_date_range, tmp_picked_vals, '--', c='cyan', label='temperature')
                    # # plt.plot(tmp_pre_date_range, swe_picked_vals, '--', c='black', linewidth=2, label='SWE',
                    # #          zorder=99)
                    # # plt.plot(recovery_date_range, recovery_ndvi, c='g', linewidth=6, label='Recovery Period')
                    # plt.scatter(recovery_date_range, recovery_ndvi, c='g', label='Recovery Period')
                    # # plt.plot(date_range, spei_picked_vals, c='r', linewidth=6,
                    # #          label='drought Event')
                    # # plt.scatter(date_range, spei_picked_vals, c='r', zorder=99)
                    #
                    # plt.plot(range(len(ndvi_gs)), ndvi_gs, '--', c='g', zorder=99, label='ndvi')
                    # plt.plot(range(len(spei_gs)), spei_gs, '--', c='r', zorder=99, label='drought index')
                    # # plt.plot(range(len(pre)), pre, '--', c='blue', zorder=99, label='Precip')
                    # # pre_picked = T.pick_vals_from_1darray(pre,recovery_date_range)
                    # # pre_mean = np.mean(pre_picked)
                    # # plt.plot(recovery_date_range,[pre_mean]*len(recovery_date_range))
                    # plt.legend()
                    #
                    # minx = 9999
                    # maxx = -9999
                    #
                    # for ii in recovery_date_range:
                    #     if ii > maxx:
                    #         maxx = ii
                    #     if ii < minx:
                    #         minx = ii
                    #
                    # for ii in date_range_new:
                    #     if ii > maxx:
                    #         maxx = ii
                    #     if ii < minx:
                    #         minx = ii
                    #
                    # # xtick = []
                    # # for iii in np.arange(len(ndvi)):
                    # #     year = 1982 + iii / 12
                    # #     mon = iii % 12 + 1
                    # #     mon = '%02d' % mon
                    # #     xtick.append('{}.{}'.format(year, mon))
                    # # # plt.xticks(range(len(xtick))[::3], xtick[::3], rotation=90)
                    # # plt.xticks(range(len(xtick)), xtick, rotation=90)
                    # plt.grid()
                    # plt.xlim(minx - 5, maxx + 5)
                    #
                    # lon, lat, address = Tools().pix_to_address(pix)
                    # print(address)
                    # # exit()
                    # try:
                    #     plt.rcParams['font.sans-serif'] = ['Hanzipen']
                    #     plt.rc('font', family='Hanzipen')
                    #     plt.title('lon:{:0.2f} lat:{:0.2f} address:{}\n'.format(lon, lat, address) +
                    #               'recovery_time:'+str(recovery_time)
                    #               )
                    #
                    # except:
                    #     plt.title('lon:{:0.2f} lat:{:0.2f}\n'.format(lon, lat)+
                    #               'recovery_time:' + str(recovery_time)
                    #               )
                    # plt.figure()
                    # DIC_and_TIF().show_pix(pix)
                    # plt.show()
            #         # #################plot end ##################
                recovery_time_dic[pix] = recovery_time_result
            else:
                recovery_time_dic[pix] = []
        T.save_dict_to_binary(recovery_time_dic, outf)
        pass


    def __drought_indx_to_gs_indx(self,indx,gs_mons,vals_len):
        void_list = [0] * vals_len
        void_list[indx] = 1
        selected_indx = []
        for i in range(len(void_list)):
            mon = i % 12 + 1
            if mon in gs_mons:
                selected_indx.append(void_list[i])
        if 1 in selected_indx:
            trans_indx = selected_indx.index(1)
            return trans_indx
        else:
            return None
        pass

    def __pick_gs_vals(self,vals,gs_mons):
        picked_vals = []
        for i in range(len(vals)):
            mon = i % 12 + 1
            if mon in gs_mons:
                picked_vals.append(vals[i])
        return picked_vals


    def __split_999999(self,selected_indx):
        selected_indx_ = []
        selected_indx_s = []
        for i in selected_indx:
            if i > 9999:
                if len(selected_indx_) > 0:
                    selected_indx_s.append(selected_indx_)
                selected_indx_ = []
                continue
            else:
                selected_indx_.append(i)
        if len(selected_indx_s) == 0:
            return None
        return selected_indx_s[0]
        pass

    def search1(self, ndvi,drought_indx, event_start_index,search_end_indx):
        # print(event_start_index)
        if event_start_index+search_end_indx >= len(ndvi):
            return None
        selected_indx = []
        for i in range(event_start_index,event_start_index+search_end_indx):
            ndvi_i = ndvi[i]
            if ndvi_i < 0:
                selected_indx.append(i)
            else:
                selected_indx.append(999999)

        recovery_indx_gs = self.__split_999999(selected_indx)
        return recovery_indx_gs
        # plt.scatter(event_start_index, [0])
        # plt.plot(ndvi)
        # plt.plot(drought_indx)
        # plt.show()

        pass


class Main_flow_Dataframe_NDVI_SPEI_legacy:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr/Main_flow_Dataframe_NDVI_SPEI_legacy/'
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'data_frame.df'

    def run(self):
        # 0 generate a void dataframe
        df = self.__gen_df_init()
        # self._check_spatial(df)
        # exit()
        # 1 add drought event and delta legacy into df
        df = self.Carbon_loss_to_df(df)
        # print(df)
        # df = self.add_legacy_123_to_df(df)
        # df = self.add_greenness_loss_to_df(df)
        # 2 add lon lat into df
        df = self.add_lon_lat_to_df(df)
        # 3 add iso-hydricity into df
        df = self.add_isohydricity_to_df(df)
        # 4 add correlation into df
        # df = self.add_gs_sif_spei_correlation_to_df(df)
        # 5 add canopy height into df
        # df = self.add_canopy_height_to_df(df)
        # 6 add rooting depth into df
        # df = self.add_rooting_depth_to_df(df)
        # 7 add TWS into df
        df = self.add_TWS_to_df(df)
        # 8 add is gs into df
        # self.add_is_gs_drought_to_df(df)
        # 9 add landcover to df
        # 10 add kplc to df
        df = self.add_koppen_landuse_to_df(df)
        # 11 add koppen to df
        df = self.add_split_landuse_and_kp_to_df(df)
        # 12 add climate delta and cv into df
        # df = self.add_climate_delta_to_df(df)
        # df = self.add_climate_cv_to_df(df)
        # df = self.add_climate_delta_cv_to_df(df)
        # df = self.add_climate_trend_to_df(df)
        # 13 add sand to df
        # 14 add waterbalance to df
        # df = self.add_waterbalance(df)
        # 15 add corr to df
        # df = self.add_gs_sif_spei_correlation_to_df(df)
        # 16 add sand to df
        # df = self.add_sand(df)
        # 17 add delta spei to df
        # df = self.add_delta_SPEI(df)
        # 18 add awc to df
        # df = self.add_AWC_to_df(df)
        # 19 add forest ratio to df
        # df = self.add_forest_ratio_in_df(df)
        # 20 add phenology info to df (from recovery project)
        # df = self.add_drought_year_SOS(df)
        # 21 add thaw data into df
        # df = self.add_thaw_date(df)
        # df = self.add_thaw_date_anomaly(df)
        # df = self.add_thaw_date_std_anomaly(df)
        # 22 add temp trend to df
        # df = self.add_temperature_trend_to_df(df)
        # 23 add MAT MAP to df
        df = self.add_MAT_MAP_to_df(df)
        # -1 df to excel
        # add pre_drought vars
        for n in [3,6,12,24]:
            # print(n)
            df = self.add_pre_drought_growth_variables(df,n)
        for n in [3, 6, 12, 24]:
            df = self.add_pre_drought_climate_variables(df,n)
        for n in [3,6,12,24]:
            self.add_pre_drought_TWS(df,n)
        for n in [3,6,12,24]:
            self.add_pre_drought_SM(df,n)
        T.save_df(df,self.dff)
        self.__df_to_excel(df,self.dff)
        pass


    def _check_spatial(self,df):
        spatial_dic = {}
        for i,row in df.iterrows():
            pix = row.pix
            spatial_dic[pix] = row.lon
            # spatial_dic[pix] = row.isohydricity
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(arr)
        plt.show()
        pass

    def __load_HI(self):
        HI_tif = data_root + '/waterbalance/HI_difference.tif'
        HI_arr = to_raster.raster2array(HI_tif)[0]
        HI_arr[HI_arr < -9999] = np.nan
        HI_dic = DIC_and_TIF().spatial_arr_to_dic(HI_arr)
        return HI_dic

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

    def __gen_df_init(self):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __df_to_excel(self,df,dff,head=1000):
        if head == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            df = df.head(head)
            df.to_excel('{}.xlsx'.format(dff))

        pass

    def add_gs_sif_spei_correlation_to_df(self,df):

        corr_dic = Correlation_CSIF_SPEI().correlation()
        corr_list = []
        corr_p_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            if not pix in corr_dic:
                corr_list.append(np.nan)
                corr_p_list.append(np.nan)
                continue
            corr,p = corr_dic[pix]
            corr_list.append(corr)
            corr_p_list.append(p)
        df['gs_sif_spei_corr'] = corr_list
        df['gs_sif_spei_corr_p'] = corr_p_list


        return df


    def add_koppen_landuse_to_df(self,df):
        kp_f = data_root + 'Koppen/cross_koppen_landuse_pix.npy'
        koppen_landuse_dic = T.load_npy(kp_f)
        koppen_landuse_dic_set = {}
        for kl in koppen_landuse_dic:
            pixs = koppen_landuse_dic[kl][0]
            pixs = set(pixs)
            koppen_landuse_dic_set[kl] = pixs

        kl_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix_ = row.pix
            kl_ = None
            for kl in koppen_landuse_dic_set:
                pixs = koppen_landuse_dic_set[kl]
                if pix_ in pixs:
                    kl_ = kl
                    break
            kl_list.append(kl_)
        df['climate_zone'] = kl_list
        return df

    def add_split_landuse_and_kp_to_df(self,df):
        # df_f = Prepare_CWD_X_pgs_egs_lgs2().this_class_arr + 'prepare/data_frame.df'
        # df = T.load_df(df_f)
        kp_list = []
        lc_list = []
        for _,row in tqdm(df.iterrows(),total=len(df),desc='adding landuse and koppen to df columns'):
            kl = row.climate_zone
            if kl == None:
                kp_list.append(None)
                lc_list.append(None)
                continue
            lc,kp = kl.split('.')
            kp_list.append(kp)
            lc_list.append(lc)
        df['kp'] = kp_list
        df['lc'] = lc_list
        return df

    def drop_duplicated_sample(self,df):
        df_drop_dup = df.drop_duplicates(subset=['pix','carbon_loss','recovery_date_range'])
        return df_drop_dup
        # df_drop_dup.to_excel(self.this_class_arr + 'drop_dup.xlsx')
        pass

    def Carbon_loss_to_df(self,df):
        f = Main_flow_Carbon_loss().this_class_arr + 'Recovery_time_Legacy/recovery_time_legacy.pkl'
        events_dic = T.load_dict_from_binary(f)
        # print(events_dic)
        # exit()
        pix_list = []
        recovery_time_list = []
        drought_event_date_range_list = []
        recovery_date_range_list = []
        legacy_list = []

        for pix in tqdm(events_dic,desc='1. carbon_loss_to_df'):
            # print(pix)
            # exit()
            events = events_dic[pix]
            for event in events:
                recovery_time = event['recovery_time']
                drought_event_date_range = event['drought_event_date_range']
                recovery_date_range = event['recovery_date_range']
                legacy = event['carbon_loss']

                drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(drought_event_date_range)
                recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(recovery_date_range)

                pix_list.append(pix)
                recovery_time_list.append(recovery_time)
                drought_event_date_range_list.append(tuple(drought_event_date_range))
                recovery_date_range_list.append(tuple(recovery_date_range))
                legacy_list.append(legacy)
        # print(pix_list)
        # exit()
        df['pix'] = pix_list
        df['drought_event_date_range'] = drought_event_date_range_list
        df['recovery_date_range'] = recovery_date_range_list
        df['recovery_time'] = recovery_time_list
        df['carbon_loss'] = legacy_list
        # print(df)
        # exit()
        return df
        pass


    def add_legacy_123_to_df(self,df):
        fdir = Main_flow_Legacy().this_class_arr+'/'

        for n in range(1,4):
            f = fdir + 'legacy_dic_{}.npy'.format(n)
            legacy_dic = T.load_npy(f)
            legacy_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc='adding legacy {} to df'.format(n)):
                pix = row.pix
                drought_event_date_range = row.drought_event_date_range
                evt_start = drought_event_date_range[0]
                # print(legacy_dic[pix])
                evt_dic_i = legacy_dic[pix]
                if not evt_start in evt_dic_i:
                    legacy_list.append(np.nan)
                    continue
                legacy = evt_dic_i[evt_start]
                legacy_list.append(legacy)
            df['legacy_{}'.format(n)] = legacy_list

        return df



        pass

    def add_greenness_loss_to_df(self,df):
        # greenness_loss
        f = Main_flow_Greenness_loss().this_class_arr + 'Recovery_time_Legacy/recovery_time_legacy.pkl'
        dic = T.load_dict_from_binary(f)
        # for key in dic:
        #     print(key)
        #     events = dic[key]
        #     for evt in events:
        #         print(evt)
        # exit()
        greenness_loss_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc='adding greenness loss to df'):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            evt_start = drought_event_date_range[0]
            # print(legacy_dic[pix])
            if not pix in dic:
                greenness_loss_list.append(np.nan)
                continue
            events = dic[pix]
            success = 0
            for evt in events:
                drought_event_date_range_gl_start = evt['drought_event_date_range'][0]
                greenness_loss = evt['greenness_loss']
                if drought_event_date_range_gl_start == evt_start:
                    greenness_loss_list.append(greenness_loss)
                    success += 1
                    if success > 1:
                        for e in events:
                            print(e)
                        exit()
            if success == 0:
                greenness_loss_list.append(np.nan)
        # print(greenness_loss_list)
        # exit()
        df['greenness_loss'] = greenness_loss_list

        return df


    def delta_legacy_to_df(self,df):
        part1_dic = DIC_and_TIF().void_spatial_dic()
        part2_dic = DIC_and_TIF().void_spatial_dic()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            gs_mon = Global_vars().gs_mons()
            pix = row.pix
            legacy = row.legacy
            drought_event_date_range = row.drought_event_date_range
            drought_start = drought_event_date_range[0]
            drought_year = drought_start // len(gs_mon)
            half = (2015 - 1982 + 1) / 2
            if drought_year < half:
                part1_dic[pix].append(legacy)
            else:
                part2_dic[pix].append(legacy)
        delta_legacy_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            part1 = part1_dic[pix]
            part2 = part2_dic[pix]
            p1_mean = np.mean(part1)
            p2_mean = np.mean(part2)
            delta_legacy = p2_mean - p1_mean
            delta_legacy_list.append(delta_legacy)

        df['delta_legacy'] = delta_legacy_list
        return df
        pass


    def legacy_trend_to_df(self,df):
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            legacy = row.legacy
            drought_event_date_range = row.drought_event_date_range
            drought_start = drought_event_date_range[0]
            spatial_dic[pix].append((drought_start,legacy))

        result_dic = {}
        for pix in tqdm(spatial_dic):
            vals = spatial_dic[pix]
            if len(vals) <= 2:
                continue
            x_list = []
            y_list = []
            for x,y in vals:
                x_list.append(x)
                y_list.append(y)
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            reg = LinearRegression()
            x_list = x_list.reshape((-1,1))
            reg.fit(x_list,y_list)
            coef = reg.coef_[0]
            score = reg.score(x_list,y_list)
            result_dic[pix] = (coef,score)
            # plt.scatter(x_list,y_list)
            # plt.title('len(vals) {} coef: {} score: {}'.format(len(vals),coef,score))
            # plt.show()
        coef_list = []
        score_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            if not pix in result_dic:
                coef_list.append(np.nan)
                score_list.append(np.nan)
                continue
            coef, score = result_dic[pix]
            coef_list.append(coef)
            score_list.append(score)
        df['trend'] = coef_list
        df['trend_score'] = score_list
        return df
        pass


    def add_lon_lat_to_df(self, df):
        # DIC_and_TIF().spatial_tif_to_lon_lat_dic()
        pix_to_lon_lat_dic_f = DIC_and_TIF().this_class_arr + 'pix_to_lon_lat_dic.npy'
        lon_lat_dic = T.load_npy(pix_to_lon_lat_dic_f)
        # print(pix)
        lon_list = []
        lat_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc='adding lon lat into df'):
            pix = row.pix
            lon, lat = lon_lat_dic[pix]
            lon_list.append(lon)
            lat_list.append(lat)
        df['lon'] = lon_list
        df['lat'] = lat_list

        return df

    def add_isohydricity_to_df(self,df):
        tif = data_root + 'Isohydricity/tif_all_year/ISO_Hydricity.tif'
        dic = DIC_and_TIF().spatial_tif_to_dic(tif)
        iso_hyd_list = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='adding iso-hydricity to df'):
            pix = row.pix
            if not pix in dic:
                iso_hyd_list.append(np.nan)
                continue
            isohy = dic[pix]
            iso_hyd_list.append(isohy)
        df['isohydricity'] = iso_hyd_list

        return df


    def add_gs_sif_spei_correlation_to_df(self,df):

        corr_dic = Correlation_CSIF_SPEI().correlation()
        corr_list = []
        corr_p_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            if not pix in corr_dic:
                corr_list.append(np.nan)
                corr_p_list.append(np.nan)
                continue
            corr,p = corr_dic[pix]
            corr_list.append(corr)
            corr_p_list.append(p)
        df['gs_sif_spei_corr'] = corr_list
        df['gs_sif_spei_corr_p'] = corr_p_list


        return df


    def add_canopy_height_to_df(self,df):
        f = data_root + 'Canopy_Height/per_pix/Canopy_Height.npy'
        dic = T.load_npy(f)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            if not pix in dic:
                val_list.append(np.nan)
                continue

            val = dic[pix]
            val_list.append(val)

        df['canopy_height'] = val_list
        return df

    def add_rooting_depth_to_df(self,df):
        f = data_root + 'rooting_depth/per_pix/rooting_depth.npy'
        dic = T.load_npy(f)

        val_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix

            if not pix in dic:
                val_list.append(np.nan)
                continue

            val = dic[pix]
            val_list.append(val)

        df['rooting_depth'] = val_list
        return df

    def add_TWS_to_df(self,df):

        fdir = data_root + 'TWS/GRACE/per_pix/'
        tws_dic = T.load_npy_dir(fdir)
        tws_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            recovery_date_range = row['recovery_date_range']
            pix = row.pix
            if not pix in tws_dic:
                tws_list.append(np.nan)
                continue
            vals = tws_dic[pix]
            picked_val = T.pick_vals_from_1darray(vals,recovery_date_range)
            picked_val[picked_val<-999]=np.nan
            mean = np.nanmean(picked_val)
            tws_list.append(mean)
        df['TWS_recovery_period'] = tws_list

        # exit()
        return df

    def add_is_gs_drought_to_df(self,df):

        is_gs_list = []
        gs_mon = list(range(4,11))
        for i,row in tqdm(df.iterrows(),total=len(df)):
            drought_start = row.event[0]
            mon = drought_start % 12 + 1
            if mon in gs_mon:
                is_gs = 1
            else:
                is_gs = 0
            is_gs_list.append(is_gs)
        df['is_gs'] = is_gs_list
        return df

        pass

    def add_climate_delta_to_df(self,df):
        fdir = data_root + r'Climate_408/'
        for climate in os.listdir(fdir):
            f = fdir + climate + '/delta/delta.npy'
            dic = T.load_npy(f)
            val_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=climate):
                pix = row.pix
                if not pix in dic:
                    val_list.append(np.nan)
                    continue
                val = dic[pix]
                val_list.append(val)
            df['{}_delta'.format(climate)]=val_list
        # exit()
        return df
        pass

    def add_climate_cv_to_df(self, df):
        fdir = data_root + r'Climate_408/'
        for climate in os.listdir(fdir):
            f = fdir + climate + '/CV/CV.npy'
            dic = T.load_npy(f)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df), desc=climate):
                pix = row.pix
                if not pix in dic:
                    val_list.append(np.nan)
                    continue
                val = dic[pix]
                val_list.append(val)
            df['{}_cv'.format(climate)] = val_list
        # exit()
        return df
        pass

    def add_climate_delta_cv_to_df(self, df):
        fdir = data_root + r'Climate_408/'
        for climate in os.listdir(fdir):
            f = fdir + climate + '/CV/CV_delta.npy'
            dic = T.load_npy(f)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df), desc=climate):
                pix = row.pix
                if not pix in dic:
                    val_list.append(np.nan)
                    continue
                val = dic[pix]
                val_list.append(val)
            df['{}_cv_delta'.format(climate)] = val_list
        # exit()
        return df
        pass

    def add_climate_trend_to_df(self,df):
        fdir = data_root + r'Climate_408/'
        for climate in os.listdir(fdir):
            f = fdir + climate + '/trend/trend.npy'
            dic = T.load_npy(f)
            val_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=climate):
                pix = row.pix
                if not pix in dic:
                    val_list.append(np.nan)
                    continue
                val = dic[pix]
                val_list.append(val)
            df['{}_trend'.format(climate)]=val_list
        # exit()
        return df
        pass

    def add_waterbalance(self,df):
        wb_dic = self.__load_HI()
        wb_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            if pix in wb_dic:
                wb = wb_dic[pix]
                wb_list.append(wb)
            else:
                wb_list.append(np.nan)
        df['waterbalance'] = wb_list
        return df

    def __load_soil(self):
        sand_tif = data_root + 'HWSD/T_SAND_resample.tif'
        silt_tif = data_root + 'HWSD/T_SILT_resample.tif'
        clay_tif = data_root + 'HWSD/T_CLAY_resample.tif'

        sand_arr = to_raster.raster2array(sand_tif)[0]
        silt_arr = to_raster.raster2array(silt_tif)[0]
        clay_arr = to_raster.raster2array(clay_tif)[0]

        sand_arr[sand_arr < -9999] = np.nan
        silt_arr[silt_arr < -9999] = np.nan
        clay_arr[clay_arr < -9999] = np.nan

        sand_dic = DIC_and_TIF().spatial_arr_to_dic(sand_arr)
        silt_dic = DIC_and_TIF().spatial_arr_to_dic(silt_arr)
        clay_dic = DIC_and_TIF().spatial_arr_to_dic(clay_arr)

        return sand_dic

    def add_sand(self,df):
        sand_dic = self.__load_soil()
        sand_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if pix in sand_dic:
                sand = sand_dic[pix]
                sand_list.append(sand)
            else:
                sand_list.append(np.nan)
        df['sand'] = sand_list
        return df
        pass

    def add_delta_SPEI(self,df):
        f = data_root + 'SPEI/delta/delta.npy'
        dic = T.load_npy(f)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc='add_delta_SPEI'):
            pix = row.pix
            if not pix in dic:
                val_list.append(np.nan)
                continue
            val = dic[pix]
            val_list.append(val)
        df['{}_delta'.format('SPEI')] = val_list
        return df
        pass

    def add_AWC_to_df(self,df):
        awc_class_val_dic = {
            1:150,
            2:125,
            3:100,
            4:75,
            5:50,
            6:15,
            7:0,
            127:np.nan,
        }
        tif = data_root + 'HWSD/awc_05.tif'
        arr = to_raster.raster2array(tif)[0]
        spatial_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        awc_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            awc_cls = spatial_dic[pix]
            awc = awc_class_val_dic[awc_cls]
            awc_list.append(awc)
        df['awc'] = awc_list
        return df


    def add_AGB_to_df(self,df):
        sand_dic = self.__load_soil()
        sand_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if pix in sand_dic:
                sand = sand_dic[pix]
                sand_list.append(sand)
            else:
                sand_list.append(np.nan)
        df['AGB'] = sand_list
        return df
        pass


    def add_forest_ratio_in_df(self,df):
        ratio_f = Main_flow_pick_pure_forest_pixels().this_class_tif + 'ratio_of_forest.tif'
        arr = to_raster.raster2array(ratio_f)[0]
        spatial_dic = DIC_and_TIF().spatial_arr_to_dic(arr)

        ratio_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if pix in spatial_dic:
                ratio = spatial_dic[pix]
                ratio_list.append(ratio)
            else:
                ratio_list.append(np.nan)
        df['ratio_of_forest'] = ratio_list

        return df

    def add_drought_year_SOS(self, df):
        sos_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                'transform_early_peak_late_dormant_period_annual/early_start.npy'
        sos_dic = T.load_npy(sos_f)
        sos_list = []
        sos_anomaly_list = []
        sos_std_anomaly_list = []
        for indx, row in tqdm(df.iterrows(), total=len(df), desc='add_current_season_SOS...'):
            pix = row['pix']
            drought_start = row['drought_event_date_range'][0]
            # winter_mark = row['winter_mark']
            year_index = drought_start // 12
            # if winter_mark == 1:
            #     year_index += 1
            # else:
            #     sos_list.append(None)
            #     sos_anomaly_list.append(None)
            #     sos_std_anomaly_list.append(None)
            #     continue
            sos = sos_dic[pix]
            if len(sos) != 34:
                sos_list.append(None)
                sos_anomaly_list.append(None)
                sos_std_anomaly_list.append(None)
                continue
            sos_anomaly, sos_std_anomaly = self.__cal_phe_anomaly(sos_dic, pix)
            sos_i = sos[year_index]
            sos_anomaly_i = sos_anomaly[year_index]
            sos_std_anomaly_i = sos_std_anomaly[year_index]

            sos_list.append(sos_i)
            sos_anomaly_list.append(sos_anomaly_i)
            sos_std_anomaly_list.append(sos_std_anomaly_i)

        df['drought_year_sos'] = sos_list
        df['drought_year_sos_anomaly'] = sos_anomaly_list
        df['drought_year_sos_std_anomaly'] = sos_std_anomaly_list

        return df

    def __cal_phe_anomaly(self, dic, pix):

        var_list = []
        for y in range(len(dic[pix])):
            var_i = dic[pix][y]
            var_list.append(var_i)

        std_anomaly = {}
        anomaly = {}
        for i in range(len(var_list)):
            std = np.std(var_list)
            mean = np.mean(var_list)
            anomaly_i = var_list[i] - mean
            std_anomaly_i = (var_list[i] - mean) / std
            anomaly[i] = anomaly_i
            std_anomaly[i] = std_anomaly_i
        return anomaly, std_anomaly

    def add_thaw_date(self,df):
        fdir = data_root + 'GLOBSWE/thaw_tif/'
        thaw_year_dic = {}
        for tif in tqdm(os.listdir(fdir),desc='loading thaw tif ...'):
            year = tif.split('.')[0]
            year = int(year)
            arr = to_raster.raster2array(fdir + tif)[0]
            spatial_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            thaw_year_dic[year] = spatial_dic

        thaw_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            drought_year = drought_event_date_range[0]//12 + 1982
            thaw = thaw_year_dic[drought_year][pix]
            if thaw <= 0:
                thaw_list.append(np.nan)
            else:
                thaw_list.append(thaw)

        df['thaw_date'] = thaw_list

        return df

    def add_thaw_date_std_anomaly(self,df):
        fdir = data_root + 'GLOBSWE/thaw_tif_std_anomaly/'
        thaw_year_dic = {}
        for tif in tqdm(sorted(os.listdir(fdir)),desc='loading thaw tif ...'):
            # print(tif)
            if not tif.endswith('tif'):
                continue
            year = tif.split('.')[0]
            year = int(year)
            arr = to_raster.raster2array(fdir + tif)[0]
            spatial_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            thaw_year_dic[year] = spatial_dic
        # exit()
        thaw_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            drought_year = drought_event_date_range[0]//12 + 1982
            thaw = thaw_year_dic[drought_year][pix]
            if thaw < -9999:
                thaw_list.append(np.nan)
            else:
                thaw_list.append(thaw)

        df['thaw_date_std_anomaly'] = thaw_list

        return df

    def add_thaw_date_anomaly(self,df):
        fdir = data_root + 'GLOBSWE/thaw_tif_anomaly/'
        thaw_year_dic = {}
        for tif in tqdm(sorted(os.listdir(fdir)),desc='loading thaw tif ...'):
            # print(tif)
            if not tif.endswith('tif'):
                continue
            year = tif.split('.')[0]
            year = int(year)
            arr = to_raster.raster2array(fdir + tif)[0]
            spatial_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            thaw_year_dic[year] = spatial_dic
        # exit()
        thaw_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            drought_year = drought_event_date_range[0]//12 + 1982
            thaw = thaw_year_dic[drought_year][pix]
            if thaw < -9999:
                thaw_list.append(np.nan)
            else:
                thaw_list.append(thaw)
        df['thaw_date_anomaly'] = thaw_list

        return df


    def add_temperature_trend_to_df(self,df):
        tem_trend_tif = results_root_main_flow + 'tif/Tif/annual_tmp_trend/annual_tmp_trend.tif'
        temp_trend_arr = to_raster.raster2array(tem_trend_tif)[0]
        tmp_trend_dic = DIC_and_TIF().spatial_arr_to_dic(temp_trend_arr)
        tmp_trend_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if pix in tmp_trend_dic:
                tmp_trend = tmp_trend_dic[pix]
                tmp_trend_list.append(tmp_trend)
            else:
                tmp_trend_list.append(np.nan)
        df['temp_trend_list'] = tmp_trend_list

        return df
        pass

    def add_MAT_MAP_to_df(self,df):
        tif = data_root + 'Climate_408/PRE/MAPRE.tif'
        arr = to_raster.raster2array(tif)[0]
        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if pix in dic:
                tmp_trend = dic[pix]
                val_list.append(tmp_trend)
            else:
                val_list.append(np.nan)
        df['MAP'] = val_list

        tif = data_root + 'Climate_408/TMP/MATMP.tif'
        arr = to_raster.raster2array(tif)[0]
        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if pix in dic:
                tmp_trend = dic[pix]
                val_list.append(tmp_trend)
            else:
                val_list.append(np.nan)
        df['MAT'] = val_list

        return df


    def add_pre_drought_growth_variables(self,df,n=3):

        '''
        add CSIF NDVI
        '''

        # NDVI_dir = data_root + 'NDVI/per_pix/'
        NDVI_dir = data_root + 'NDVI/per_pix_anomaly_clean_detrend_180/'
        CSIF_dir = data_root + 'CSIF/per_pix_anomaly_180/'

        ndvi_dic = T.load_npy_dir(NDVI_dir)
        csif_dic = T.load_npy_dir(CSIF_dir)
        # exit()
        NDVI_picked_list = []
        CSIF_picked_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            if not pix in ndvi_dic:
                NDVI_picked_list.append(np.nan)
                CSIF_picked_list.append(np.nan)
                continue
            if not pix in csif_dic:
                NDVI_picked_list.append(np.nan)
                CSIF_picked_list.append(np.nan)
                continue
            NDVI = ndvi_dic[pix]
            CSIF = csif_dic[pix]
            drought_start = row['drought_event_date_range'][0]
            picked_index = []
            for ni in range(1,n+1):
                indx = drought_start - ni
                if indx < 0:
                    picked_index = []
                    break
                picked_index.append(indx)
            # if len(picked_index) == 0:
            #     print(drought_start)
            picked_index = picked_index[::-1]
            NDVI_picked = T.pick_vals_from_1darray(NDVI,picked_index)
            CSIF_picked = T.pick_vals_from_1darray(CSIF,picked_index)
            NDVI_picked_mean = np.nanmean(NDVI_picked)
            CSIF_picked_mean = np.nanmean(CSIF_picked)
            NDVI_picked_list.append(NDVI_picked_mean)
            CSIF_picked_list.append(CSIF_picked_mean)
        df['NDVI_pre_{}'.format(n)] = NDVI_picked_list
        df['CSIF_pre_{}'.format(n)] = CSIF_picked_list
        return df

        pass
    def add_pre_drought_climate_variables(self,df,n=3):

        '''
        add precip temp and VPD
        '''

        fdir = data_root + 'Climate_180/'
        for var in os.listdir(fdir):
            if var.startswith('.'):
                continue
            var_dir = data_root + 'Climate_180/{}/per_pix_anomaly/'.format(var)
            var_dic = T.load_npy_dir(var_dir)
            picked_list = []
            for i,row in tqdm(df.iterrows(),total=len(df)):
                pix = row.pix
                if not pix in var_dic:
                    picked_list.append(np.nan)
                    continue
                val = var_dic[pix]
                drought_start = row['drought_event_date_range'][0]
                picked_index = []
                for ni in range(1,n+1):
                    indx = drought_start - ni
                    if indx < 0:
                        picked_index = []
                        break
                    picked_index.append(indx)
                # if len(picked_index) == 0:
                #     print(drought_start)
                picked_index = picked_index[::-1]
                val_picked = T.pick_vals_from_1darray(val,picked_index)
                picked_mean = np.nanmean(val_picked)
                picked_list.append(picked_mean)
            df['{}_previous_{}'.format(var,n)] = picked_list
        return df

        pass


    def add_pre_drought_TWS(self,df,n):
        var_dir = data_root + 'TWS/GRACE/per_pix_clean/'
        var_dic = T.load_npy_dir(var_dir)
        picked_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if not pix in var_dic:
                picked_list.append(np.nan)
                continue
            val = var_dic[pix]
            drought_start = row['drought_event_date_range'][0]
            picked_index = []
            for ni in range(1, n + 1):
                indx = drought_start - ni
                if indx < 0:
                    picked_index = []
                    break
                picked_index.append(indx)
            # if len(picked_index) == 0:
            #     print(drought_start)
            picked_index = picked_index[::-1]
            val_picked = T.pick_vals_from_1darray(val, picked_index)
            picked_mean = np.nanmean(val_picked)
            picked_list.append(picked_mean)
        df['{}_previous_{}'.format('TWS', n)] = picked_list

        return df

    def add_pre_drought_SM(self,df,n):
        var_dir = data_root + 'SM/per_pix_clean_anomaly_180/'
        var_dic = T.load_npy_dir(var_dir)
        picked_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if not pix in var_dic:
                picked_list.append(np.nan)
                continue
            val = var_dic[pix]
            drought_start = row['drought_event_date_range'][0]
            picked_index = []
            for ni in range(1, n + 1):
                indx = drought_start - ni
                if indx < 0:
                    picked_index = []
                    break
                picked_index.append(indx)
            # if len(picked_index) == 0:
            #     print(drought_start)
            picked_index = picked_index[::-1]
            val_picked = T.pick_vals_from_1darray(val, picked_index)
            picked_mean = np.nanmean(val_picked)
            picked_list.append(picked_mean)
        df['{}_previous_{}'.format('SM', n)] = picked_list

        return df


class Analysis:

    def __init__(self):
        self.this_class_arr = results_root_main_flow_2002 + 'arr/Analysis/'
        self.this_class_png = results_root_main_flow_2002 + 'png/Analysis/'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass


    def run(self):

        # self.Isohyd()
        # self.Isohyd_corr()
        # self.Isohyd_corr_events()
        # self.Isohyd_corr_spatial()
        # self.Isohyd_corr_spatial_scatter()
        # self.GRACE()
        # self.plot_all_variables()
        # self.plot_previous_png()
        self.Koppen_and_isohydricity()
        pass

    def load_df(self):
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        self.df = T.load_df(dff)
        df = self.df
        return df,dff

        pass

    def __divide_MA(self,arr,min_v=None,max_v=None,step=None,n=None,round_=2,include_external=False):
        if min_v == None:
            min_v = np.min(arr)
        if max_v == None:
            max_v = np.max(arr)
        if n == None and step == None:
            raise UserWarning('step or n is required')
        if n == None:
            d = np.arange(start=min_v,step=step,stop=max_v)
            if include_external:
                print(d)
                print('n=None')
                exit()
        elif step == None:
            d = np.linspace(min_v,max_v,num=n)
            if include_external:
                d = np.insert(d,0,np.min(arr))
                d = np.append(d,np.max(arr))
                # print(d)
                # exit()
        else:
            d = np.nan
            raise UserWarning('n and step cannot exist together')
        d_str = []
        for i in d:
            d_str.append('{}'.format(round(i, round_)))
        # print d_str
        # exit()
        return d,d_str
        pass

    def Isohyd(self):
        # n = 3
        # var = 'legacy_{}'.format(n)
        #
        var = 'carbon_loss'
        # var = 'greenness_loss'
        df,dff = self.load_df()
        df,_ = Global_vars().clean_df(df)
        isohydricity_series = df.isohydricity
        Iso_d,Iso_d_str = self.__divide_MA(isohydricity_series,min_v=-0.5,max_v=1.5,n=10)

        spatial_dic = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = 1
            spatial_dic[pix] = val

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        Global_vars().mask_arr_with_NDVI(arr)
        plt.subplot(211)
        plt.imshow(arr,resample=False,interpolation='nearest')
        DIC_and_TIF().plot_back_ground_arr()
        plt.title(var)
        plt.subplot(212)
        box = []
        for i in tqdm(range(len(Iso_d))):
            if i+1 >= len(Iso_d):
                continue
            df_temp = df[df.isohydricity>Iso_d[i]]
            df_temp = df_temp[df_temp.isohydricity<Iso_d[i+1]]
            # vals = df_temp['legacy_3']
            vals = df_temp[var]
            vals = T.remove_np_nan(vals)
            box.append(vals)
        plt.boxplot(box,showfliers=False)
        plt.xticks(range(len(Iso_d)),Iso_d_str)
        plt.show()


    def Isohyd_corr_events(self):
        # var1 = 'greenness_loss'
        var1 = 'carbon_loss'
        # var1 = 'legacy_1'
        # var1 = 'legacy_2'
        # var1 = 'legacy_3'


        var2 = 'TWS_recovery_period'
        # var2 = 'NDVI_pre_24'
        # var2 = 'CSIF_pre_3'
        # var2 = 'PRE_previous_24'
        # var2 = 'VPD_previous_3'
        # var2 = 'TMP_previous_24'


        df,dff = self.load_df()
        df,valid_pix_dic = Global_vars().clean_df(df)
        isohydricity_series = df['isohydricity']
        Iso_d, Iso_d_str = self.__divide_MA(isohydricity_series, min_v=0, max_v=1., n=11,include_external=False)

        spatial_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            val = 1
            spatial_dic[pix] = val
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        Global_vars().mask_arr_with_NDVI(arr)
        plt.figure(figsize=(10,7))
        plt.subplot(311)
        plt.imshow(arr, resample=False, interpolation='nearest')
        DIC_and_TIF().plot_back_ground_arr_north_sphere()
        plt.xticks([],[])
        plt.yticks([],[])
        plt.subplot(312)
        x = []
        p_list = []
        class_number = []
        for i in tqdm(range(len(Iso_d))):
            if i + 1 >= len(Iso_d):
                continue
            df_temp = df[df.isohydricity > Iso_d[i]]
            df_temp = df_temp[df_temp.isohydricity < Iso_d[i + 1]]
            number = len(df_temp)
            # vals = df_temp['legacy_3']
            vals1 = df_temp[var1]
            vals2 = df_temp[var2]

            vals1 = np.array(vals1)
            vals2 = np.array(vals2)
            vals1 = -vals1
            vals1_new = []
            vals2_new = []
            for ii in range(len(vals1)):
                if np.isnan(vals1[ii]):
                    continue
                if np.isnan(vals2[ii]):
                    continue
                vals1_new.append(vals1[ii])
                vals2_new.append(vals2[ii])
            try:
                r,p = stats.pearsonr(vals1_new,vals2_new)
            except:
                r = np.nan
                p = np.nan
            x.append(r)
            if p < 0.05:
                p_list.append('r')
            else:
                p_list.append('b')
            class_number.append(number)
        plt.bar(range(len(x)),x,color=p_list,align='edge')
        plt.xticks(range(len(Iso_d)), Iso_d_str,rotation=90)
        plt.xlabel('IsoHydricity')
        plt.ylabel('correlation')
        plt.title('{} vs {}'.format(var1,var2))
        plt.subplot(313)
        plt.bar(range(len(x)), class_number, align='edge')
        plt.xticks(range(len(Iso_d)), Iso_d_str, rotation=90)
        plt.ylabel('events number')
        plt.tight_layout()
        plt.show()

        pass


    def kernel_Isohyd_corr_spatial(self,params):
        var1,var2_list,outdir = params
        flag = 0
        plt.figure(figsize=(20,13))

        for var2 in var2_list:
            flag += 1
            print(var1, var2)
            vmin = 0
            vmax = 1
            df, dff = self.load_df()
            df, valid_pix_dic = Global_vars().clean_df(df)
            isohydricity_series = df['isohydricity']
            Iso_d, Iso_d_str = self.__divide_MA(isohydricity_series, min_v=vmin, max_v=vmax, n=11,
                                                include_external=False)
            Iso_spatial_dic = {}
            for i, row in df.iterrows():
                pix = row.pix
                val = row['isohydricity']
                Iso_spatial_dic[pix] = val
            Iso_arr = DIC_and_TIF().pix_dic_to_spatial_arr(Iso_spatial_dic)

            selected_pix_dic = {}

            for i in tqdm(range(len(Iso_d))):
                if i + 1 >= len(Iso_d):
                    continue
                df_temp = df[df.isohydricity > Iso_d[i]]
                df_temp = df_temp[df_temp.isohydricity < Iso_d[i + 1]]
                pix_series = df_temp.pix
                pix_series = np.array(pix_series)
                pix_series = set(pix_series)
                selected_pix_dic[i] = tuple(pix_series)

            var1_dic = DIC_and_TIF().void_spatial_dic()
            var2_dic = DIC_and_TIF().void_spatial_dic()

            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row.pix
                val1 = row[var1]
                val2 = row[var2]
                var1_dic[pix].append(val1)
                var2_dic[pix].append(val2)

            arr1 = DIC_and_TIF().pix_dic_to_spatial_arr_mean(var1_dic)
            arr2 = DIC_and_TIF().pix_dic_to_spatial_arr_mean(var2_dic)

            corr_list = []
            p_list = []
            pix_number = []
            for i in range(len(Iso_d)):
                if i + 1 >= len(Iso_d):
                    continue
                pix_selected = selected_pix_dic[i]
                picked_val1 = T.pick_vals_from_2darray(arr1, pix_selected, pick_nan=True)
                picked_val2 = T.pick_vals_from_2darray(arr2, pix_selected, pick_nan=True)
                df_temp_ = pd.DataFrame()
                df_temp_['var1'] = picked_val1
                df_temp_['var2'] = picked_val2
                df_temp_ = df_temp_.dropna()
                picked_val1 = df_temp_['var1']
                picked_val2 = df_temp_['var2']
                picked_val1 = -picked_val1
                try:
                    corr = stats.pearsonr(picked_val1, picked_val2)
                    r, p = corr
                except:
                    r, p = [np.nan] * 2
                corr_list.append(r)
                if p < 0.05:
                    p_list.append('r')
                else:
                    p_list.append('b')
                pix_number.append(len(picked_val2))

            # plt.subplot(311)
            # plt.imshow(Iso_arr[:180],vmin=vmin,vmax=vmax)
            # plt.colorbar()
            # DIC_and_TIF().plot_back_ground_arr_north_sphere()
            # plt.xticks([],[])
            # plt.yticks([],[])
            # plt.grid()
            # plt.title('IsoHydricity')

            plt.subplot(5, 7, flag)
            plt.bar(range(len(corr_list)), corr_list, color=p_list, align='edge')
            plt.xticks(range(len(Iso_d)), Iso_d_str, rotation=90)
            # plt.grid()
            # plt.xlabel('IsoHydricity')
            # plt.ylabel('correlation')
            plt.title('{}'.format(var2))

            # plt.subplot(313)
            # plt.bar(range(len(corr_list)),pix_number,align='edge')
            # plt.xticks(range(len(Iso_d)), Iso_d_str, rotation=90)
            # plt.ylabel('pixel number')

        plt.tight_layout()
        outf = outdir + '{}.pdf'.format(var1)
        plt.savefig(outf, dpi=300)
        plt.close()
        # plt.show()

        pass

    def Isohyd_corr_spatial(self):
        outdir = self.this_class_png + 'Isohyd_corr_spatial/'
        T.mk_dir(outdir)
        var2_list,var1_list = self.__gen_variables()
        # print(len(var2_list))
        params = []
        for var1 in var1_list:
            params.append([var1,var2_list,outdir])

        MULTIPROCESS(self.kernel_Isohyd_corr_spatial,params).run()


        pass
    def Isohyd_corr_spatial_scatter(self):
        # var1 = 'greenness_loss'
        var1 = 'carbon_loss'
        # var1 = 'legacy_1'
        # var1 = 'legacy_2'
        # var1 = 'legacy_3'


        # var2 = 'TWS_recovery_period'
        # var2 = 'NDVI_pre_6'
        # var2 = 'CSIF_pre_3'
        var2 = 'PRE_previous_3'
        # var2 = 'VPD_previous_6'
        # var2 = 'TMP_previous_3'

        vmin = 0.
        vmax = 0.9
        df,dff = self.load_df()
        df,valid_pix_dic = Global_vars().clean_df(df)
        isohydricity_series = df['isohydricity']
        Iso_d, Iso_d_str = self.__divide_MA(isohydricity_series, min_v=vmin, max_v=vmax, n=4,include_external=False)
        Iso_spatial_dic = {}
        for i,row in df.iterrows():
            pix = row.pix
            val = row['isohydricity']
            Iso_spatial_dic[pix] = val
        Iso_arr = DIC_and_TIF().pix_dic_to_spatial_arr(Iso_spatial_dic)

        selected_pix_dic = {}

        for i in tqdm(range(len(Iso_d))):
            if i + 1 >= len(Iso_d):
                continue
            df_temp = df[df.isohydricity > Iso_d[i]]
            df_temp = df_temp[df_temp.isohydricity < Iso_d[i + 1]]
            pix_series = df_temp.pix
            pix_series = np.array(pix_series)
            pix_series = set(pix_series)
            selected_pix_dic[i] = tuple(pix_series)


        var1_dic = DIC_and_TIF().void_spatial_dic()
        var2_dic = DIC_and_TIF().void_spatial_dic()

        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val1 = row[var1]
            val2 = row[var2]
            var1_dic[pix].append(val1)
            var2_dic[pix].append(val2)

        arr1 = DIC_and_TIF().pix_dic_to_spatial_arr_mean(var1_dic)
        arr2 = DIC_and_TIF().pix_dic_to_spatial_arr_mean(var2_dic)

        corr_list = []
        p_list = []
        pix_number = []
        flag = 0
        for i in range(len(Iso_d)):
            if i + 1 >= len(Iso_d):
                continue
            flag += 1
            pix_selected = selected_pix_dic[i]
            picked_val1 = T.pick_vals_from_2darray(arr1,pix_selected,pick_nan=True)
            picked_val2 = T.pick_vals_from_2darray(arr2,pix_selected,pick_nan=True)
            df_temp_ = pd.DataFrame()
            df_temp_['var1'] = picked_val1
            df_temp_['var2'] = picked_val2
            df_temp_ = df_temp_.dropna()
            picked_val1 = df_temp_['var1']
            picked_val2 = df_temp_['var2']
            picked_val1 = -picked_val1
            ax = plt.subplot(3,3,flag)
            KDE_plot().plot_scatter(picked_val1,picked_val2,ax=ax,plot_fit_line=True)
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            corr = stats.pearsonr(picked_val1, picked_val2)
            r, p = corr
            ax.set_title('isohyd {} r:{:0.3f} p:{:0.3f}'.format(flag,r,p))

            corr_list.append(r)
            if p < 0.05:
                p_list.append('r')
            else:
                p_list.append('b')
            pix_number.append(len(picked_val2))
        plt.tight_layout()
        plt.show()

        pass

    def plot_back_ground_arr(self):
        tif_template = DIC_and_TIF().tif_template
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        back_ground = []
        for i in range(len(arr)):
            temp = []
            for j in range(len(arr[0])):
                val = arr[i][j]
                if val < -90000:
                    temp.append(100)
                else:
                    temp.append(70)
            back_ground.append(temp)
        back_ground = np.array(back_ground)
        # plt.imshow(back_ground, 'gray', vmin=0, vmax=1.4,zorder=-1)

        return back_ground

        pass


    def GRACE(self):
        df,dff = self.load_df()
        df,_ = Global_vars().clean_df(df)

        # TWS_recovery_period = df.TWS_recovery_period
        x_variable = df.CSIF_pre_3
        '''
        carbon_loss	legacy_1	legacy_2	legacy_3
        '''
        Y = df['carbon_loss']
        Y = np.array(Y)
        x_variable = np.array(x_variable)
        Y = -Y
        KDE_plot().plot_scatter(x_variable,Y,plot_fit_line=True)

        # plt.scatter(TWS_recovery_period,Y)
        plt.show()





    def climate_analysis(self):
        df, dff = self.load_df()

        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            pass

        pass

    def plot_all_variables(self):
        outdir = self.this_class_png + 'plot_all_variables/'
        T.mk_dir(outdir)
        n = 3
        outf = outdir + 'plot_all_variables_pre_{}.png'.format(n)
        df, dff = self.load_df()
        T.print_head_n(df)
        X,Y = Global_vars().variables(n=n)
        df,_ = Global_vars().clean_df(df)
        df_new = pd.DataFrame()
        for x in X:
            df_new[x] = df[x]
        df_new[Y] = df[Y]
        df_new['lc'] = df['lc']
        corr_arr = df_new.corr()
        corr_arr[corr_arr==1]=np.nan
        # plt.imshow(corr_arr)
        plt.figure(figsize=(20,20))
        # sns.pairplot(df_new,markers='.',kind='reg',diag_kind='kde')
        # sns.pairplot(df_new,kind="hist",hue='lc',hue_order=['Grasslands','Forest','Shrublands',])
        sns.pairplot(df_new,kind="hist",hue='lc',hue_order=['Grasslands',])
        plt.savefig(outf,dpi=300)

    def __gen_lc_pix(self):

        df,dff = self.load_df()
        lc_list = Global_vars().landuse_list()
        lc_dic = {}
        for lc in lc_list:
            lc_dic[lc] = []
        for i,row in df.iterrows():
            pix = row.pix
            lc = row.lc
            if not lc:
                continue
            lc_dic[lc].append(pix)

        return lc_dic


        pass


    def kernel_plot_previous_png(self,params):
        var1_list,var2 = params

        for var1 in var1_list:
            back_ground_var = 'isohydricity'
            # back_ground_var = 'waterbalance'
            outpngdir = self.this_class_png + 'plot_previous_png_{}/'.format(back_ground_var)
            T.mk_dir(outpngdir)

            # print(var1,var2)
            # lc_dic = ''
            lc_dic = self.__gen_lc_pix()
            outf = outpngdir + '{}_{}.png'.format(var2, var1)
            # if os.path.isfile(outf):
            #     return None


            df, dff = self.load_df()
            df, valid_pix_dic = Global_vars().clean_df(df)
            # isohydricity_series = df['isohydricity']
            isohydricity_series = df[back_ground_var]
            isohydricity_series = isohydricity_series.dropna()
            isohydricity_series = np.array(isohydricity_series)
            # plt.hist(isohydricity_series,bins=100)
            # plt.show()
            # Iso_d = [-100,-25,25,100]
            Iso_d = [-0.5,0.3,0.6,1]
            Iso_d_str = [str(iso) for iso in Iso_d]
            vmin = -100
            vmax = 100
            # print(Iso_d)
            # print(Iso_d_str)
            # vmin = -120
            # vmax = 120
            # Iso_d, Iso_d_str = self.__divide_MA(isohydricity_series, min_v=vmin, max_v=vmax, n=4,
            #                                     include_external=False)

            # print(Iso_d)
            # print(Iso_d_str)
            # exit()
            # print(Iso_d)
            # exit()
            # Iso_spatial_dic = {}
            # for i, row in df.iterrows():
            #     pix = row.pix
            #     val = row['isohydricity']
            #     Iso_spatial_dic[pix] = val
            # Iso_arr = DIC_and_TIF().pix_dic_to_spatial_arr(Iso_spatial_dic)

            selected_pix_dic = {}

            for i in tqdm(range(len(Iso_d))):
                if i + 1 >= len(Iso_d):
                    continue
                df_temp = df[df[back_ground_var] > Iso_d[i]]
                df_temp = df_temp[df_temp[back_ground_var] < Iso_d[i + 1]]
                pix_series = df_temp.pix
                pix_series = np.array(pix_series)
                pix_series = set(pix_series)
                selected_pix_dic[i] = tuple(pix_series)

            var1_dic = DIC_and_TIF().void_spatial_dic()
            var2_dic = DIC_and_TIF().void_spatial_dic()

            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row.pix
                val1 = row[var1]
                val2 = row[var2]
                var1_dic[pix].append(val1)
                var2_dic[pix].append(val2)

            arr1 = DIC_and_TIF().pix_dic_to_spatial_arr_mean(var1_dic)
            arr2 = DIC_and_TIF().pix_dic_to_spatial_arr_mean(var2_dic)

            corr_list = []
            p_list = []
            pix_number = []
            flag = 0
            plt.figure(figsize=(15, 5))
            for i in range(len(Iso_d)):
                if i + 1 >= len(Iso_d):
                    continue
                flag += 1
                pix_selected = selected_pix_dic[i]
                for lc in lc_dic:
                    lc_pix = lc_dic[lc]
                    lc_pix = tuple(set(lc_pix))
                    # intersect_pix = lc_pix & pix_selected
                    intersect_pix = list(set(lc_pix) & set(pix_selected))
                    picked_val1 = T.pick_vals_from_2darray(arr1, intersect_pix, pick_nan=True)
                    picked_val2 = T.pick_vals_from_2darray(arr2, intersect_pix, pick_nan=True)
                    df_temp_ = pd.DataFrame()
                    df_temp_['var1'] = picked_val1
                    df_temp_['var2'] = picked_val2
                    df_temp_ = df_temp_.dropna()
                    if 'TWS' in var1:
                        df_temp_ = df_temp_[df_temp_['var1'] > -20]
                        df_temp_ = df_temp_[df_temp_['var1'] < 20]
                    picked_val1 = np.array(df_temp_['var1'])
                    picked_val2 = np.array(df_temp_['var2'])
                    picked_val2 = -picked_val2
                    ax = plt.subplot(1, len(Iso_d)-1, flag)
                    color = Global_vars().color_dic_lc()
                    ax.scatter(picked_val1, picked_val2,
                               s=40, marker='o',
                               facecolors='none',
                               edgecolors=color[lc],
                               alpha=0.6,
                               label=lc)

                    try:
                        a, b, r = KDE_plot().linefit(picked_val1, picked_val2)
                    except Exception as e:
                        a,b,r = [np.nan]*3
                        print(e)
                        print(picked_val1)
                        print(picked_val2)
                        plt.scatter(picked_val1, picked_val2)
                        plt.show()
                        exit()
                    KDE_plot().plot_fit_line(a, b, r, picked_val1, ax=ax, c=color[lc], linewidth=2,
                                             is_label=True,is_formula=False)
                    ax.set_xlabel(var1)
                    ax.set_ylabel(var2)
                    # corr = stats.pearsonr(picked_val1, picked_val2)
                    # r, p = corr
                    if Iso_d[i] < 0:
                        ax.set_title('{} less than {:0.2f}'.format(back_ground_var,Iso_d[i + 1]))
                    else:
                        ax.set_title('{} {:0.2f} to {:0.2f}'.format(back_ground_var,Iso_d[i], Iso_d[i + 1]))
                    pix_number.append(len(picked_val2))
                plt.legend()
            plt.tight_layout()
            plt.savefig(outf, dpi=300)
            plt.close()
            # exit()
            # plt.show()

        pass


    def plot_previous_png(self):
        var1_list, y_list = self.__gen_variables()

        # print(var1_list)
        # exit()

        params = []
        for var2 in y_list:

            # params.append([var1_list,var2])
            self.kernel_plot_previous_png([var1_list,var2])
        # MULTIPROCESS(self.kernel_plot_previous_png,params).run()


    def __gen_variables(self):

        n_list = [3, 6, 12, 24]
        climate_list_1 = [
            'PRE',
            'VPD',
            'TMP',
            'TWS',
            'SM',
        ]

        veg_list = [
            'CSIF',
            'NDVI',
        ]

        y_list = [
            'carbon_loss',
            # 'greenness_loss',
            # 'legacy_1',
            # 'legacy_2',
            # 'legacy_3',
        ]

        # var_n_list = []
        var1_list = []
        for n in n_list:
            for c in climate_list_1:
                var1 = '{}_previous_{}'.format(c, n)
                var1_list.append(var1)
            for v in veg_list:
                var2 = '{}_pre_{}'.format(v, n)
                var1_list.append(var2)
            # var_n_list.append(var1_list)
        var1_list.append('TWS_recovery_period')
        return var1_list,y_list



    def kernel_Koppen_and_isohydricity(self,var_name):
        outdir = self.this_class_png + 'Koppen_and_isohydricity/'
        T.mk_dir(outdir)
        df,dff = self.load_df()
        df,_ = Global_vars().clean_df(df)
        kp_list = Global_vars().koppen_list()
        iso_hyd_divide = [-999,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,999]
        iso_hyd_divide_range = []
        for i,v in enumerate(iso_hyd_divide):
            if i + 1 == len(iso_hyd_divide):
                break
            range_ = [iso_hyd_divide[i],iso_hyd_divide[i+1]]
            iso_hyd_divide_range.append(range_)
        outf = outdir + '{}.png'.format(var_name)
        flag = 0
        plt.figure(figsize=(20,20))
        for kp in kp_list:
            df_kp = df[df['kp']==kp]
            for min_iso,max_iso in iso_hyd_divide_range:
                df_iso = df_kp[df_kp['isohydricity']>min_iso]
                df_iso = df_iso[df_iso['isohydricity']<max_iso]
                df_temp = pd.DataFrame()
                df_temp['carbon_loss'] = df_iso['carbon_loss']
                df_temp[var_name] = df_iso[var_name]
                df_temp = df_temp.dropna()
                flag += 1
                ax = plt.subplot(len(kp_list),len(iso_hyd_divide_range),flag)
                x,y = df_temp[var_name],df_temp['carbon_loss']
                y = -y
                title = '{} {} {}-{}'.format(var_name,kp,min_iso,max_iso)
                try:
                    KDE_plot().plot_scatter(x,y,plot_fit_line=True,ax=ax,title=title,silent=True)
                    # KDE_plot().plot_scatter(x,y,plot_fit_line=True,ax=ax,title=title,silent=False)
                    if not 'TWS' in var_name:
                        ax.set_ylim(0,20)
                        ax.set_xlim(-2,2)
                except:
                    ax.plot([],[])
                    ax.set_title(title)
                if flag % len(iso_hyd_divide_range) != 1:
                    ax.set_yticks([])
                if not 'TWS' in var_name:
                    if flag <= len(iso_hyd_divide_range) * (len(kp_list)-1):
                        ax.set_xticks([])
        plt.tight_layout()
        plt.savefig(outf,dpi=144)
        plt.close()


    def Koppen_and_isohydricity(self):
        params = []
        for n in [3,6,12,24]:
            xlist,y = Global_vars().variables(n)
            for x in xlist:
                params.append(x)
        MULTIPROCESS(self.kernel_Koppen_and_isohydricity,params).run()
        # print(params)
        # exit()
        pass


class Tif:


    def __init__(self):
        self.this_class_tif = results_root_main_flow + 'tif/Tif/'
        Tools().mk_dir(self.this_class_tif, force=True)

    def run(self):

        # self.carbon_loss()
        self.drought_start()
        pass


    def load_df(self):
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)

        return df,dff

        pass

    def carbon_loss(self):
        df ,dff = self.load_df()
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        outdir = self.this_class_tif + 'carbon_loss/'
        T.mk_dir(outdir)
        outf = outdir + 'carbon_loss.tif'
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix

            val = row['carbon_loss']
            spatial_dic[pix].append(val)

        mean_arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(spatial_dic)
        DIC_and_TIF().arr_to_tif(mean_arr,outf)


        pass

    def drought_start(self):

        df,dff = self.load_df()

        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for i,row in tqdm(df.iterrows(),total=len(df)):

            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            start = drought_event_date_range[-1]
            spatial_dic[pix].append(start)

        arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(spatial_dic)
        plt.imshow(arr)
        plt.colorbar()
        plt.show()

class Main_flow_RF:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Main_flow_RF\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Main_flow_RF\\'
        self.this_class_png = results_root_main_flow + 'png\\Main_flow_RF\\'
        T.mk_dir(self.this_class_arr, force=True)
        T.mk_dir(self.this_class_tif, force=True)
        T.mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        pass

    def permutation_train_results(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.2)
        # rf = RandomForestRegressor(n_estimators=300)
        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = rf.score(X_test, y_test)
        result = permutation_importance(rf, X_train, y_train, scoring=None,
                                        n_repeats=10, random_state=42,
                                        n_jobs=1)
        importances = result.importances_mean
        importances_dic = dict(zip(X.columns, importances))
        return importances_dic, r2

    def importance_train_results(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.2)
        # rf = RandomForestRegressor(n_estimators=300)
        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = rf.score(X_test, y_test)
        importances = rf.feature_importances_
        importances_dic = dict(zip(X.columns, importances))
        return importances_dic, r2

    def discard_hierarchical_clustering(self, df, var_list, dest_Y, t=0.0, isplot=False):
        '''
        url:
        https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
        '''
        from collections import defaultdict

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import spearmanr
        from scipy.cluster import hierarchy

        # print(df)
        # exit()
        df = df.dropna()
        var_list_copy = copy.copy(var_list)
        if dest_Y in var_list_copy:
            var_list_copy.remove(dest_Y)
        var_list = var_list_copy
        X = df[var_list]
        corr = np.array(X.corr())
        corr_linkage = hierarchy.ward(corr)

        cluster_ids = hierarchy.fcluster(corr_linkage, t=t, criterion='distance')
        # cluster_ids = hierarchy.fcluster(corr_linkage, t=t, criterion='inconsistent')
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features_indx = [v[0] for v in cluster_id_to_feature_ids.values()]
        selected_features = []
        for i in selected_features_indx:
            selected_features.append(var_list[i])

        if isplot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            dendro = hierarchy.dendrogram(
                corr_linkage, labels=var_list, ax=ax1, leaf_rotation=90
            )
            dendro_idx = np.arange(0, len(dendro['ivl']))
            ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
            ax2.set_xticks(dendro_idx)
            ax2.set_yticks(dendro_idx)
            ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
            ax2.set_yticklabels(dendro['ivl'])
            fig.tight_layout()
        return selected_features

    def discard_vif_vars(self, df, vars_list, dest_Y):
        ##################实时计算#####################
        vars_list_copy = copy.copy(vars_list)
        if dest_Y in vars_list_copy:
            vars_list_copy.remove(dest_Y)
        X = df[vars_list_copy]
        X = X.dropna()
        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif.round(1)
        selected_vif_list = []
        for i in range(len(vif)):
            feature = vif['features'][i]
            VIF_val = vif['VIF Factor'][i]
            if VIF_val < 5.:
                selected_vif_list.append(feature)
        return selected_vif_list

        pass

    def permutation_RF_no_winter(self, dest_Y):
        var_dic = self.gen_variables_no_winter()
        for pre_months in var_dic:
            print(pre_months)
            dest_var = dest_Y
            outdir = self.this_class_arr + 'permutation_RF_no_winter\\'
            outf = outdir + '{}'.format(pre_months)
            mk_dir(outdir, force=True)
            arr_dir = results_root + 'arr\\Main_flow_Prepare\\'
            df_f = arr_dir + 'prepare\\data_frame_threshold_{}.df'.format(0)
            df = pd.read_pickle(df_f)
            ################################
            ################################
            ################################
            df = df[df['winter_mark_new'] == 0]
            ################################
            ################################
            ################################

            print(df.columns)
            kl_list = list(set(list(df['climate_zone'])))
            kl_list.remove(None)
            kl_list.sort()
            results_dic = {}
            for kl in kl_list:
                print(kl)
                vars_list = var_dic[pre_months]

                results_key = kl
                df_kl = df[df['climate_zone'] == kl]
                df_kl = df_kl.replace([np.inf, -np.inf], np.nan)
                vars_list.append(dest_Y)
                XXX = df_kl[vars_list]
                if len(XXX) < 100:
                    continue

                vif_selected_features = self.discard_vif_vars(XXX, vars_list)
                selected_features = self.discard_hierarchical_clustering(XXX, vif_selected_features, t=1,
                                                                         isplot=False)
                # print(vif_discard_vars)
                # exit()
                selected_features.append('water_balance')
                selected_features.append('timing_int')
                selected_features.append('lag')
                selected_features.append('NDVI_pre_{}_mean'.format(pre_months))
                print(selected_features)
                df1 = pd.DataFrame(df_kl)
                vars_list1 = list(set(selected_features))
                vars_list1.sort()
                vars_list1.append(dest_Y)
                XX = df1[vars_list1]
                XX = XX.dropna()
                vars_list1.remove(dest_Y)
                X = XX[vars_list1]
                Y = XX[dest_Y]
                if len(df1) < 100:
                    # if len(X) == 0:
                    continue
                # X = X.dropna()
                importances_dic, r2 = self.train_results(X, Y, X.columns)
                results_dic[results_key] = (importances_dic, r2)
                labels = []
                importance = []
                for key in importances_dic:
                    labels.append(key)
                    importance.append(importances_dic[key])
            fw = outf + '.txt'
            fw = open(fw, 'w')
            fw.write(str(results_dic))

        pass

    def permutation_RF_with_winter(self, dest_Y):
        var_dic = self.gen_variables_with_winter()
        for pre_months in var_dic:
            print(pre_months)
            dest_var = dest_Y
            outdir = self.this_class_arr + 'permutation_RF_with_winter\\'
            outf = outdir + '{}'.format(pre_months)
            mk_dir(outdir, force=True)
            arr_dir = results_root + 'arr\\Main_flow_Prepare\\'
            df_f = arr_dir + 'prepare\\data_frame_threshold_{}.df'.format(0)
            df = pd.read_pickle(df_f)
            ################################
            ################################
            ################################
            df = df[df['winter_mark_new'] == 1]
            # df = df[df['winter_mark'] == 0]
            # df = df[df['winter_mark'] == 1]
            # df = df[df['recovery_start_gs'] == 'first']
            ################################
            ################################
            ################################

            print(df.columns)
            kl_list = list(set(list(df['climate_zone'])))
            kl_list.remove(None)
            kl_list.sort()
            results_dic = {}
            no_swe_koppen = ['A', 'B', 'Cf', 'Csw']

            for kl in kl_list:
                print(kl)
                koppen = kl.split('.')[1]
                if 'A' in kl:
                    continue
                # if not kl == 'Grasslands.Cf':
                #     continue
                vars_list = var_dic[pre_months]
                vars_list = copy.copy(vars_list)
                print(vars_list)
                if koppen in no_swe_koppen:
                    if 'dormant_SWE' in vars_list:
                        vars_list.remove('dormant_SWE')
                results_key = kl
                df_kl = df[df['climate_zone'] == kl]
                df_kl = df_kl.replace([np.inf, -np.inf], np.nan)
                vars_list.append(dest_Y)

                XXX = df_kl[vars_list]
                if len(XXX) < 100:
                    continue
                XXX = XXX.dropna(axis=1, how="all")
                # XXX.to_excel('XXX.xlsx')
                # exit()
                vif_selected_features = self.discard_vif_vars(XXX, vars_list)
                selected_features = self.discard_hierarchical_clustering(XXX, vif_selected_features, t=1,
                                                                         isplot=False)
                # print(vif_discard_vars)
                # exit()
                selected_features.append('water_balance')
                selected_features.append('timing_int')
                selected_features.append('dormant_length')
                selected_features.append('NDVI_pre_{}_mean'.format(pre_months))
                print(selected_features)
                df1 = pd.DataFrame(df_kl)
                vars_list1 = list(set(selected_features))
                vars_list1.sort()
                vars_list1.append(dest_Y)
                XX = df1[vars_list1]
                XX = XX.dropna()
                vars_list1.remove(dest_Y)
                X = XX[vars_list1]
                Y = XX[dest_Y]
                if len(df1) < 100:
                    continue
                # print(X)
                # X.to_excel('x.xlsx')
                # exit()
                importances_dic, r2 = self.train_results(X, Y, X.columns)
                results_dic[results_key] = (importances_dic, r2)
                labels = []
                importance = []
                for key in importances_dic:
                    labels.append(key)
                    importance.append(importances_dic[key])
            fw = outf + '.txt'
            fw = open(fw, 'w')
            fw.write(str(results_dic))

        pass

        pass



def main():
    Main_flow_Early_Peak_Late_Dormant().run()

    # Main_Flow_Pick_drought_events().run()
    # Main_flow_Legacy().run()
    # Main_flow_Carbon_loss().run()
    # Main_flow_Greenness_loss().run()
    # Main_flow_Dataframe_NDVI_SPEI_legacy().run()
    # Analysis().run()
    # Tif().run()
    # Main_flow_RF().run()



    pass



if __name__ == '__main__':
    main()
    pass