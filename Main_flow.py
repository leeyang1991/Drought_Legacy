# coding=gbk

from __init__ import *
from analysis import *


class Global_vars:

    def __init__(self):

        pass

    def koppen_landuse(self):
        kl_list = [u'Forest.A', u'Forest.B', u'Forest.Cf', u'Forest.Csw', u'Forest.Df', u'Forest.Dsw', u'Forest.E',
         u'Grasslands.A', u'Grasslands.B', u'Grasslands.Cf', u'Grasslands.Csw', u'Grasslands.Df', u'Grasslands.Dsw',
         u'Grasslands.E', u'Shrublands.A', u'Shrublands.B', u'Shrublands.Cf', u'Shrublands.Csw', u'Shrublands.Df',
         u'Shrublands.Dsw', u'Shrublands.E']
        return kl_list

    def koppen_list(self):
        koppen_list = [u'A', u'B', u'Cf', u'Csw', u'Df', u'Dsw', u'E',]
        return koppen_list
        pass


    def marker_dic(self):
        markers_dic = {
                       'Shrublands': "o",
                       'Forest': "X",
                       'Grasslands': "p",
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

        gs = list(range(4,10))

        return gs

    def variables(self):
        X = [
            'isohydricity',
            'canopy_height',
            'rooting_depth',
            'PRE_delta',
            'TMP_delta',
            'VPD_delta',
            'SPEI_delta',
            'PRE_trend',
            'TMP_trend',
            'VPD_trend',
            'PRE_cv_delta',
            'TMP_cv_delta',
            'VPD_cv_delta',
            'PRE_cv',
            'TMP_cv',
            'VPD_cv',
            'waterbalance',
            'sand',
            'awc',
             ]
        # Y = 'delta_legacy'
        Y = 'trend'

        return X,Y

        pass

    def clean_df(self,df):
        df = df.drop_duplicates(subset=['pix', 'delta_legacy'])
        # self.__df_to_excel(df,dff+'drop')

        df = df[df['ratio_of_forest'] > 0.90]
        df = df[df['lat'] > 30]
        df = df[df['lat'] < 60]
        # df = df[df['delta_legacy'] < -0]
        df = df[df['trend_score'] > 0.2]
        # df = df[df['gs_sif_spei_corr'] > 0]

        trend = df['trend']
        trend_mean = np.nanmean(trend)
        trend_std = np.nanstd(trend)
        up = trend_mean + trend_std
        down = trend_mean - trend_std
        df = df[df['trend'] > down]
        df = df[df['trend'] < up]

        # quantile = 0.4
        # delta_legacy = df['delta_legacy']
        # delta_legacy = delta_legacy.dropna()
        #
        # # print(delta_legacy)
        # q = np.quantile(delta_legacy,quantile)
        # # print(q)
        # df = df[df['delta_legacy']<q]
        # T.print_head_n(df)
        print(len(df))
        # exit()

        return df
class Main_flow_Early_Peak_Late_Dormant:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Main_flow_Early_Peak_Late_Dormant\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Main_flow_Early_Peak_Late_Dormant\\'
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
        self.tif_bi_weekly_annual()
        # 2 data transform
        self.data_transform_annual()
        # 3 smooth bi-weekly to daily
        # self.hants_smooth_annual()
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
        f = self.this_class_arr + 'early_peak_late_dormant_period_long_term\\early_peak_late_dormant_period_long_term.npy'
        dic = T.load_npy(f)
        return dic

    def return_gs(self):
        f = self.this_class_arr + 'early_peak_late_dormant_period_long_term\\early_peak_late_dormant_period_long_term.npy'
        dic = T.load_npy(f)
        gs_dic = {}
        for pix in dic:
            val = dic[pix]['GS_mon']
            gs_dic[pix] = val
        return gs_dic
        pass


    def check_early_peak_late_dormant_period_long_term(self):
        outtifdir = self.this_class_tif + 'early_peak_late_dormant_period_long_term\\'
        T.mk_dir(outtifdir)
        f = self.this_class_arr + 'early_peak_late_dormant_period_long_term\\early_peak_late_dormant_period_long_term.npy'
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
        fdir = self.this_class_arr + 'transform_early_peak_late_dormant_period_annual\\'
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


        outdir = self.this_class_arr + 'transform_early_peak_late_dormant_period_annual\\'
        T.mk_dir(outdir)
        fdir = self.this_class_arr + 'early_peak_late_dormant_period_annual\\'
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

    def hants_smooth_annual(self):
        outdir = self.this_class_arr + 'hants_smooth_annual\\'
        T.mk_dir(outdir)
        gs_f = Phenology_based_on_Temperature_NDVI().this_class_arr + 'growing_season_index.npy'
        gs_dic = T.load_npy(gs_f)
        per_pix_dir = self.this_class_arr + 'data_transform_annual\\'
        for year in os.listdir(per_pix_dir):
            # print year
            per_pix_dir_i = per_pix_dir + year + '\\'
            outf = outdir + year
            ndvi_dic = {}
            for f in os.listdir(per_pix_dir_i):
                dic = T.load_npy(per_pix_dir_i+f)
                ndvi_dic.update(dic)
            hants_dic = {}
            for pix in tqdm(ndvi_dic):
                if not pix in gs_dic:
                    continue
                vals = ndvi_dic[pix]
                smoothed_vals = self.__kernel_hants(vals)
                hants_dic[pix] = np.array(smoothed_vals)
            np.save(outf,hants_dic)


    def data_transform_annual(self):

        fdir = self.this_class_tif + 'tif_bi_weekly_annual\\'
        outdir = self.this_class_arr + 'data_transform_annual\\'
        T.mk_dir(outdir)
        for year in os.listdir(fdir):
            # print year,'\n'
            # for f in os.listdir(fdir + year):
            #     print f
            outdir_i = outdir + year + '\\'
            T.mk_dir(outdir_i)
            Pre_Process().data_transform(fdir + year + '\\', outdir_i)
        pass

    def early_peak_late_dormant_period_annual(self,threshold_i=0.2):
        hants_smooth_dir = self.this_class_arr + 'hants_smooth_annual\\'
        outdir = self.this_class_arr + 'early_peak_late_dormant_period_annual\\'
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
        hants_smooth_f = self.this_class_arr + 'hants_smooth\\hants_smooth.npy'
        outdir = self.this_class_arr + 'early_peak_late_dormant_period_long_term\\'
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
        outdir = self.this_class_arr + 'hants_smooth\\'
        outf = outdir + 'hants_smooth'
        T.mk_dir(outdir)
        gs_f = Phenology_based_on_Temperature_NDVI().this_class_arr + 'growing_season_index.npy'
        gs_dic = T.load_npy(gs_f)
        per_pix_dir = self.this_class_arr + 'NDVI_bi_weekly_per_pix\\'
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


    def tif_bi_weekly_annual(self):
        fdir = data_root + 'NDVI\\tif_05deg_bi_weekly\\'
        outdir = self.this_class_tif + 'tif_bi_weekly_annual\\'
        T.mk_dir(outdir)
        date_list = []
        for y in range(1982, 2016):
            temp = []
            for m in range(1, 13):
                for d in [1, 15]:
                    temp.append('{}{:02d}{:02d}.tif'.format(y,m,d))
            date_list.append(temp)
        for annual in date_list:
            yyyy = annual[0].split('.')[0][:4]
            # print yyyy
            outdir_i = outdir + yyyy + '\\'
            T.mk_dir(outdir_i)
            for mon in annual:
                f = fdir + mon
                shutil.copy(f,outdir_i+mon)


    def tif_bi_weekly_mean_long_term(self):
        fdir = data_root + 'NDVI\\tif_05deg_bi_weekly\\'
        outdir = self.this_class_tif + 'tif_bi_weekly_mean\\'
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
        fdir = self.this_class_tif + 'tif_bi_weekly_mean\\'
        outdir = self.this_class_arr + 'NDVI_bi_weekly_per_pix\\'
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
            return None
        xnew, ynew = self.__interp__(vals)
        ynew = np.array([ynew])
        results = HANTS(sample_count=365, inputs=ynew, low=-10000, high=10000,
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
        self.this_class_arr = results_root_main_flow + 'arr\\SPEI_preprocess\\'
        self.this_class_tif = results_root_main_flow + 'tif\\SPEI_preprocess\\'
        self.this_class_png = results_root_main_flow + 'png\\SPEI_preprocess\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        # self.do_pick()
        pass

    def do_pick(self):
        outdir = self.this_class_arr + 'drought_events\\'
        fdir = data_root + 'SPEI\\per_pix_clean\\'
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


class Main_flow_Recovery_time_Legacy:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Recovery_time_Legacy\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Recovery_time_Legacy\\'
        self.this_class_png = results_root_main_flow + 'png\\Recovery_time_Legacy\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        # 1 cal recovery time
        event_dic,spei_dic,sif_dic,pred_ndvi_dic = self.load_data()
        out_dir = self.this_class_arr + 'Recovery_time_Legacy\\'
        self.gen_recovery_time_legacy(event_dic,spei_dic, sif_dic,pred_ndvi_dic,out_dir)
        pass

    def load_data(self,condition=''):
        # events_dir = results_root_main_flow + 'arr\\SPEI_preprocess\\drought_events\\'
        # SPEI_dir = data_root + 'SPEI\\per_pix_clean\\'
        # SIF_dir = data_root + 'CSIF\\per_pix_anomaly_detrend\\'

        events_dir = results_root_main_flow + 'arr\\SPEI_preprocess\\events_408\\spei\\'
        SPEI_dir = data_root + 'SPEI\\per_pix_408\\'
        SIF_dir = data_root + 'NDVI\\per_pix_clean_anomaly_smooth_detrend\\'
        pred_ndvi_dir = SPEI_NDVI_Reg().this_class_arr + 'pred_NDVI.npy'

        event_dic = T.load_npy_dir(events_dir,condition)
        spei_dic = T.load_npy_dir(SPEI_dir,condition)
        sif_dic = T.load_npy_dir(SIF_dir,condition)
        pred_ndvi_dic = T.load_npy(pred_ndvi_dir)

        return event_dic,spei_dic,sif_dic,pred_ndvi_dic
        pass


    def __cal_legacy(self,ndvi_obs,ndvi_pred,recovery_range):
        selected_obs = T.pick_vals_from_1darray(ndvi_obs,recovery_range)
        selected_pred = T.pick_vals_from_1darray(ndvi_pred,recovery_range)
        diff = selected_obs - selected_pred
        legacy = np.sum(diff)
        return legacy
        pass

    def gen_recovery_time_legacy(self, events, spei_dic, ndvi_dic, pred_ndvi_dic, out_dir):
        '''
        生成全球恢复期
        :param interval: SPEI_{interval}
        :return:
        '''

        # pre_dic = Main_flow_Prepare().load_X_anomaly('PRE')

        growing_date_range = list(range(4,10))
        Tools().mk_dir(out_dir, force=True)
        outf = out_dir + 'recovery_time_legacy_reg'
        # 1 加载事件
        # interval = '%02d' % interval
        # 2 加载NDVI 与 SPEI
        recovery_time_dic = {}
        for pix in tqdm(ndvi_dic):
            if pix in events:
                ndvi = ndvi_dic[pix]
                ndvi = np.array(ndvi)
                if not pix in pred_ndvi_dic:
                    continue
                ndvi_pred,r2 = pred_ndvi_dic[pix]
                # if r2 < 0.1:
                #     continue
                # ndvi_pred = [0] * len(ndvi)
                ndvi_pred = np.array(ndvi_pred)
                if not pix in spei_dic:
                    continue
                spei = spei_dic[pix]
                spei = np.array(spei)
                event = events[pix]
                recovery_time_result = []
                for timing,date_range in event:
                    # print(date_range)
                    event_start_index = T.pick_min_indx_from_1darray(spei, date_range)
                    event_start_index_trans = self.__drought_indx_to_gs_indx(event_start_index,growing_date_range,len(ndvi))
                    if event_start_index_trans == None:
                        continue
                    ndvi_gs = self.__pick_gs_vals(ndvi,growing_date_range)
                    spei_gs = self.__pick_gs_vals(spei,growing_date_range)
                    # ndvi_gs_pred = self.__pick_gs_vals(ndvi_pred,growing_date_range)
                    ndvi_gs_pred = ndvi_pred
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
                    # mark: In Out Tropical
                    search_end_indx = 3 * len(growing_date_range)
                    recovery_range = self.search1(ndvi_gs, ndvi_gs_pred,spei_gs, event_start_index_trans,search_end_indx)
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
                    legacy = self.__cal_legacy(ndvi_gs,ndvi_gs_pred,recovery_range)
                    recovery_time_result.append({
                        'recovery_time': recovery_time,
                        'recovery_date_range': recovery_range,
                        'drought_event_date_range': date_range_new,
                        'legacy': legacy,
                    })
                    #
                    # ################# plot ##################
            #         print('recovery_time',recovery_time)
            #         print('growing_date_range',growing_date_range)
            #         print('recovery_range',recovery_range)
            #         print('legacy',legacy)
            #         recovery_date_range = recovery_range
            #         recovery_ndvi = Tools().pick_vals_from_1darray(ndvi_gs, recovery_date_range)
            #
            #         # pre_picked_vals = Tools().pick_vals_from_1darray(pre, tmp_pre_date_range)
            #         # tmp_picked_vals = Tools().pick_vals_from_1darray(tmp, tmp_pre_date_range)
            #         # if len(swe) == 0:
            #         #     continue
            #         # swe_picked_vals = Tools().pick_vals_from_1darray(swe, tmp_pre_date_range)
            #
            #         plt.figure(figsize=(8, 6))
            #         # plt.plot(tmp_pre_date_range, pre_picked_vals, '--', c='blue', label='precipitation')
            #         # plt.plot(tmp_pre_date_range, tmp_picked_vals, '--', c='cyan', label='temperature')
            #         # plt.plot(tmp_pre_date_range, swe_picked_vals, '--', c='black', linewidth=2, label='SWE',
            #         #          zorder=99)
            #         # plt.plot(recovery_date_range, recovery_ndvi, c='g', linewidth=6, label='Recovery Period')
            #         plt.scatter(recovery_date_range, recovery_ndvi, c='g', label='Recovery Period')
            #         # plt.plot(date_range, spei_picked_vals, c='r', linewidth=6,
            #         #          label='drought Event')
            #         # plt.scatter(date_range, spei_picked_vals, c='r', zorder=99)
            #
            #         plt.plot(range(len(ndvi_gs)), ndvi_gs, '--', c='g', zorder=99, label='ndvi')
            #         plt.plot(range(len(spei_gs)), spei_gs, '--', c='r', zorder=99, label='drought index')
            #         # plt.plot(range(len(pre)), pre, '--', c='blue', zorder=99, label='Precip')
            #         # pre_picked = T.pick_vals_from_1darray(pre,recovery_date_range)
            #         # pre_mean = np.mean(pre_picked)
            #         # plt.plot(recovery_date_range,[pre_mean]*len(recovery_date_range))
            #         plt.legend()
            #
            #         minx = 9999
            #         maxx = -9999
            #
            #         for ii in recovery_date_range:
            #             if ii > maxx:
            #                 maxx = ii
            #             if ii < minx:
            #                 minx = ii
            #
            #         for ii in date_range_new:
            #             if ii > maxx:
            #                 maxx = ii
            #             if ii < minx:
            #                 minx = ii
            #
            #         # xtick = []
            #         # for iii in np.arange(len(ndvi)):
            #         #     year = 1982 + iii / 12
            #         #     mon = iii % 12 + 1
            #         #     mon = '%02d' % mon
            #         #     xtick.append('{}.{}'.format(year, mon))
            #         # # plt.xticks(range(len(xtick))[::3], xtick[::3], rotation=90)
            #         # plt.xticks(range(len(xtick)), xtick, rotation=90)
            #         plt.grid()
            #         plt.xlim(minx - 5, maxx + 5)
            #
            #         lon, lat, address = Tools().pix_to_address(pix)
            #         try:
            #             plt.title('lon:{:0.2f} lat:{:0.2f} address:{}\n'.format(lon, lat, address) +
            #                       'recovery_time:'+str(recovery_time)
            #                       )
            #
            #         except:
            #             plt.title('lon:{:0.2f} lat:{:0.2f}\n'.format(lon, lat)+
            #                       'recovery_time:' + str(recovery_time)
            #                       )
            #         plt.show()
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

    def search1(self, ndvi,ndvi_pred,drought_indx, event_start_index,search_end_indx):
        # print(event_start_index)
        if event_start_index+search_end_indx >= len(ndvi):
            return None
        selected_indx = []
        for i in range(event_start_index,event_start_index+search_end_indx):
            ndvi_i = ndvi[i]
            if ndvi_i < ndvi_pred[i]:
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
        self.this_class_arr = results_root_main_flow + 'arr\\Main_flow_Dataframe_NDVI_SPEI_legacy\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Main_flow_Dataframe_NDVI_SPEI_legacy\\'
        self.this_class_png = results_root_main_flow + 'png\\Main_flow_Dataframe_NDVI_SPEI_legacy\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)

        self.dff = self.this_class_arr + 'data_frame.df'

    def run(self):
        # 0 generate a void dataframe
        df = self.__gen_df_init()
        # self._check_spatial(df)
        # exit()
        # 1 add drought event and delta legacy into df
        # df = self.legacy_to_df(df)
        # df = self.delta_legacy_to_df(df)
        # df = self.legacy_trend_to_df(df)
        # 2 add lon lat into df
        # df = self.add_lon_lat_to_df(df)
        # 3 add iso-hydricity into df
        # df = self.add_isohydricity_to_df(df)
        # 4 add correlation into df
        # df = self.add_gs_sif_spei_correlation_to_df(df)
        # 5 add canopy height into df
        # df = self.add_canopy_height_to_df(df)
        # 6 add rooting depth into df
        # df = self.add_rooting_depth_to_df(df)
        # 7 add TWS into df
        # for year in range(4):
        #     print(year)
        #     df = self.add_TWS_to_df(df,year)
        # 8 add is gs into df
        # self.add_is_gs_drought_to_df(df)
        # 9 add landcover to df
        # 10 add kplc to df
        # df = self.add_koppen_landuse_to_df(df)
        # 11 add koppen to df
        # df = self.add_split_landuse_and_kp_to_df(df)
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
        df = self.add_drought_year_SOS(df)
        # -1 df to excel
        df = self.drop_duplicated_sample(df)
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
        HI_tif = data_root + '\\waterbalance\\HI_difference.tif'
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
        kp_f = data_root + 'Koppen\\cross_koppen_landuse_pix.npy'
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
        # df_f = Prepare_CWD_X_pgs_egs_lgs2().this_class_arr + 'prepare\\data_frame.df'
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
        df_drop_dup = df.drop_duplicates(subset=['pix','legacy','recovery_date_range'])
        return df_drop_dup
        # df_drop_dup.to_excel(self.this_class_arr + 'drop_dup.xlsx')
        pass

    def legacy_to_df(self,df):
        f = Main_flow_Recovery_time_Legacy().this_class_arr + 'Recovery_time_Legacy\\recovery_time_legacy_reg.pkl'
        events_dic = T.load_dict_from_binary(f)
        pix_list = []
        recovery_time_list = []
        drought_event_date_range_list = []
        recovery_date_range_list = []
        legacy_list = []

        for pix in tqdm(events_dic,desc='1. legacy_to_df'):
            # print(pix)
            # exit()
            events = events_dic[pix]
            for event in events:
                recovery_time = event['recovery_time']
                drought_event_date_range = event['drought_event_date_range']
                recovery_date_range = event['recovery_date_range']
                legacy = event['legacy']

                pix_list.append(pix)
                recovery_time_list.append(recovery_time)
                drought_event_date_range_list.append(tuple(drought_event_date_range))
                recovery_date_range_list.append(tuple(recovery_date_range))
                legacy_list.append(legacy)

        df['pix'] = pix_list
        df['drought_event_date_range'] = drought_event_date_range_list
        df['recovery_date_range'] = recovery_date_range_list
        df['recovery_time'] = recovery_time_list
        df['legacy'] = legacy_list
        return df
        pass

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
        fdir = data_root + r'Isohydricity\per_pix_all_year\\'
        dic = T.load_npy_dir(fdir)
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
        f = data_root + 'Canopy_Height\\per_pix\\Canopy_Height.npy'
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
        f = data_root + 'rooting_depth\\per_pix\\rooting_depth.npy'
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

    def add_TWS_to_df(self,df,year):

        fdir = data_root + 'TWS\\water_gap\\per_pix_anomaly\\'
        tws_dic = T.load_npy_dir(fdir)
        tws_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            drought_start = row.event[0]
            pix = row.pix
            if not pix in tws_dic:
                tws_list.append(np.nan)
                continue
            vals = tws_dic[pix]
            # print(len(vals))
            # exit()
            start_indx = drought_start + 12 * (year - 1)
            end_indx = drought_start + 12 * (year)
            if start_indx < 0:
                tws_list.append(np.nan)
                continue
            if end_indx >= len(vals):
                tws_list.append(np.nan)
                continue
            picked_indx = list(range(start_indx,end_indx))
            picked_val = T.pick_vals_from_1darray(vals,picked_indx)
            mean = np.nanmean(picked_val)
            tws_list.append(mean)
        if year == 0:
            year = -1
        df['TWS_{}'.format(year)] = tws_list

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
        fdir = data_root + r'Climate_408\\'
        for climate in os.listdir(fdir):
            f = fdir + climate + '\\delta\\delta.npy'
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
        fdir = data_root + r'Climate_408\\'
        for climate in os.listdir(fdir):
            f = fdir + climate + '\\CV\\CV.npy'
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
        fdir = data_root + r'Climate_408\\'
        for climate in os.listdir(fdir):
            f = fdir + climate + '\\CV\\CV_delta.npy'
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
        fdir = data_root + r'Climate_408\\'
        for climate in os.listdir(fdir):
            f = fdir + climate + '\\trend\\trend.npy'
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
        sand_tif = data_root + 'HWSD\\T_SAND_resample.tif'
        silt_tif = data_root + 'HWSD\\T_SILT_resample.tif'
        clay_tif = data_root + 'HWSD\\T_CLAY_resample.tif'

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
        f = data_root + 'SPEI\\delta\\delta.npy'
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
        tif = data_root + 'HWSD\\awc_05.tif'
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
                'transform_early_peak_late_dormant_period_annual\\early_start.npy'
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

        # class Main_flow_Dataframe:
#
#     def __init__(self):
#         self.this_class_arr = results_root_main_flow + 'arr\\Main_flow_Dataframe\\'
#         self.this_class_tif = results_root_main_flow + 'tif\\Main_flow_Dataframe\\'
#         self.this_class_png = results_root_main_flow + 'png\\Main_flow_Dataframe\\'
#
#         Tools().mk_dir(self.this_class_arr, force=True)
#         Tools().mk_dir(self.this_class_tif, force=True)
#         Tools().mk_dir(self.this_class_png, force=True)
#
#         self.dff = self.this_class_arr + 'data_frame.df'
#
#     def run(self):
#         # 0 generate a void dataframe
#         df = self.__gen_df_init()
#         # self._check_spatial(df)
#         # exit()
#         # 1 add drought event into df
#         df = self.events_to_df(df)
#         # 2 add lon lat into df
#         # df = self.add_lon_lat_to_df(df)
#         # 3 add iso-hydricity into df
#         # df = self.add_isohydricity_to_df(df)
#         # 4 add correlation into df
#         # df = self.add_gs_sif_spei_correlation_to_df(df)
#         # 5 add canopy height into df
#         # df = self.add_canopy_height_to_df(df)
#         # 6 add rooting depth into df
        df = self.add_rooting_depth_to_df(df)
#         # 7 add TWS into df
#         # for year in range(4):
#         #     print(year)
#         #     df = self.add_TWS_to_df(df,year)
#         # 8 add is gs into df
#         # self.add_is_gs_drought_to_df(df)
#
#         # -1 df to excel
#         T.save_df(df,self.dff)
#         self.__df_to_excel(df,self.dff)
#         pass
#
#
#     def drop_duplicated_sample(self,df):
#         df_drop_dup = df.drop_duplicates(subset=['pix','Y','recovery_date_range'])
#         return df_drop_dup
#         # df_drop_dup.to_excel(self.this_class_arr + 'drop_dup.xlsx')
#         pass
#
#
#     def _check_spatial(self,df):
#         spatial_dic = {}
#         for i,row in df.iterrows():
#             pix = row.pix
#             spatial_dic[pix] = row.lon
#             # spatial_dic[pix] = row.isohydricity
#         arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
#         DIC_and_TIF().plot_back_ground_arr()
#         plt.imshow(arr)
#         plt.show()
#         pass
#
#     def __load_df(self):
#         dff = self.dff
#         df = T.load_df(dff)
#         T.print_head_n(df)
#         print('len(df):',len(df))
#         return df,dff
#
#     def __gen_df_init(self):
#         df = pd.DataFrame()
#         if not os.path.isfile(self.dff):
#             T.save_df(df,self.dff)
#         else:
#             df,dff = self.__load_df()
#             return df
#             # raise Warning('{} is already existed'.format(self.dff))
#
#     def __df_to_excel(self,df,dff,head=1000):
#         if head == None:
#             df.to_excel('{}.xlsx'.format(dff))
#         else:
#             df = df.head(head)
#             df.to_excel('{}.xlsx'.format(dff))
#
#         pass
#
#     def events_to_df(self,df):
#         fdir = Main_Flow_Pick_drought_events().this_class_arr + 'drought_events\\'
#         events_dic = T.load_npy_dir(fdir)
#         pix_list = []
#         event_list = []
#         for pix in tqdm(events_dic,desc='1. events_to_df'):
#             events = events_dic[pix]
#             for event in events:
#                 pix_list.append(pix)
#                 event_list.append(event)
#         df['pix'] = pix_list
#         df['event'] = event_list
#         return df
#         pass
#
#     def add_lon_lat_to_df(self, df):
#         # DIC_and_TIF().spatial_tif_to_lon_lat_dic()
#         pix_to_lon_lat_dic_f = DIC_and_TIF().this_class_arr + 'pix_to_lon_lat_dic.npy'
#         lon_lat_dic = T.load_npy(pix_to_lon_lat_dic_f)
#         # print(pix)
#         lon_list = []
#         lat_list = []
#         for i, row in tqdm(df.iterrows(), total=len(df), desc='adding lon lat into df'):
#             pix = row.pix
#             lon, lat = lon_lat_dic[pix]
#             lon_list.append(lon)
#             lat_list.append(lat)
#         df['lon'] = lon_list
#         df['lat'] = lat_list
#
#         return df
#
#     def add_isohydricity_to_df(self,df):
#         fdir = data_root + r'Isohydricity\per_pix_all_year\\'
#         dic = T.load_npy_dir(fdir)
#         iso_hyd_list = []
#         for i,row in tqdm(df.iterrows(),total=len(df),desc='adding iso-hydricity to df'):
#             pix = row.pix
#             if not pix in dic:
#                 iso_hyd_list.append(np.nan)
#                 continue
#             isohy = dic[pix]
#             iso_hyd_list.append(isohy)
#         df['isohydricity'] = iso_hyd_list
#
#         return df
#
#
#     def add_gs_sif_spei_correlation_to_df(self,df):
#
#         corr_dic = Correlation_CSIF_SPEI().correlation()
#         corr_list = []
#         corr_p_list = []
#         for i,row in tqdm(df.iterrows(),total=len(df)):
#             pix = row.pix
#             if not pix in corr_dic:
#                 corr_list.append(np.nan)
#                 corr_p_list.append(np.nan)
#                 continue
#             corr,p = corr_dic[pix]
#             corr_list.append(corr)
#             corr_p_list.append(p)
#         df['gs_sif_spei_corr'] = corr_list
#         df['gs_sif_spei_corr_p'] = corr_p_list
#
#
#         return df
#
#
#     def add_canopy_height_to_df(self,df):
#         f = data_root + 'Canopy_Height\\per_pix\\Canopy_Height.npy'
#         dic = T.load_npy(f)
#
#         val_list = []
#         for i, row in tqdm(df.iterrows(), total=len(df)):
#             pix = row.pix
#
#             if not pix in dic:
#                 val_list.append(np.nan)
#                 continue
#
#             val = dic[pix]
#             val_list.append(val)
#
#         df['canopy_height'] = val_list
#         return df
#
#     def add_rooting_depth_to_df(self,df):
#         f = data_root + 'rooting_depth\\per_pix\\rooting_depth.npy'
#         dic = T.load_npy(f)
#
#         val_list = []
#         for i,row in tqdm(df.iterrows(),total=len(df)):
#             pix = row.pix
#
#             if not pix in dic:
#                 val_list.append(np.nan)
#                 continue
#
#             val = dic[pix]
#             val_list.append(val)
#
#         df['rooting_depth'] = val_list
#         return df
#
#     def add_TWS_to_df(self,df,year):
#
#         fdir = data_root + 'TWS\\water_gap\\per_pix_anomaly\\'
#         tws_dic = T.load_npy_dir(fdir)
#         tws_list = []
#         for i,row in tqdm(df.iterrows(),total=len(df)):
#             drought_start = row.event[0]
#             pix = row.pix
#             if not pix in tws_dic:
#                 tws_list.append(np.nan)
#                 continue
#             vals = tws_dic[pix]
#             # print(len(vals))
#             # exit()
#             start_indx = drought_start + 12 * (year - 1)
#             end_indx = drought_start + 12 * (year)
#             if start_indx < 0:
#                 tws_list.append(np.nan)
#                 continue
#             if end_indx >= len(vals):
#                 tws_list.append(np.nan)
#                 continue
#             picked_indx = list(range(start_indx,end_indx))
#             picked_val = T.pick_vals_from_1darray(vals,picked_indx)
#             mean = np.nanmean(picked_val)
#             tws_list.append(mean)
#         if year == 0:
#             year = -1
#         df['TWS_{}'.format(year)] = tws_list
#
#         # exit()
#         return df
#
#     def add_is_gs_drought_to_df(self,df):
#
#         is_gs_list = []
#         gs_mon = list(range(4,11))
#         for i,row in tqdm(df.iterrows(),total=len(df)):
#             drought_start = row.event[0]
#             mon = drought_start % 12 + 1
#             if mon in gs_mon:
#                 is_gs = 1
#             else:
#                 is_gs = 0
#             is_gs_list.append(is_gs)
#         df['is_gs'] = is_gs_list
#         return df
#
#         pass


class Main_flow_RF:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Main_flow_RF\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Main_flow_RF\\'
        self.this_class_png = results_root_main_flow + 'png\\Main_flow_RF\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        self.get_feature_importance()
        # self.plot_results()
        pass



    def rf_importance_train_results(self,X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.2)
        rf = RandomForestRegressor(n_estimators=100)
        # rf = RandomForestClassifier(n_estimators=300, random_state=42)
        # rf.fit(X_train, y_train)
        rf.fit(X, y)
        # y_pred = rf.predict(X)
        # y_pred = rf.predict(X_test)
        # r2 = rf.score(X_test, y_test)
        r2 = rf.score(X, y)
        # plt.scatter(y_pred,y)
        # plt.show()
        # KDE_plot().plot_scatter(y_test,y_pred,plot_fit_line=True,max_n=10000,is_plot_1_1_line=True,title='aaa')
        # plt.show()
        importances = rf.feature_importances_
        importances_dic = dict(zip(X.columns, importances))
        return importances_dic, r2



    def importance_train_results_BRT(self,X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.2)
        reg = GradientBoostingRegressor()
        # rf = RandomForestClassifier(n_estimators=300, random_state=42)
        # reg.fit(X_train, y_train)
        reg.fit(X, y)
        y_pred = reg.predict(X_test)
        # r2 = reg.score(X_test, y_test)
        r2 = reg.score(X, y)

        importances = reg.feature_importances_
        importances_dic = dict(zip(X.columns, importances))
        # print(r2)
        # plt.scatter(y_test, y_pred)
        # plt.figure()
        # plt.bar(X.columns,importances)
        # plt.show()
        return importances_dic, r2


    def importance_train_results_linear(self,X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.2)
        # rf = RandomForestRegressor(n_estimators=100)
        # rf = RandomForestClassifier(n_estimators=300, random_state=42)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2 = reg.score(X_test, y_test)
        importances = reg.coef_
        importances_dic = dict(zip(X.columns, importances))
        return importances_dic, r2

    def permutation_train_results(self,X, y):
        # print(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.5)
        rf = RandomForestRegressor(n_estimators=100)
        # rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf.fit(X_train, y_train)
        # rf.fit(X, y)
        y_pred = rf.predict(X_test)
        # y_pred = rf.predict(X_train)
        # y_pred = rf.predict(X)
        # print(y_test)
        # y_test = np.array(y_test)
        # y_test = np.reshape(y_test,(-1,1))
        # y_pred = np.reshape(y_pred,(-1,1))
        # print(y_pred)
        r2 = rf.score(X_test,y_test)
        # r2 = rf.score(X,y)
        # plt.scatter(y_test,y_pred)
        KDE_plot().plot_scatter(y_test,y_pred,plot_fit_line=True,is_plot_1_1_line=True,title='r2:{:0.2f}'.format(r2),s=2)
        # KDE_plot().plot_scatter(y_train,y_pred,plot_fit_line=True,is_plot_1_1_line=True,title='r2:{:0.2f}'.format(r2),s=2)
        # KDE_plot().plot_scatter(y,y_pred,plot_fit_line=True,is_plot_1_1_line=True,title='r2:{:0.2f}'.format(r2),s=2)
        plt.show()
        # r2 = rf.score(X, y)
        # result = permutation_importance(rf, X_train, y_train, scoring=None,
        #                                 n_repeats=10, random_state=42,
        #                                 n_jobs=5)
        # result = permutation_importance(rf, X_test, y_test, scoring=None,
        #                                 n_repeats=10, random_state=42,
        #                                 n_jobs=5)
        result = permutation_importance(rf, X, y, scoring=None,
                                        n_repeats=10, random_state=42,
                                        n_jobs=5)
        importances = result.importances_mean
        importances_dic = dict(zip(X.columns, importances))
        # plt.scatter(y_test,y_pred)
        # plt.show()
        return importances_dic, r2


    def discard_hierarchical_clustering(self,df, var_list, dest_Y, t=0.0, isplot=False):
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

    def discard_vif_vars(self,df, vars_list,dest_Y):
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
    def drop_duplicated_delta_legacy(self,df):
        df_drop_dup = df.drop_duplicates(subset=['pix','delta_legacy'])
        return df_drop_dup

    def __df_to_excel(self,df,dff,head=1000):
        if head == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            df = df.head(head)
            df.to_excel('{}.xlsx'.format(dff))

        pass

    def get_feature_importance(self):

        x_vars, Y_var = Global_vars().variables()

        dest_var = Y_var
        outdir = self.this_class_arr + 'get_feature_importance\\'
        outf = outdir + 'permutation_RF'
        # outf = outdir + 'BRT'
        # outf = outdir + 'RF'
        # outf = outdir + 'linear'
        T.mk_dir(outdir,force=True)
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        df = Global_vars().clean_df(df)
        print(df.columns)
        # kl_list = list(set(list(df['lc'])))
        kl_list = list(set(list(df['climate_zone'])))
        kl_list.remove(None)
        kl_list.sort()
        results_dic = {}
        for kl in kl_list:
            print(kl)
            vars_list = x_vars
            # df_kl = df[df['lc'] == kl]
            df_kl = df[df['climate_zone'] == kl]
            df_kl = df_kl.replace([np.inf, -np.inf], np.nan)
            all_vars_list = copy.copy(vars_list)
            all_vars_list.append(dest_var)
            XXX = df_kl[vars_list]
            if len(XXX) < 100:
                print('{} sample number < 100'.format(kl))
                continue
            spatial_dic = {}
            for i,row in df_kl.iterrows():
                pix = row.pix
                spatial_dic[pix] = 1
            ############################################
            ############################################
            # DIC_and_TIF().plot_back_ground_arr()
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            # plt.imshow(arr,cmap='gray')
            # print(len(df_kl))
            # plt.show()
            ############################################
            ############################################
            # vif_selected_features = self.discard_vif_vars(XXX, vars_list,dest_var)
            # selected_features = self.discard_hierarchical_clustering(XXX, vif_selected_features,dest_var, t=1, isplot=False)
            # print(vif_discard_vars)
            # exit()
            selected_features = x_vars
            print(selected_features)
            df1 = pd.DataFrame(df_kl)
            vars_list1 = list(set(selected_features))
            vars_list1.sort()
            vars_list1.append(dest_var)
            XX = df1[vars_list1]
            XX = XX.dropna()
            vars_list1.remove(dest_var)
            X = XX[vars_list1]
            Y = XX[dest_var]


            if len(df1) < 100:
                print('{} df1 sample number < 100'.format(kl))
                continue
            # X = X.dropna()
            # importances_dic_permutation, r2 = self.rf_importance_train_results(X, Y)
            # importances_dic_permutation, r2 = self.importance_train_results_linear(X, Y)
            importances_dic_permutation, r2 = self.permutation_train_results(X, Y)
            # importances_dic_permutation, r2 = self.importance_train_results_BRT(X, Y)
            results_dic[kl] = (importances_dic_permutation,r2)
            labels = []
            importance = []
            for key in importances_dic_permutation:
                labels.append(key)
                importance.append(importances_dic_permutation[key])
        fw = outf+'.txt'
        fw = open(fw,'w')
        fw.write(str(results_dic))
        fw.close()


        pass

    def plot_results(self):
        fdir = self.this_class_arr + 'get_feature_importance\\'
        # f = fdir + 'linear.txt'
        # f = fdir + 'permutation_RF.txt'
        f = fdir + 'RF.txt'
        results_dic = T.load_dict_txt(f)
        for i in results_dic:
            imp = results_dic[i][0]
            r2 = results_dic[i][1]
            x_list = Global_vars().variables()[0]
            x_imp = []
            for x in x_list:
                x_imp.append(imp[x])
            plt.figure()
            plt.bar(list(range(len(x_imp))),x_imp)
            plt.xticks(list(range(len(x_imp))),x_list,rotation=90)
            plt.title('{} r2:{}'.format(i,r2))
            plt.tight_layout()
            # print(imp)
        plt.show()
        pass

class Main_flow_correlation:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Main_flow_correlation\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Main_flow_correlation\\'
        self.this_class_png = results_root_main_flow + 'png\\Main_flow_correlation\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass


    def run(self):
        self.run_corr()
        pass

    def load_df(self):

        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        T.print_head_n(df)
        return df


    def single_corr(self,X, y):
        corr_dic = {}
        for i,x in enumerate(X):
            val = X[x]
            # print(y)
            r, p = stats.pearsonr(val, y)
            # corr_list.append((r,p))
            corr_dic[x] = (r,p)
        return corr_dic
        # return r, p


    def run_corr(self):

        x_vars, Y_var = Global_vars().variables()

        dest_var = Y_var
        outdir = self.this_class_arr + 'corr\\'
        outf = outdir + 'corr'
        # outf = outdir + 'linear'
        T.mk_dir(outdir,force=True)
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        df = Global_vars().clean_df(df)
        print(df.columns)
        # kl_list = list(set(list(df['lc'])))
        kl_list = list(set(list(df['climate_zone'])))
        kl_list.remove(None)
        kl_list.sort()
        results_dic = {}
        for kl in kl_list:
            print(kl)
            vars_list = x_vars

            # df_kl = df[df['lc'] == kl]
            df_kl = df[df['climate_zone'] == kl]
            df_kl = df_kl.replace([np.inf, -np.inf], np.nan)
            all_vars_list = copy.copy(vars_list)
            all_vars_list.append(dest_var)
            XXX = df_kl[vars_list]
            if len(XXX) < 100:
                print('{} sample number < 100'.format(kl))
                continue
            selected_features = x_vars
            print(selected_features)
            df1 = pd.DataFrame(df_kl)
            vars_list1 = list(set(selected_features))
            vars_list1.sort()
            vars_list1.append(dest_var)
            XX = df1[vars_list1]
            XX = XX.dropna()
            vars_list1.remove(dest_var)
            X = XX[vars_list1]
            Y = XX[dest_var]
            if len(df1) < 100:
                print('{} df1 sample number < 100'.format(kl))
                continue
            # X = X.dropna()
            corr_dic = self.single_corr(X, Y)
            results_dic[kl] = corr_dic
        fw = outf+'.txt'
        fw = open(fw,'w')
        fw.write(str(results_dic))
        fw.close()


        pass


class Main_flow_Hot_Map_corr_RF:
    '''
    permutation importance
    RF Work flow
    '''
    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Main_flow_Hot_Map_corr_RF\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Main_flow_Hot_Map_corr_RF\\'
        self.this_class_png = results_root_main_flow + 'png\\Main_flow_Hot_Map_corr_RF\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        self.hotmap()
        pass

    def load_df(self):

        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        T.print_head_n(df)
        return df


    def sort_dic(self,indic):

        val_list = []
        dic_label = []
        for key in indic:
            dic_label.append(key)
            val_list.append(indic[key])
        sort_indx = np.argsort(val_list)
        val_list = np.array(val_list)
        dic_label = np.array(dic_label)
        sort_dic = dict(zip(dic_label[sort_indx], range(1, len(dic_label) + 1)))
        return sort_dic
        pass



    def __r_to_color(self,importance,xmin=-0.3,xmax=0.3,color_class=256):
        if np.isnan(importance):
            importance = 0.
        importance_range = np.linspace(xmin, xmax, color_class)
        if importance < xmin:
            pos_indx = 0
        elif importance > xmax:
            pos_indx = len(importance_range) - 1
        else:
            pos_indx = int(round(((importance - xmin) / (xmax - xmin) * len(importance_range)), 0)) - 1

        cmap = sns.diverging_palette(236, 0, s=99, l=50, n=color_class, center="light")
        # plt.figure()
        # sns.palplot(cmap)
        # plt.show()
        c = cmap[pos_indx]
        return c
        pass



    def __imp_to_size_permutation(self,imp):
        if imp <= 5:
            size = 1
        elif 5 <= imp < 10:
            size = 20
        elif 10 <= imp < 20:
            size = 70
        elif 20 <= imp:
            size = 120
        elif np.isnan(imp):
            size = 0.
        else:
            # print imp
            size = imp
            # raise UserWarning('error')
        return size
        pass

    def __imp_to_size_jenks_normalized(self,var_input,result_dic,var_list):
        # from math import isfinite
        # print np.isfinite(123)
        import jenkspy
        from jenkspy import JenksNaturalBreaks
        if not var_input in result_dic:
            return 0,[]
        imp_list = []
        for var in var_list:
            if not var in result_dic:
                imp = 0
            else:
                imp = result_dic[var]
            imp_list.append(imp)


        jnb = JenksNaturalBreaks(nb_class=4)
        jnb.fit(imp_list)
        breaks = jnb.inner_breaks_
        size_list = [1,10,50,150]
        # print breaks
        # plt.bar(var_list,imp_list)
        # plt.show()
        # max_imp = np.max(imp_list)
        # min_imp = np.min(imp_list)
        # ref_imp_dic = {}
        # for var in result_dic:
        #     imp = result_dic[var]
        #     ref_imp = (imp - min_imp)/(max_imp - min_imp)
        #     ref_imp = ref_imp * 100
        #     ref_imp_dic[var] = ref_imp
        imp = result_dic[var_input]
        if imp <= breaks[0]:
            size = size_list[0]
        elif imp >= breaks[-1]:
            size = size_list[-1]
        else:
            size = -999
            for i in range(len(breaks)):
                if breaks[i] <= imp < breaks[i+1]:
                    size_class = i + 1
                    size = size_list[size_class]
                    break
        # for i in range(len(breaks)):


        # if imp <= breaks[0]:
        #     size = 1
        # elif breaks[0] <= imp < breaks[1]:
        #     size = 10
        # elif breaks[1] <= imp < breaks[2]:
        #     size = 40
        # elif breaks[2] <= imp:
        #     size = 180
        # else:
        #     size = -1
        if size == -999:
            print(imp,breaks)
            exit()
        return size,breaks
        # return imp


    def __imp_to_size_rank(self,var,result_dic,var_list):


        rank_list = []
        for i in var_list:
            rank = result_dic[i]
            rank_list.append(rank)
        size_list = [1,20,70,120]
        size_dic = {}
        for ii,rank in enumerate(rank_list):
            size = size_list[(rank*len(size_list))//(len(var_list)+1)]
            # print imp,size
            size_dic[var_list[ii]] = size

        return size_dic[var]
        pass



    def hotmap(self):
        x_list,dest_Y = Global_vars().variables()
        outpngdir = self.this_class_png + 'hotmap_lc\\'
        T.mk_dir(outpngdir)
        ##########################################################   圈   ###########################################################################
        rf_result_f = Main_flow_RF().this_class_arr + 'get_feature_importance\\permutation_RF.txt'
        # rf_result_f = Main_flow_RF().this_class_arr + 'get_feature_importance\\BRT.txt'
        # rf_result_f = Main_flow_RF().this_class_arr + 'get_feature_importance\\RF.txt'
        # rf_result_f = Main_flow_RF().this_class_arr + 'get_feature_importance\\linear.txt'
        ##########################################################   圈   ###########################################################################
        ##########################################################  背景  ###########################################################################
        partial_corr_result_f = Main_flow_correlation().this_class_arr + 'corr\\corr.txt'
        ##########################################################  背景  ###########################################################################


        var_list = x_list
        var_list = var_list[::-1]
        # print rf_result_f
        # rf_result_dic = T.load_npy(rf_result_f)
        rf_result_dic = eval(open(rf_result_f,'r').read())
        # partial_corr_result_dic = T.load_npy(partial_corr_result_f)
        partial_corr_result_dic = T.load_dict_txt(partial_corr_result_f)

        # print rf_result_dic
        # pause()
        print(partial_corr_result_dic)
        # exit()
        # continue
        # df = self.load_df()
        # print df.columns
        # eln_lc_list = []
        # # for timing in Global_vars().timing_list():
        # for lc in Global_vars().landuse_list():
        #     for timing in range(11):
        #         key = '{}_{}'.format(timing,lc)
        #         eln_lc_list.append(key)
        lc_list = Global_vars().koppen_landuse()
        # lc_list = Global_vars().landuse_list()
        # plt.figure(figsize=(4,6.2))
        y = 0
        y_labels = []
        imps_list = []
        R2_list = []
        x_labels = []
        for key in lc_list:
            # for key in rf_result_dic:
            # print key
            if not key in partial_corr_result_dic:
                continue
            corr_dic = partial_corr_result_dic[key]
            # print(corr_dic)
            # exit()
            if not key in rf_result_dic:
                y += 1
                y_labels.append('')
                continue
            result_dic,r2 = rf_result_dic[key]
            R2_list.append(r2)
            x_labels.append(key)

            # print result_dic
            # exit()

            y_labels.append('{}_r2:{:0.2f}'.format(key,r2))

            importance_list = []
            importance_label = []
            for var in result_dic:
                importance_label.append(var)
                importance_list.append(result_dic[var])
            sort_indx = np.argsort(importance_list)
            importance_list = np.array(importance_list)
            importance_label = np.array(importance_label)
            sort_dic = dict(zip(importance_label[sort_indx],range(1,len(importance_label)+1)))
            # print sort_dic
            # print corr_dic
            # exit()

            # result_dic = sort_dic
            print(result_dic)
            x = 0
            for var in var_list:
                if not var in result_dic:
                    # print 'aaa'
                    imp = np.nan
                else:
                    # permutation importance
                    # imp = result_dic[var]/r2*50.
                    # imp = result_dic[var]*50.
                    imp = result_dic[var]*100.
                    # impurity importance
                    # imp = result_dic[var]
                r,p = corr_dic[var]
                color = self.__r_to_color(r)
                imps_list.append(imp)
                # permutation importance
                size = self.__imp_to_size_permutation(imp)
                # impurity importance
                # size = self.__imp_to_size_rank(var,result_dic,var_list)
                # size,breaks = self.__imp_to_size_jenks_normalized(var,result_dic,var_list)

                # print breaks
                # exit()
                # if size == 0.:
                #     plt.scatter(y, x, marker='x', alpha=1, c='black', zorder=99, edgecolors='black', s=20,
                #                 linewidths=2)
                # else:
                # if p < 0.1:
                #     plt.scatter(y, x, marker='o', alpha=1, c='', zorder=99, edgecolors='black', s=size, linewidths=2)
                # else:
                #     plt.scatter(y, x, marker='s', alpha=1, c=color, zorder=0, s=200)
                #     plt.scatter(y, x, marker='x', alpha=1, zorder=99, c='black', s=50, linewidth=2)
                plt.scatter(y, x, marker='o', alpha=1, color='', zorder=99, edgecolors='black', s=size, linewidths=2)
                plt.scatter(y, x, marker='s', alpha=1, color=color, zorder=0,s=200)

                x += 1
            y += 1
        plt.yticks(range(len(var_list)),var_list)
        plt.xticks(range(len(y_labels)),y_labels,rotation=90)
        plt.axis('equal')
        # plt.title()
        # plt.scatter(-1, 0, marker='o', alpha=1, c='', zorder=99, edgecolors='black', s=2, linewidths=2, label='<5%')
        # plt.scatter(-1, 1, marker='o', alpha=1, c='', zorder=99, edgecolors='black', s=20, linewidths=2,
        #             label='5%<10%')
        # plt.scatter(-1, 2, marker='o', alpha=1, c='', zorder=99, edgecolors='black', s=70, linewidths=2,
        #             label='10%<20%')
        # plt.scatter(-1, 3, marker='o', alpha=1, c='', zorder=99, edgecolors='black', s=120, linewidths=2,
        #             label='>20%')
        #
        # plt.scatter(-1, 0, marker='o', alpha=1, c='white', zorder=99, edgecolors='white', s=2, linewidths=4)
        # plt.scatter(-1, 1, marker='o', alpha=1, c='', zorder=99, edgecolors='white', s=20, linewidths=4,
        #            )
        # plt.scatter(-1, 2, marker='o', alpha=1, c='', zorder=99, edgecolors='white', s=70, linewidths=4,
        #             )
        # plt.scatter(-1, 3, marker='o', alpha=1, c='', zorder=99, edgecolors='white', s=120, linewidths=4,
        #             )
        # plt.legend()
        # plt.title('{} winter'.format(with_winter_str))
        # plt.tight_layout()
        # plt.savefig(outpngdir + 'hotmap.pdf')
        # plt.close()
        # plt.show()
        # plt.figure()
        # imps_list = np.array(imps_list)
        # imps_list[imps_list<=0] = 0.
        # imps_list = T.remove_np_nan(imps_list)
        # plt.hist(imps_list,bins=30)

        plt.figure()
        plt.bar(x_labels,R2_list)
        plt.xticks(rotation=90)
        # plt.ylim(0,0.5)
        plt.tight_layout()
        plt.show()
        # plt.savefig(outpngdir + 'r2.pdf')
        # plt.close()


class Main_flow_pick_pure_forest_pixels():

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Main_flow_pick_pure_forest_pixels\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Main_flow_pick_pure_forest_pixels\\'
        self.this_class_png = results_root_main_flow + 'png\\Main_flow_pick_pure_forest_pixels\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)

    def run(self):
        self.ratio_of_forest()
        pass


    def ratio_of_forest(self):

        outf = self.this_class_tif + 'ratio_of_forest.tif'
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        pix = df.pix
        valid_pix = set(list(pix))
        glc_f = data_root + 'landcover\\glc2000_v1_1.tif'
        glc_arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(glc_f)

        rows = len(glc_arr)
        cols = len(glc_arr[0])

        start_lat = originY
        end_lat = originY+ pixelHeight * rows

        start_lon = originX
        end_lon = originX + pixelWidth * cols

        rows_resample = int(np.ceil((start_lat - end_lat)/0.5))
        cols_resample = int(np.ceil((end_lon - start_lon)/0.5))

        cols_dic = {}
        for c in range(cols_resample):
            cols_dic[c] = []
        rows_dic = {}
        for r in range(rows_resample):
            rows_dic[r] = []

        for c in range(cols):
            c_key = int(c // (cols/cols_resample))
            cols_dic[c_key].append(c)

        for r in range(rows):
            r_key = int(r // (rows/rows_resample))
            rows_dic[r_key].append(r)

        spatial_ratio_dic = {}
        for r_key in tqdm(rows_dic):
            for c_key in cols_dic:
                # print(c_key)
                # print(r_key)
                key = (r_key,c_key)
                if not key in valid_pix:
                    continue
                c_pix_list = cols_dic[c_key]
                r_pix_list = rows_dic[r_key]
                tot = 0.
                forest = 0.
                for c_high in c_pix_list:
                    for r_high in r_pix_list:
                        tot += 1
                        glc_val = glc_arr[r_high][c_high]
                        if glc_val <= 10:
                            forest += 1
                ratio = forest / tot
                spatial_ratio_dic[key] = ratio
                # print(key,ratio)
                        # pix_high_res = (c_high,r_high)
                        # pix_high_res_list.append(pix_high_res)
                # spatial_indx_dic[key] = pix_high_res_list
        ratio_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_ratio_dic)
        DIC_and_TIF().arr_to_tif(ratio_arr,outf)





        pass

class Main_flow_Partial_Dependence_Plots:
    '''
    Ref:
    https://towardsdatascience.com/looking-beyond-feature-importance-37d2807aaaa7
    '''
    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Main_flow_Partial_Dependence_Plots\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Main_flow_Partial_Dependence_Plots\\'
        self.this_class_png = results_root_main_flow + 'png\\Main_flow_Partial_Dependence_Plots\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass


    def run(self):
        df_f = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(df_f)
        x_vars,y_vars = Global_vars().variables()
        df = Global_vars().clean_df(df)
        self.partial_dependent_plot(df,x_vars,y_vars)

        pass


    def partial_dependent_plot(self,df,x_vars,y_vars):
        outpngdir = self.this_class_png + 'partial_dependent_plot\\'
        T.mk_dir(outpngdir)
        outdir = self.this_class_png + 'partial_dependent_plot\\'
        T.mk_dir(outdir,force=True)

        flag = 0
        plt.figure(figsize=(12, 8))
        for var in tqdm(x_vars):
            flag += 1
            ax = plt.subplot(4, 4, flag)
            vars_list = x_vars
            # print(timing)
            XXX = df[vars_list]
            # print(len(XXX))
            if len(XXX) < 100:
                continue
            selected_features = vars_list
            vars_list1 = copy.copy(selected_features)
            vars_list1.append(y_vars)
            XX = df[vars_list1]
            XX = XX.dropna()
            vars_list1.remove(y_vars)
            X = XX[vars_list1]
            Y = XX[y_vars]
            if len(df) < 100:
                continue
            # print(X)
            # print(Y)
            # exit()
            model, r2 = self.train_model(X, Y)
            print(r2)
            # exit()
            df_partial_plot = self.__get_PDPvalues(var, X, model)
            ppx = df_partial_plot[var]
            ppy = df_partial_plot['PDs']
            ppx_smooth = SMOOTH().smooth_convolve(ppx,window_len=11)
            ppy_smooth = SMOOTH().smooth_convolve(ppy,window_len=11)
            plt.plot(ppx_smooth, ppy_smooth, lw=2,)
            plt.xlabel(var)
            plt.ylabel(y_vars)
            # title = 'r2: {}'.format(r2)
            # plt.title(title)
            plt.tight_layout()
            # plt.legend()
            # plt.show()
            # plt.savefig(outpngdir + title + '.pdf',dpi=300)
            # plt.close()
        plt.show()




    def train_model(self,X,y):
        print(len(X))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.1)
        # rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        # rf = LinearRegression()
        rf.fit(X_train, y_train)
        r2 = rf.score(X_test,y_test)
        y_pred = rf.predict(X_test)
        # y_pred = rf.predict(X_train)
        # plt.scatter(y_pred,y_test)
        print(r2)
        # plt.scatter(y_pred,y_train)
        # plt.show()

        return rf,r2

    def __get_PDPvalues(self, col_name, data, model, grid_resolution=50):
        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution)
        Y_pdp = []
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict(Xnew)
            Y_pdp.append(np.mean(Y_temp))
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp})

    def __plot_PDP(self,col_name, data, model):
        df = self.__get_PDPvalues(col_name, data, model)
        plt.rcParams.update({'font.size': 16})
        plt.rcParams["figure.figsize"] = (6,5)
        fig, ax = plt.subplots()
        # ax.plot(data[col_name], np.zeros(data[col_name].shape)+min(df['PDs'])-1, 'k|', ms=15)  # rug plot
        ax.plot(df[col_name], df['PDs'], lw = 2)
        ax.set_ylabel('Recovery time')
        ax.set_xlabel(col_name)
        plt.tight_layout()
        return ax






def main():
    # Main_Flow_Pick_drought_events().run()
    # Main_flow_Dataframe().run()
    Main_flow_Dataframe_NDVI_SPEI_legacy().run()
    # Main_flow_Recovery_time_Legacy().run()
    # Main_flow_RF().run()
    # Main_flow_correlation().run()
    # Main_flow_Hot_Map_corr_RF().run()
    # Main_flow_pick_pure_forest_pixels().run()
    # Main_flow_Partial_Dependence_Plots().run()

    pass


if __name__ == '__main__':
    main()