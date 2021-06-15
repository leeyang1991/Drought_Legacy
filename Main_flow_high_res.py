# coding=utf-8
import numpy as np

from __init__ import *
# from Main_flow_csif_legacy_2002 import *

results_root_main_flow_high_res = this_root + 'results_root_main_flow_high_res/'
results_root_main_flow = results_root_main_flow_high_res
results_root = results_root_main_flow_high_res

class Global_vars:
    def __init__(self):
        # self.growing_date_range = list(range(5,11))
        self.vars_()
        self.tif_template_7200_3600 = this_root + 'conf/tif_template_005.tif'
        self.growing_date_range = self.gs_mons()
        pass

    def vars_(self):
        self.lc_broad_needle = ['Needleleaf','Broadleaf']
        self.drought_type = ['repeatedly_initial_spei12','repeatedly_subsequential_spei12']
        self.dominate = ['supply','demand']


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

    def growing_season_indx_to_all_year_indx(self,indexs):
        new_indexs = []
        for indx in indexs:
            n_year = indx // len(self.growing_date_range)
            res_mon = indx % len(self.growing_date_range)
            real_indx = n_year * 12 + res_mon + self.growing_date_range[0]
            new_indexs.append(real_indx)
        new_indexs = tuple(new_indexs)
        return new_indexs

    def pick_gs_vals(self,vals,gs_mons):
        picked_vals = []
        for i in range(len(vals)):
            mon = i % 12 + 1
            if mon in gs_mons:
                picked_vals.append(vals[i])
        return picked_vals

    def map_time_series_indx_and_gs_series_indx(self,time_series_length=192):
        gs_range = self.gs_mons()
        drought_event_date_range = range(time_series_length)
        map_dic = {}
        for i in drought_event_date_range:
            year = i // 12
            mon = i % 12 + 1
            if not mon in gs_range:
                continue
            indx_i = gs_range.index(mon)
            new_indx = year * len(gs_range) + indx_i
            map_dic[i] = new_indx
        return map_dic

    def clean_df(self,df):
        df = df[df['lat'] > 23]
        return df


class Main_Flow_Pick_drought_events:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr/Pick_drought_events/'
        self.this_class_tif = results_root_main_flow + 'tif/Pick_drought_events/'
        self.this_class_png = results_root_main_flow + 'png/Pick_drought_events/'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        # threshold = -1.2
        n = 4 * 12
        # self.single_events(n,threshold)
        self.repetitive_events(n)
        # self.check_single_events()
        # self.check_repetitive_events()
        # self.check_single_and_repetitive_events()
        pass




    def kernel_repetitive_events(self,parmas):
        fdir,f,outdir,n,_,product = parmas
        gs_mons = Global_vars().gs_mons()
        single_event_dic = {}
        dic = T.load_npy(fdir + f)
        fname = f.split('/')[-1]
        for pix in dic:
            vals = dic[pix]

            if product == 'VPD':
                vals = -vals
            # vals_detrend = signal.detrend(vals)
            # vals = vals_detrend
            # print(len(vals))
            # plt.plot
            threshold = np.quantile(vals, 0.05)
            # print('threshold',threshold)
            # plt.plot(vals)
            # plt.show()
            event_list, key = self.kernel_find_drought_period([vals, pix, threshold])
            if len(event_list) == 0:
                continue
            # print('event_list',event_list)
            events_gs = []  # 只要生长季事件
            for i in event_list:
                level, drought_range = i
                is_gs = 0
                mon = drought_range[0] % 12 + 1
                if mon in gs_mons:
                    is_gs = 1
                if is_gs:
                    events_gs.append(drought_range)
            if len(events_gs) <= 1:  # 全时段只有一次以下的事件，则不存在repetitive事件
                continue
            # if events_4[0][0] - 36 < 0:
            #     continue

            # for i in range(len(events_gs)):
            #     # 如果事件距离第一年不足3年，则舍去，
            #     # 原因：不知道第一年前有没有干旱事件。
            #     # 还可改进：扩大找干旱事件的年份
            #     # 或者忽略
            #     # if events_gs[i][0] - n < 0:
            #     #     continue
            #     if i + 1 >= len(events_gs):
            #         continue
            #     if events_gs[i+1][0] - events_gs[i][0] < n:
            #         repetitive_events.append(events_gs[i])

            # find initial drought events
            initial_drought = []
            initial_drought_indx = []
            if events_gs[0][0] - n >= 0:
                initial_drought.append(events_gs[0])
                initial_drought_indx.append(0)
            for i in range(len(events_gs)):
                if i + 1 >= len(events_gs):
                    continue
                if events_gs[i + 1][0] - n > events_gs[i][0]:
                    initial_drought.append(events_gs[i + 1])
                    initial_drought_indx.append(i + 1)
            if len(initial_drought) == 0:
                continue

            repetitive_events = []
            for init_indx in initial_drought_indx:
                init_date = events_gs[init_indx][0]
                one_recovery_period = list(range(init_date, init_date + n))
                repetitive_events_i = []
                for i in range(len(events_gs)):
                    date_i = events_gs[i][0]
                    if date_i in one_recovery_period:
                        repetitive_events_i.append(tuple(events_gs[i]))
                if len(repetitive_events_i) > 1:
                    repetitive_events.append(repetitive_events_i)
            if len(repetitive_events) == 0:
                continue
            single_event_dic[pix] = repetitive_events
        np.save(outdir + fname, single_event_dic)

        pass

    def repetitive_events(self,n,threshold='auto'):

        # threshold = -2
        # n = 4 * 12
        # product = 'SPEI12'
        # fdir = data_root + '{}/per_pix/'.format(product)

        product = 'VPD'
        fdir = data_root + '{}/per_pix_anomaly/'.format(product)
        # fdir = data_root + '{}/per_pix/'.format(product)

        # product = 'Precip'
        # fdir = data_root + 'Precip_terra/per_pix_anomaly/'
        outdir = self.this_class_arr + 'repetitive_events_{}_{}/'.format(product,threshold)
        T.mk_dir(outdir, force=True)


        params = []
        for f in tqdm(os.listdir(fdir)):
            params.append([fdir,f,outdir,n,threshold,product])
            # self.kernel_repetitive_events([fdir,f,outdir,n,threshold,product])
        MULTIPROCESS(self.kernel_repetitive_events,params).run()
        # arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # DIC_and_TIF(Global_vars().tif_template_7200_3600).plot_back_ground_arr()
        # plt.show()
    # def pick_repetitive_events(self):
    #
    #     pass


    def single_events(self,n,threshold):
        # threshold = -2
        # n = 4 * 12
        outdir = self.this_class_arr + 'single_events_{}/'.format(threshold)
        # fdir = data_root + 'SPEI/per_pix_2002/'
        # fdir = data_root + 'CWD/CWD/per_pix_anomaly/'
        fdir = data_root + 'SPEI12/per_pix/'
        params = []
        for f in os.listdir(fdir):
            params.append([fdir + f,outdir,n,threshold])
            # self.pick_events([fdir + f,outdir])
        MULTIPROCESS(self.pick_events,params).run()
            # self.pick_events(fdir + f,outdir)
        pass

    def pick_events(self,params):
        # 前n个月和后n个月无极端干旱事件
        f, outdir,n,threshold = params
        # n = 4*12
        gs_mons = Global_vars().gs_mons()
        T.mk_dir(outdir,force=True)
        single_event_dic = {}
        single_event_number_dic = {}
        dic = T.load_npy(f)
        fname = f.split('/')[-1]
        for pix in dic:
            vals = dic[pix]
            # vals_detrend = signal.detrend(vals)
            # vals = vals_detrend
            # print(len(vals))
            # threshold = -1.5
            # plt.plot
            # threshold = np.quantile(vals, 0.05)
            # print('threshold',threshold)
            # plt.plot(vals)
            # plt.show()
            event_list,key = self.kernel_find_drought_period([vals,pix,threshold])
            # if len(event_list) == 0:
            #     continue
            # print('event_list',event_list)
            events_4 = []
            for i in event_list:
                level,drought_range = i
                is_gs = 0
                mon = drought_range[0] % 12 + 1
                if mon in gs_mons:
                    is_gs = 1
                if is_gs:
                    events_4.append(drought_range)
            # print(events_4)
            # exit()
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
            if len(single_event) == 0:
                continue
            # print(single_event)
            # print(n)
            # exit()
            # sleep(0.1)
            single_event_dic[pix] = single_event
            single_event_number_dic[pix] = len(single_event)
            # for evt in single_event:
            #     picked_vals = T.pick_vals_from_1darray(vals,evt)
            #     plt.scatter(evt,picked_vals,c='r')
            # plt.plot(vals)
            # plt.show()
        np.save(outdir + fname,single_event_dic)
        # events_number_arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(single_event_number_dic)
        # spatial_dic = {}
        # for pix in single_event_dic:
        #     evt_num = len(single_event_dic[pix])
        #     if evt_num == 0:
        #         continue
        #     spatial_dic[pix] = evt_num
        # DIC_and_TIF().plot_back_ground_arr()
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(events_number_arr)
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
            else:
                level = 0

            events_list.append([level, new_i])
        return events_list, key

    def check_single_events(self):

        fdir = self.this_class_arr + 'repetitive_events_Precip_auto/'
        flag = 0
        spatial_dic = {}
        for f in tqdm(os.listdir(fdir)):
            dic = T.load_npy(fdir + f)
            for pix in dic:
                events = dic[pix]
                if len(events) == 0:
                    continue
                for e in events:
                    flag += 1
                spatial_dic[pix] = len(events)
        print(flag)
        arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        DIC_and_TIF(Global_vars().tif_template_7200_3600).plot_back_ground_arr()
        plt.show()

    def check_repetitive_events(self):
        fdir = self.this_class_arr + 'repetitive_events_Precip_auto/'
        dic = T.load_npy_dir(fdir)
        spatial_dic = {}
        for pix in dic:
            events = dic[pix]
            spatial_dic[pix] = len(events)
        arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.colorbar()
        DIC_and_TIF(Global_vars().tif_template_7200_3600).plot_back_ground_arr()
        plt.show()


        pass

    def check_single_and_repetitive_events(self):
        outtif = self.this_class_tif + 'check_single_and_repetitive_events.tif'
        repetitive_events_fdir = self.this_class_arr + 'repetitive_events/'
        single_events_fdir = self.this_class_arr + 'single_events/'

        repetitive_events_dic = T.load_npy_dir(repetitive_events_fdir)
        single_events_dic = T.load_npy_dir(single_events_fdir)
        repetitive_events_spatial_dic = {}
        for pix in repetitive_events_dic:
            events = repetitive_events_dic[pix]
            repetitive_events_spatial_dic[pix] = len(events)

        single_events_spatial_dic = {}
        for pix in single_events_dic:
            events = single_events_dic[pix]
            single_events_spatial_dic[pix] = len(events)

        void_dic = DIC_and_TIF(Global_vars().tif_template_7200_3600).void_spatial_dic_zero()
        for pix in tqdm(void_dic):
            flag = 0
            if pix in repetitive_events_spatial_dic:
                flag += 1
                void_dic[pix] = 2
            if pix in single_events_spatial_dic:
                void_dic[pix] = 1
                flag += 1
            if flag == 2:
                void_dic[pix] = 3


        events_type_number_arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(void_dic)
        events_type_number_arr[events_type_number_arr==0]=np.nan
        DIC_and_TIF(Global_vars().tif_template_7200_3600).arr_to_tif(events_type_number_arr,outtif)
        # plt.imshow(events_type_number_arr)
        # DIC_and_TIF(Global_vars().tif_template_7200_3600).plot_back_ground_arr()
        # plt.show()


        pass


class Main_Flow_Pick_drought_events_05:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr/Main_Flow_Pick_drought_events_05/'
        self.this_class_tif = results_root_main_flow + 'tif/Main_Flow_Pick_drought_events_05/'
        self.this_class_png = results_root_main_flow + 'png/Main_Flow_Pick_drought_events_05/'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        threshold = -1.8
        n = 4 * 12
        self.single_events(n,threshold)
        self.repetitive_events(n,threshold)
        # self.check_single_events()
        # self.check_repetitive_events()
        self.check_single_and_repetitive_events()
        pass



    def kernel_repetitive_events(self,params):
        fdir,f,threshold,n,outdir=params
        gs_mons = Global_vars().gs_mons()
        single_event_dic = {}
        dic = T.load_npy(fdir + f)
        fname = f.split('/')[-1]
        for pix in dic:
            r, c = pix
            if r > 180:
                continue
            vals = dic[pix]
            # vals_detrend = signal.detrend(vals)
            # vals = vals_detrend
            # print(len(vals))
            # plt.plot
            # threshold = np.quantile(vals, 0.05)
            # print('threshold',threshold)
            # plt.plot(vals)
            # plt.show()
            event_list, key = self.kernel_find_drought_period([vals, pix, threshold])
            if len(event_list) == 0:
                continue
            # print('event_list',event_list)
            events_gs = []  # 只要生长季事件
            for i in event_list:
                level, drought_range = i
                is_gs = 0
                mon = drought_range[0] % 12 + 1
                if mon in gs_mons:
                    is_gs = 1
                if is_gs:
                    events_gs.append(drought_range)
            if len(events_gs) <= 1:  # 全时段只有一次以下的事件，则不存在repetitive事件
                continue
            # if events_4[0][0] - 36 < 0:
            #     continue

            # for i in range(len(events_gs)):
            #     # 如果事件距离第一年不足3年，则舍去，
            #     # 原因：不知道第一年前有没有干旱事件。
            #     # 还可改进：扩大找干旱事件的年份
            #     # 或者忽略
            #     # if events_gs[i][0] - n < 0:
            #     #     continue
            #     if i + 1 >= len(events_gs):
            #         continue
            #     if events_gs[i+1][0] - events_gs[i][0] < n:
            #         repetitive_events.append(events_gs[i])

            # find initial drought events
            initial_drought = []
            initial_drought_indx = []
            if events_gs[0][0] - n >= 0:
                initial_drought.append(events_gs[0])
                initial_drought_indx.append(0)
            for i in range(len(events_gs)):
                if i + 1 >= len(events_gs):
                    continue
                if events_gs[i + 1][0] - n > events_gs[i][0]:
                    initial_drought.append(events_gs[i + 1])
                    initial_drought_indx.append(i + 1)
            if len(initial_drought) == 0:
                continue

            repetitive_events = []
            for init_indx in initial_drought_indx:
                init_date = events_gs[init_indx][0]
                one_recovery_period = list(range(init_date, init_date + n))
                repetitive_events_i = []
                for i in range(len(events_gs)):
                    date_i = events_gs[i][0]
                    if date_i in one_recovery_period:
                        repetitive_events_i.append(tuple(events_gs[i]))
                if len(repetitive_events_i) > 1:
                    repetitive_events.append(repetitive_events_i)
            if len(repetitive_events) == 0:
                continue
            single_event_dic[pix] = repetitive_events
        np.save(outdir + fname, single_event_dic)

    def repetitive_events(self,n,threshold):

        outdir = self.this_class_arr + 'repetitive_events/'
        fdir = data_root + 'SPEI12/per_pix_05/'
        T.mk_dir(outdir, force=True)

        params = []
        for f in tqdm(os.listdir(fdir)):
            params.append([fdir,f,threshold,n,outdir])
        MULTIPROCESS(self.kernel_repetitive_events,params).run()

    def single_events(self,n,threshold):
        outdir = self.this_class_arr + 'single_events/'
        # fdir = data_root + 'SPEI/per_pix_2002/'
        # fdir = data_root + 'CWD/CWD/per_pix_anomaly/'
        fdir = data_root + 'SPEI12/per_pix_05/'
        params = []
        # n = 3*12
        # threshold = -1.5
        for f in os.listdir(fdir):
            params.append([fdir + f,outdir,n,threshold ])
            # self.pick_events([fdir + f,outdir])
        MULTIPROCESS(self.pick_events,params).run()
            # self.pick_events(fdir + f,outdir)
        pass

    def pick_events(self,params):
        # 前n个月和后n个月无极端干旱事件
        f, outdir,n,threshold = params
        # n = 4*12
        gs_mons = Global_vars().gs_mons()
        T.mk_dir(outdir,force=True)
        single_event_dic = {}
        single_event_number_dic = {}
        dic = T.load_npy(f)
        fname = f.split('/')[-1]
        for pix in dic:

            r,c = pix
            if r > 180:
                continue
            vals = dic[pix]
            # vals_detrend = signal.detrend(vals)
            # vals = vals_detrend
            # print(len(vals))
            # threshold = -1.5
            # plt.plot
            # threshold = np.quantile(vals, 0.05)
            # print('threshold',threshold)
            # plt.plot(vals)
            # plt.show()
            event_list,key = self.kernel_find_drought_period([vals,pix,threshold])
            # if len(event_list) == 0:
            #     continue
            # print('event_list',event_list)
            events_4 = []
            for i in event_list:
                level,drought_range = i
                is_gs = 0
                mon = drought_range[0] % 12 + 1
                if mon in gs_mons:
                    is_gs = 1
                if is_gs:
                    events_4.append(drought_range)
            # print(events_4)
            # exit()
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
            if len(single_event) == 0:
                continue
            # print(single_event)
            # print(n)
            # exit()
            # sleep(0.1)
            single_event_dic[pix] = single_event
            single_event_number_dic[pix] = len(single_event)
            # for evt in single_event:
            #     picked_vals = T.pick_vals_from_1darray(vals,evt)
            #     plt.scatter(evt,picked_vals,c='r')
            # plt.plot(vals)
            # plt.show()
        np.save(outdir + fname,single_event_dic)
        # events_number_arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(single_event_number_dic)
        # spatial_dic = {}
        # for pix in single_event_dic:
        #     evt_num = len(single_event_dic[pix])
        #     if evt_num == 0:
        #         continue
        #     spatial_dic[pix] = evt_num
        # DIC_and_TIF().plot_back_ground_arr()
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(events_number_arr)
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
            else:
                level = 0

            events_list.append([level, new_i])
        return events_list, key

    def check_single_events(self):

        fdir = self.this_class_arr + 'single_events/'
        flag = 0
        spatial_dic = {}
        for f in tqdm(os.listdir(fdir)):
            dic = T.load_npy(fdir + f)
            for pix in dic:
                events = dic[pix]
                if len(events) == 0:
                    continue
                for e in events:
                    flag += 1
                spatial_dic[pix] = len(events)
        print(flag)
        arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        DIC_and_TIF(Global_vars().tif_template_7200_3600).plot_back_ground_arr()
        plt.show()

    def check_repetitive_events(self):
        fdir = self.this_class_arr + 'repetitive_events/'
        dic = T.load_npy_dir(fdir)
        spatial_dic = {}
        for pix in dic:
            events = dic[pix]
            spatial_dic[pix] = len(events)
        arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.colorbar()
        DIC_and_TIF(Global_vars().tif_template_7200_3600).plot_back_ground_arr()
        plt.show()


        pass

    def check_single_and_repetitive_events(self):
        outtif = self.this_class_tif + 'check_single_and_repetitive_events.tif'
        repetitive_events_fdir = self.this_class_arr + 'repetitive_events/'
        single_events_fdir = self.this_class_arr + 'single_events/'

        repetitive_events_dic = T.load_npy_dir(repetitive_events_fdir)
        single_events_dic = T.load_npy_dir(single_events_fdir)
        repetitive_events_spatial_dic = {}
        for pix in repetitive_events_dic:
            events = repetitive_events_dic[pix]
            repetitive_events_spatial_dic[pix] = len(events)

        single_events_spatial_dic = {}
        for pix in single_events_dic:
            events = single_events_dic[pix]
            single_events_spatial_dic[pix] = len(events)

        void_dic = DIC_and_TIF().void_spatial_dic_zero()
        for pix in tqdm(void_dic):
            flag = 0
            if pix in repetitive_events_spatial_dic:
                flag += 1
                void_dic[pix] = 2
            if pix in single_events_spatial_dic:
                void_dic[pix] = 1
                flag += 1
            if flag == 2:
                void_dic[pix] = 3


        events_type_number_arr = DIC_and_TIF().pix_dic_to_spatial_arr(void_dic)
        events_type_number_arr[events_type_number_arr==0]=np.nan
        DIC_and_TIF().arr_to_tif(events_type_number_arr,outtif)
        # plt.imshow(events_type_number_arr)
        # DIC_and_TIF(Global_vars().tif_template_7200_3600).plot_back_ground_arr()
        # plt.show()


        pass




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
        threshold = '-2'
        self.single_events(threshold)
        # self.repetitive_events('auto','precip')
        # self.repetitive_events('auto','VPD')

        pass

    def single_events(self,threshold):
        # 1 cal recovery time
        event_dic,spei_dic,sif_dic = self.load_data_single_events(threshold=threshold)
        out_dir = self.this_class_arr + 'gen_recovery_time_legacy_single_events_{}/'.format(threshold)
        self.gen_recovery_time_legacy_single(event_dic,spei_dic, sif_dic,out_dir)
        pass

    def repetitive_events(self,threshold,product):
        event_dic, spei_dic, sif_dic = self.load_data_repetitive_events(condition='',threshold=threshold,product=product)
        out_dir = self.this_class_arr + 'gen_recovery_time_legacy_repetitive_events_{}_{}/'.format(product,threshold)
        self.gen_recovery_time_legacy_repetitive(event_dic,spei_dic, sif_dic,out_dir)


        pass


    def load_data_repetitive_events(self,condition='',threshold='',product=''):
        # events_dir = results_root_main_flow + 'arr/SPEI_preprocess/drought_events/'
        # SPEI_dir = data_root + 'SPEI/per_pix_clean/'
        # SIF_dir = data_root + 'CSIF/per_pix_anomaly_detrend/'

        if product == 'VPD':
            SPEI_dir = data_root + 'VPD/per_pix_anomaly/'
            events_dir = Main_Flow_Pick_drought_events().this_class_arr + 'repetitive_events_VPD_{}/'.format(threshold)

        elif product == 'precip':
            SPEI_dir = data_root + 'Precip_terra/per_pix_anomaly/'
            events_dir = Main_Flow_Pick_drought_events().this_class_arr + 'repetitive_events_Precip_{}/'.format(threshold)

        elif product == 'spei12':
            SPEI_dir = data_root + 'SPEI12/per_pix/'
            events_dir = Main_Flow_Pick_drought_events().this_class_arr + 'repetitive_events_{}/'.format(threshold)

        else:
            raise UserWarning('product error')


        SIF_dir = data_root + 'CSIF005/per_pix_anomaly_detrend/'

        event_dic = T.load_npy_dir(events_dir,condition)
        spei_dic = T.load_npy_dir(SPEI_dir,condition)
        sif_dic = T.load_npy_dir(SIF_dir,condition)

        if product == 'VPD':
            new_spei_dic = {}
            for pix in spei_dic:
                vals = spei_dic[pix]
                new_spei_dic[pix] = -vals
                # print(vals)
            spei_dic = new_spei_dic
        return event_dic,spei_dic,sif_dic
        pass


    def kernel_gen_recovery_time_legacy_repetitive(self,date_range,ndvi,spei,growing_date_range,):
        # print(date_range)
        # exit()
        # event_start_index = T.pick_min_indx_from_1darray(spei, date_range)
        event_start_index = date_range[0]
        # print(event_start_index)
        event_start_index_trans = self.__drought_indx_to_gs_indx(event_start_index, growing_date_range, len(ndvi))
        # print(event_start_index_trans)
        # exit()
        # if event_start_index_trans == None:
        #     print('__drought_indx_to_gs_indx failed')
        #     exit()
        #     return None
        ndvi_gs = self.__pick_gs_vals(ndvi, growing_date_range)
        spei_gs = self.__pick_gs_vals(spei, growing_date_range)
        # ndvi_gs_pred = self.__pick_gs_vals(ndvi_pred,growing_date_range)
        # print(len(ndvi_gs))
        # print(len(spei_gs))
        # exit()
        # print(len(ndvi_gs_pred))
        date_range_new = []
        for i in date_range:
            i_trans = self.__drought_indx_to_gs_indx(i, growing_date_range, len(ndvi))
            if i_trans != None:
                date_range_new.append(i_trans)
        # 1 挑出此次干旱事件的NDVI和SPEI值 （画图需要）
        # spei_picked_vals = Tools().pick_vals_from_1darray(spei, date_range)
        # 2 挑出此次干旱事件SPEI最低的索引
        # 在当前生长季搜索

        # 4 搜索恢复期
        # 4.1 获取growing season NDVI的最小值
        # 4.3 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
        search_end_indx = 5 * len(growing_date_range)
        recovery_range = self.search1(ndvi_gs, spei_gs, event_start_index_trans, search_end_indx)
        # continue
        # recovery_time, lag, recovery_start_gs, recovery_start, 'undefined'
        ###########################################
        ###########################################
        ###########################################
        # print('recovery_range',recovery_range)
        if recovery_range == None:
            # print('event_start_index+search_end_indx >= len(ndvi) '
            #       'event_start_index_trans',
            #       event_start_index_trans,)
            return None
        recovery_range = np.array(recovery_range)
        date_range_new = np.array(date_range_new)
        recovery_time = len(recovery_range)
        legacy = self.__cal_legacy(ndvi_gs, recovery_range)
        result_dic = {
            'recovery_time': recovery_time,
            'recovery_date_range': recovery_range,
            'drought_event_date_range': date_range_new,
            'carbon_loss': legacy,
        }
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
        # plt.show()
        #         # #################plot end ##################
        return result_dic
        pass


    def gen_recovery_time_legacy_repetitive(self, events, spei_dic, ndvi_dic, out_dir):
        '''
        生成全球恢复期
        :param interval: SPEI_{interval}
        :return:
        '''

        # pre_dic = Main_flow_Prepare().load_X_anomaly('PRE')

        growing_date_range = Global_vars().gs_mons()
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
                # print(len(ndvi))
                # print(len(spei))
                # exit()
                event = events[pix]
                # print(event)
                # exit()
                recovery_time_result = []

                for repetitive_events in event:
                    # print(repetitive_events)
                    # exit()

                    success = 1
                    drought_date_range_1 = repetitive_events[0]
                    drought_date_range_2 = repetitive_events[1]
                    results1 = self.kernel_gen_recovery_time_legacy_repetitive(drought_date_range_1,ndvi,spei,growing_date_range,)
                    results2 = self.kernel_gen_recovery_time_legacy_repetitive(drought_date_range_2,ndvi,spei,growing_date_range,)
                    if results1 != None and results2 != None:
                        # carbonloss1 = results1['carbon_loss']
                        # carbonloss2 = results2['carbon_loss']
                        recovery_time_result.append([results1,results2])
                    # exit()
                recovery_time_dic[pix] = recovery_time_result
            else:
                recovery_time_dic[pix] = []
        T.save_dict_to_binary(recovery_time_dic, outf)
        pass


    def load_data_single_events(self,condition='',threshold=''):
        # events_dir = results_root_main_flow + 'arr/SPEI_preprocess/drought_events/'
        # SPEI_dir = data_root + 'SPEI/per_pix_clean/'
        # SIF_dir = data_root + 'CSIF/per_pix_anomaly_detrend/'

        events_dir = Main_Flow_Pick_drought_events().this_class_arr + 'single_events_{}/'.format(threshold)
        SPEI_dir = data_root + 'SPEI12/per_pix/'
        SIF_dir = data_root + 'CSIF005/per_pix_anomaly/'

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

    def gen_recovery_time_legacy_single(self, events, spei_dic, ndvi_dic, out_dir):
        '''
        生成全球恢复期
        :param interval: SPEI_{interval}
        :return:
        '''

        # pre_dic = Main_flow_Prepare().load_X_anomaly('PRE')

        growing_date_range = Global_vars().gs_mons()
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
                # print(len(ndvi))
                # print(len(spei))
                # exit()
                event = events[pix]
                recovery_time_result = []
                for date_range in event:
                    # print(date_range)
                    # event_start_index = T.pick_min_indx_from_1darray(spei, date_range)
                    event_start_index = date_range[0]
                    event_start_index_trans = self.__drought_indx_to_gs_indx(event_start_index,growing_date_range,len(ndvi))
                    if event_start_index_trans == None:
                        continue
                    ndvi_gs = self.__pick_gs_vals(ndvi,growing_date_range)
                    spei_gs = self.__pick_gs_vals(spei,growing_date_range)
                    # ndvi_gs_pred = self.__pick_gs_vals(ndvi_pred,growing_date_range)
                    # print(len(ndvi_gs))
                    # print(len(spei_gs))
                    # exit()
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
                    # # lon, lat, address = Tools().pix_to_address(pix)
                    # # try:
                    # #     plt.title('lon:{:0.2f} lat:{:0.2f} address:{}\n'.format(lon, lat, address) +
                    # #               'recovery_time:'+str(recovery_time)
                    # #               )
                    # #
                    # # except:
                    # #     plt.title('lon:{:0.2f} lat:{:0.2f}\n'.format(lon, lat)+
                    # #               'recovery_time:' + str(recovery_time)
                    # #               )
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
        # selected_indx = [999999, 999999, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
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
        if len(selected_indx_) > 0:
            selected_indx_s.append(selected_indx_)
        if len(selected_indx_s) == 0:
            return None
        return selected_indx_s[0]
        pass

    def search1(self, ndvi,drought_indx, event_start_index,search_end_indx):
        # print(event_start_index)

        # if event_start_index+search_end_indx >= len(ndvi):
        #
        #     print('search1')
        #     print(event_start_index,search_end_indx,len(ndvi))
        #     plt.plot(ndvi,label='ndvi')
        #     plt.plot(drought_indx,label='drought_indx')
        #     plt.show()
        #     exit()
        #     return None
        selected_indx = []
        for i in range(event_start_index,event_start_index+search_end_indx):
            if i >= len(ndvi):
                break
            ndvi_i = ndvi[i]
            if ndvi_i < 0:
                selected_indx.append(i)
            else:
                selected_indx.append(999999)
        recovery_indx_gs = self.__split_999999(selected_indx)

        # if recovery_indx_gs == None:
        #     if np.std(ndvi) == 0:
        #         return None
        #     print(recovery_indx_gs)
        #     print('selected_indx',selected_indx)
        #
        #     # if 1:
        #     print(' event_start_index,search_end_indx', event_start_index,search_end_indx)
        #     plt.close()
        #     plt.plot(ndvi,label='ndvi')
        #     plt.plot(drought_indx,label='spei')
        #     plt.legend()
        #     plt.show()
        # print('recovery_indx_gs',recovery_indx_gs)

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
        # df = self.Carbon_loss_to_df(df)
        # self.minus_carbon_loss(df)
        # 2 add landcover to df
        # df = self.add_lon_lat_to_df(df)

        # df = self.add_landcover_to_df(df)
        # df = self.landcover_compose(df)
        # df = self.add_min_precip_to_df(df)
        # df = self.add_min_precip_anomaly_to_df(df)
        # df = self.add_max_vpd_to_df(df)
        # df = self.add_max_vpd_anomaly_to_df(df)
        # df = self.add_mean_precip_anomaly_to_df(df)
        # df = self.add_mean_vpd_anomaly_to_df(df)
        # df = self.add_mean_precip_to_df(df)
        # df = self.add_mean_vpd_to_df(df)
        # df = self.add_mean_sm_to_df(df)
        # df = self.add_mean_sm_anomaly_to_df(df)
        # df = self.add_bin_class_to_df(df,bin_var='min_precip_in_drought_range',n=10)
        # df = self.add_bin_class_to_df(df,bin_var='max_vpd_in_drought_range',n=10)
        # df = self.add_AI_index_to_df(df)
        # df = self.precip_vpd_dominate(df)
        # df =
        # df = self.__rename_drought_type(df)
        # df = self.__rename_dataframe_columns(df)
        # df = self.add_initial_supsequential_delta(df)
        df = self.add_zr_in_df(df)
        df = self.add_Rplant_in_df(df)
        # df
        # exit()
        T.save_df(df,self.dff)
        self.__df_to_excel(df,self.dff,random=True)
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

    def __df_to_excel(self,df,dff,n=1000,random=False):
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass

    def __divide_bins(self,arr,min_v=None,max_v=None,step=None,n=None,round_=2,include_external=False):
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

    def drop_duplicated_sample(self,df):
        df_drop_dup = df.drop_duplicates(subset=['pix','carbon_loss','recovery_date_range'])
        return df_drop_dup
        # df_drop_dup.to_excel(self.this_class_arr + 'drop_dup.xlsx')
        pass


    def __load_spei_events(self):
        # single_f = Main_flow_Carbon_loss().this_class_arr + 'gen_recovery_time_legacy_single_events/recovery_time_legacy.pkl'
        repetitive_f = Main_flow_Carbon_loss().this_class_arr + 'gen_recovery_time_legacy_repetitive_events/recovery_time_legacy.pkl'
        # single_events_dic = T.load_dict_from_binary(single_f)
        repetitive_events_dic = T.load_dict_from_binary(repetitive_f)

        return repetitive_events_dic

        pass
    def __load_precip_events(self):
        # single_f = Main_flow_Carbon_loss().this_class_arr + 'gen_recovery_time_legacy_single_events/recovery_time_legacy.pkl'
        repetitive_f = Main_flow_Carbon_loss().this_class_arr + 'gen_recovery_time_legacy_repetitive_events_precip_auto/recovery_time_legacy.pkl'
        # single_events_dic = T.load_dict_from_binary(single_f)
        repetitive_events_dic = T.load_dict_from_binary(repetitive_f)

        return repetitive_events_dic

        pass

    def __load_vpd_events(self):
        # single_f = Main_flow_Carbon_loss().this_class_arr + 'gen_recovery_time_legacy_single_events/recovery_time_legacy.pkl'
        repetitive_f = Main_flow_Carbon_loss().this_class_arr + 'gen_recovery_time_legacy_repetitive_events_VPD_auto/recovery_time_legacy.pkl'
        # single_events_dic = T.load_dict_from_binary(single_f)
        repetitive_events_dic = T.load_dict_from_binary(repetitive_f)

        return repetitive_events_dic

        pass

    def __rename_dataframe_columns(self,df):
        new_name_dic = {
            'water_balance':'Aridity_Index',
            # 'CSIF anomaly loss':'CSIF_anomaly_loss'
        }
        df = pd.DataFrame(df)
        df = df.rename(columns=new_name_dic)

        return df

    def __rename_drought_type(self,df):
        drought_type = df['drought_type']
        drought_type_new = []
        for d in drought_type:
            d = str(d)
            dnew = d.replace('repetitive','repeatedly')
            drought_type_new.append(dnew)
        df['drought_type'] = drought_type_new

        return df

    def Carbon_loss_to_df(self,df):

        repetitive_events_dic_spei12 = self.__load_spei_events()
        # repetitive_events_dic_precip = self.__load_precip_events()
        # repetitive_events_dic_vpd = self.__load_vpd_events()

        pix_list = []
        recovery_time_list = []
        drought_event_date_range_list = []
        recovery_date_range_list = []
        legacy_list = []
        drought_type = []


        for pix in tqdm(repetitive_events_dic_spei12,desc='spei12'):
            events = repetitive_events_dic_spei12[pix]
            if len(events) == 0:
                continue
            for repetetive_event in events:
                initial_event = repetetive_event[0]

                initial_recovery_time = initial_event['recovery_time']
                initial_drought_event_date_range = initial_event['drought_event_date_range']
                initial_recovery_date_range = initial_event['recovery_date_range']
                initial_legacy = initial_event['carbon_loss']
                initial_drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(initial_drought_event_date_range)
                initial_recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(initial_recovery_date_range)

                pix_list.append(pix)
                recovery_time_list.append(initial_recovery_time)
                drought_event_date_range_list.append(tuple(initial_drought_event_date_range))
                recovery_date_range_list.append(tuple(initial_recovery_date_range))
                legacy_list.append(initial_legacy)
                drought_type.append('repetitive_initial_spei12')

                subsequential_event = repetetive_event[1]
                subsequential_recovery_time = subsequential_event['recovery_time']
                subsequential_drought_event_date_range = subsequential_event['drought_event_date_range']
                subsequential_recovery_date_range = subsequential_event['recovery_date_range']
                subsequential_legacy = subsequential_event['carbon_loss']
                subsequential_drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(
                    subsequential_drought_event_date_range)
                subsequential_recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(subsequential_recovery_date_range)

                pix_list.append(pix)
                recovery_time_list.append(subsequential_recovery_time)
                drought_event_date_range_list.append(tuple(subsequential_drought_event_date_range))
                recovery_date_range_list.append(tuple(subsequential_recovery_date_range))
                legacy_list.append(subsequential_legacy)
                drought_type.append('repetitive_subsequential_spei12')

        # for pix in tqdm(repetitive_events_dic_precip,desc='precip'):
        #     events = repetitive_events_dic_precip[pix]
        #     if len(events) == 0:
        #         continue
        #     for repetetive_event in events:
        #         initial_event = repetetive_event[0]
        #
        #         initial_recovery_time = initial_event['recovery_time']
        #         initial_drought_event_date_range = initial_event['drought_event_date_range']
        #         initial_recovery_date_range = initial_event['recovery_date_range']
        #         initial_legacy = initial_event['carbon_loss']
        #         initial_drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(initial_drought_event_date_range)
        #         initial_recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(initial_recovery_date_range)
        #
        #         pix_list.append(pix)
        #         recovery_time_list.append(initial_recovery_time)
        #         drought_event_date_range_list.append(tuple(initial_drought_event_date_range))
        #         recovery_date_range_list.append(tuple(initial_recovery_date_range))
        #         legacy_list.append(initial_legacy)
        #         drought_type.append('repetitive_initial_precip')
        #
        #         subsequential_event = repetetive_event[1]
        #         subsequential_recovery_time = subsequential_event['recovery_time']
        #         subsequential_drought_event_date_range = subsequential_event['drought_event_date_range']
        #         subsequential_recovery_date_range = subsequential_event['recovery_date_range']
        #         subsequential_legacy = subsequential_event['carbon_loss']
        #         subsequential_drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(
        #             subsequential_drought_event_date_range)
        #         subsequential_recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(subsequential_recovery_date_range)
        #
        #         pix_list.append(pix)
        #         recovery_time_list.append(subsequential_recovery_time)
        #         drought_event_date_range_list.append(tuple(subsequential_drought_event_date_range))
        #         recovery_date_range_list.append(tuple(subsequential_recovery_date_range))
        #         legacy_list.append(subsequential_legacy)
        #         drought_type.append('repetitive_subsequential_precip')
        #
        # for pix in tqdm(repetitive_events_dic_vpd,desc='vpd'):
        #     events = repetitive_events_dic_vpd[pix]
        #     if len(events) == 0:
        #         continue
        #     for repetetive_event in events:
        #         initial_event = repetetive_event[0]
        #
        #         initial_recovery_time = initial_event['recovery_time']
        #         initial_drought_event_date_range = initial_event['drought_event_date_range']
        #         initial_recovery_date_range = initial_event['recovery_date_range']
        #         initial_legacy = initial_event['carbon_loss']
        #         initial_drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(initial_drought_event_date_range)
        #         initial_recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(initial_recovery_date_range)
        #
        #         pix_list.append(pix)
        #         recovery_time_list.append(initial_recovery_time)
        #         drought_event_date_range_list.append(tuple(initial_drought_event_date_range))
        #         recovery_date_range_list.append(tuple(initial_recovery_date_range))
        #         legacy_list.append(initial_legacy)
        #         drought_type.append('repetitive_initial_vpd')
        #
        #         subsequential_event = repetetive_event[1]
        #         subsequential_recovery_time = subsequential_event['recovery_time']
        #         subsequential_drought_event_date_range = subsequential_event['drought_event_date_range']
        #         subsequential_recovery_date_range = subsequential_event['recovery_date_range']
        #         subsequential_legacy = subsequential_event['carbon_loss']
        #         subsequential_drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(
        #             subsequential_drought_event_date_range)
        #         subsequential_recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(subsequential_recovery_date_range)
        #
        #         pix_list.append(pix)
        #         recovery_time_list.append(subsequential_recovery_time)
        #         drought_event_date_range_list.append(tuple(subsequential_drought_event_date_range))
        #         recovery_date_range_list.append(tuple(subsequential_recovery_date_range))
        #         legacy_list.append(subsequential_legacy)
        #         drought_type.append('repetitive_subsequential_vpd')


        df['pix'] = pix_list
        df['drought_type'] = drought_type
        df['drought_event_date_range'] = drought_event_date_range_list
        df['recovery_date_range'] = recovery_date_range_list
        df['recovery_time'] = recovery_time_list
        df['carbon_loss'] = legacy_list
        # print(df)
        # exit()
        return df
        pass

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

    def add_landcover_to_df(self,df):
        dic_f = data_root + 'landcover/gen_spatial_dic.npy'
        dic = T.load_npy(dic_f)
        lc_type_dic = {
            1:'EBF',
            2:'DBF',
            3:'DBF',
            4:'ENF',
            5:'DNF',
        }

        forest_type_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = dic[pix]
            forest_type = lc_type_dic[val]
            forest_type_list.append(forest_type)

        df['lc'] = forest_type_list
        return df
        pass

    def landcover_compose(self,df):
        lc_type_dic = {
            'EBF':'Broadleaf',
            'DBF':'Broadleaf',
            'ENF':'Needleleaf',
            'DNF':'Needleleaf',
        }

        lc_broad_needle_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            lc = row.lc
            lc_broad_needle = lc_type_dic[lc]
            lc_broad_needle_list.append(lc_broad_needle)

        df['lc_broad_needle'] = lc_broad_needle_list
        return df

    def add_lon_lat_to_df(self,df):
        # DIC_and_TIF().spatial_tif_to_lon_lat_dic()
        lon_lat_dic = DIC_and_TIF(Global_vars().tif_template_7200_3600).spatial_tif_to_lon_lat_dic()
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

    def add_min_precip_anomaly_to_df(self,df):

        fdir = data_root + 'Precip_terra/per_pix_anomaly/'
        dic = T.load_npy_dir(fdir)
        min_precip_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            precip = dic[pix]
            min_precip_indx = T.pick_min_indx_from_1darray(precip,drought_event_date_range)
            min_precip_v = precip[min_precip_indx]
            # print(min_precip_v)
            # pause()
            min_precip_list.append(min_precip_v)

        df['min_precip_anomaly_in_drought_range'] = min_precip_list
        return df
        pass

    def add_mean_precip_anomaly_to_df(self,df):

        fdir = data_root + 'Precip_terra/per_pix_anomaly/'
        dic = T.load_npy_dir(fdir)
        min_precip_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            precip = dic[pix]
            picked_val = T.pick_vals_from_1darray(precip,drought_event_date_range)
            mean_precip = np.mean(picked_val)
            min_precip_list.append(mean_precip)

        df['mean_precip_anomaly_in_drought_range'] = min_precip_list
        return df
        pass

    def add_mean_precip_to_df(self,df):

        fdir = data_root + 'Precip_terra/per_pix/'
        dic = T.load_npy_dir(fdir)
        min_precip_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            precip = dic[pix]
            picked_val = T.pick_vals_from_1darray(precip,drought_event_date_range)
            mean_precip = np.mean(picked_val)
            min_precip_list.append(mean_precip)

        df['mean_precip_in_drought_range'] = min_precip_list
        return df

    def add_mean_vpd_to_df(self,df):

        fdir = data_root + 'VPD/per_pix/'
        dic = T.load_npy_dir(fdir)
        max_vpd_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            vpd = dic[pix]
            picked_val = T.pick_vals_from_1darray(vpd, drought_event_date_range)
            mean_val = np.mean(picked_val)
            max_vpd_list.append(mean_val)

        df['mean_vpd_in_drought_range'] = max_vpd_list
        return df
        pass

    def add_mean_vpd_anomaly_to_df(self,df):

        fdir = data_root + 'VPD/per_pix_anomaly/'
        dic = T.load_npy_dir(fdir)
        max_vpd_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            vpd = dic[pix]
            picked_val = T.pick_vals_from_1darray(vpd, drought_event_date_range)
            mean_val = np.mean(picked_val)
            max_vpd_list.append(mean_val)

        df['mean_vpd_anomaly_in_drought_range'] = max_vpd_list
        return df
        pass

    def add_mean_sm_to_df(self,df):

        fdir = data_root + 'terraclimate/soil/per_pix/'
        dic = T.load_npy_dir(fdir)
        max_vpd_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            vpd = dic[pix]
            picked_val = T.pick_vals_from_1darray(vpd, drought_event_date_range)
            mean_val = np.mean(picked_val)
            max_vpd_list.append(mean_val)

        df['mean_soil_in_drought_range'] = max_vpd_list
        return df
        pass


    def add_mean_sm_anomaly_to_df(self,df):

        fdir = data_root + 'terraclimate/soil/per_pix_anomaly/'
        dic = T.load_npy_dir(fdir)
        max_vpd_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            vpd = dic[pix]
            picked_val = T.pick_vals_from_1darray(vpd, drought_event_date_range)
            mean_val = np.mean(picked_val)
            max_vpd_list.append(mean_val)

        df['mean_soil_anomaly_in_drought_range'] = max_vpd_list
        return df
        pass

    def add_max_vpd_anomaly_to_df(self,df):

        fdir = data_root + 'VPD/per_pix_anomaly/'
        dic = T.load_npy_dir(fdir)
        max_vpd_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            vpd = dic[pix]
            max_vpd_indx = T.pick_max_indx_from_1darray(vpd,drought_event_date_range)
            max_vpd_v = vpd[max_vpd_indx]
            # print(max_vpd_v)
            # pause()
            max_vpd_list.append(max_vpd_v)

        df['max_vpd_anomaly_in_drought_range'] = max_vpd_list
        return df
        pass


    def add_max_vpd_to_df(self,df):

        fdir = data_root + 'VPD/per_pix/'
        dic = T.load_npy_dir(fdir)
        max_vpd_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            vpd = dic[pix]
            max_vpd_indx = T.pick_max_indx_from_1darray(vpd,drought_event_date_range)
            max_vpd_v = vpd[max_vpd_indx]
            # print(max_vpd_v)
            # pause()
            max_vpd_list.append(max_vpd_v)

        df['max_vpd_in_drought_range'] = max_vpd_list
        return df
        pass


    def add_min_precip_to_df(self,df):

        fdir = data_root + 'Precip_terra/per_pix/'
        dic = T.load_npy_dir(fdir)
        min_precip_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            drought_event_date_range = row.drought_event_date_range
            precip = dic[pix]
            min_precip_indx = T.pick_min_indx_from_1darray(precip,drought_event_date_range)
            min_precip_v = precip[min_precip_indx]
            # print(min_precip_v)
            # pause()
            min_precip_list.append(min_precip_v)

        df['min_precip_in_drought_range'] = min_precip_list
        return df
        pass


    def add_bin_class_to_df(self,df,bin_var,n=20):

        # bin_var = 'min_precip_in_drought_range'
        min_precip_in_drought_range = df[bin_var]
        d, d_str = self.__divide_bins(min_precip_in_drought_range,min_v=-2.5,max_v=2.5,
                                      n=n,round_=2,include_external=True)
        bin_class_list = []
        for _,row in tqdm(df.iterrows(),total=len(df)):
            bin_val = row[bin_var]
            bin_class = np.nan
            lc_broad_needle = row['lc_broad_needle']
            for j in range(len(d)):
                if j + 1 >= len(d):
                    continue
                if bin_val >= d[j] and bin_val < d[j + 1]:
                    bin_class = d[j]
                    # bin_class = lc_broad_needle + '_' + d_str[j]
            if bin_class == np.nan:
                print(bin_val)
                print(d)
            bin_class_list.append(bin_class)
        df[bin_var + '_bin_class'] = bin_class_list
        return df

    def add_AI_index_to_df(self,df):
        tif = data_root + 'Water_balance_005/AI.tif'
        dic = DIC_and_TIF(Global_vars().tif_template_7200_3600).spatial_tif_to_dic(tif)
        wb_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = dic[pix]/10000.
            wb_list.append(val)
        df['Aridity_Index'] = wb_list
        return df

        pass

    def precip_vpd_dominate(self,df):
        dominate_list = []

        var_demand = 'max_vpd_anomaly_in_drought_range'
        var_supply = 'min_precip_anomaly_in_drought_range'
        for i, row in tqdm(df.iterrows(), total=len(df)):
            val_demand = row[var_demand]
            val_supply = row[var_supply]

            val_demand = np.array(val_demand)
            val_supply = np.array(val_supply)

            val_supply = -val_supply
            if val_demand > val_supply:
                dominate = 'demand'
            else:
                dominate = 'supply'
            dominate_list.append(dominate)
        df['dominate'] = dominate_list
        return df

    def cal_rt_rs_rc(self, df):
        # SIF_dir = data_root + 'CSIF005/per_pix_anomaly_detrend/'
        SIF_dir = data_root + 'CSIF005/per_pix/'
        gs_range = Global_vars().gs_mons()
        n = 2
        sif_dic = T.load_npy_dir(SIF_dir)
        rt_list = []
        rc_list = []
        rs_list = []
        gs_map_dic = Global_vars().map_time_series_indx_and_gs_series_indx()
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if not pix in sif_dic:
                rt_list.append(np.nan)
                rc_list.append(np.nan)
                rs_list.append(np.nan)
                continue
            ndvi = sif_dic[pix]
            ndvi_gs = []
            for i,val in enumerate(ndvi):
                mon = i % 12 + 1
                if mon in gs_range:
                    ndvi_gs.append(val)
            drought_event_date_range = row.drought_event_date_range
            drought_event_date_range_gs = []
            for i in drought_event_date_range:
                if not i in gs_map_dic:
                    continue
                gs_indx = gs_map_dic[i]
                drought_event_date_range_gs.append(gs_indx)
            # print(drought_event_date_range_gs)
            # exit()
            if len(drought_event_date_range_gs) == 0:
                rt_list.append(np.nan)
                rc_list.append(np.nan)
                rs_list.append(np.nan)
                continue
            drought_year = drought_event_date_range_gs[0] // len(gs_range)
            prev_year_range = list(range(drought_year - n,drought_year))
            if drought_year + 3 >= len(ndvi) / 12:
                rt_list.append(np.nan)
                rc_list.append(np.nan)
                rs_list.append(np.nan)
                continue
            post_year_range = list(range(drought_year + 1,drought_year + 1 + n))
            # ndvi_post_indx = range(drought_event_date_range[-1] + 0, drought_event_date_range[-1] + n)
            # ndvi_prev_indx = range(drought_event_date_range[0] - n, drought_event_date_range[0])
            # ndvi_duration_indx = drought_event_date_range
            ndvi_reshape = np.reshape(ndvi,(-1,12))
            ndvi_reshape = ndvi_reshape.T
            gs_range_indx = np.array(gs_range) - 1
            ndvi_reshape_gs = T.pick_vals_from_1darray(ndvi_reshape,gs_range_indx)
            ndvi_reshape_gs = ndvi_reshape_gs.T

            drought_year_mean = np.mean(T.pick_vals_from_1darray(ndvi_reshape_gs,[drought_year]))
            prev_year_mean = np.mean(T.pick_vals_from_1darray(ndvi_reshape_gs,prev_year_range))
            post_year_mean = np.mean(T.pick_vals_from_1darray(ndvi_reshape_gs,post_year_range))

            rt = drought_year_mean / prev_year_mean
            rc = post_year_mean / drought_year_mean
            rs = post_year_mean / prev_year_mean
            rt_list.append(rt)
            rc_list.append(rc)
            rs_list.append(rs)
        # 'Resistance (Rt), Recovery (Rc), Resilience (Rs)'
        df['Resilience_rs'] = rs_list
        df['Resistance_rt'] = rt_list
        df['Recovery_rc'] = rc_list
        return df

    def minus_carbon_loss(self,df):
        carbon_loss = df['carbon_loss']
        carbon_loss = np.array(carbon_loss)
        carbon_loss = -carbon_loss
        df['CSIF_anomaly_loss'] = carbon_loss
        return df



    def add_initial_supsequential_delta(self,df):
        tif = Tif().this_class_tif + 'delta/subseq_init_delta.tif'
        dic = DIC_and_TIF(Global_vars().tif_template_7200_3600).spatial_tif_to_dic(tif)
        delta_list = []

        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = dic[pix]
            delta_list.append(val)

        df['subseq-init_csif_anomaly_loss'] = delta_list


        return df

    def add_zr_in_df(self,df):
        tif = data_root + 'plant-strategies/Zr_005_unify.tif'
        dic = DIC_and_TIF(Global_vars().tif_template_7200_3600).spatial_tif_to_dic(tif)
        val_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = dic[pix]
            val_list.append(val)
        df['zr'] = val_list

        return df
    def add_Rplant_in_df(self,df):
        tif = data_root + 'plant-strategies/Rplant_005_unify.tif'
        dic = DIC_and_TIF(Global_vars().tif_template_7200_3600).spatial_tif_to_dic(tif)
        val_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = dic[pix]
            val_list.append(val)
        df['Rplant'] = val_list

        return df

# class Main_flow_Dataframe_NDVI_SPEI_legacy_threshold:
#
#     def __init__(self,threshold):
#         self.this_class_arr = results_root_main_flow + 'arr/Main_flow_Dataframe_NDVI_SPEI_legacy/'
#         Tools().mk_dir(self.this_class_arr, force=True)
#         self.dff = self.this_class_arr + 'data_frame_{}.df'.format(threshold)
#         self.threshold = threshold
#
#     def run(self):
#         # 0 generate a void dataframe
#         df = self.__gen_df_init()
#         # self._check_spatial(df)
#         # exit()
#         # 1 add drought event and delta legacy into df
#         df = self.Carbon_loss_to_df(df,self.threshold)
#         # 2 add landcover to df
#         df = self.add_landcover_to_df(df)
#         df = self.landcover_compose(df)
#         T.save_df(df,self.dff)
#         self.__df_to_excel(df,self.dff,random=True)
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
#             return df
#         else:
#             df,dff = self.__load_df()
#             return df
#             # raise Warning('{} is already existed'.format(self.dff))
#
#     def __df_to_excel(self,df,dff,n=1000,random=False):
#         if n == None:
#             df.to_excel('{}.xlsx'.format(dff))
#         else:
#             if random:
#                 df = df.sample(n=n, random_state=1)
#                 df.to_excel('{}.xlsx'.format(dff))
#             else:
#                 df = df.head(n)
#                 df.to_excel('{}.xlsx'.format(dff))
#
#         pass
#
#
#
#     def drop_duplicated_sample(self,df):
#         df_drop_dup = df.drop_duplicates(subset=['pix','carbon_loss','recovery_date_range'])
#         return df_drop_dup
#         # df_drop_dup.to_excel(self.this_class_arr + 'drop_dup.xlsx')
#         pass
#
#     def Carbon_loss_to_df(self,df,threshold):
#         single_f = Main_flow_Carbon_loss().this_class_arr + 'gen_recovery_time_legacy_single_events_{}/recovery_time_legacy.pkl'.format(threshold)
#         repetitive_f = Main_flow_Carbon_loss().this_class_arr + 'gen_recovery_time_legacy_repetitive_events_{}/recovery_time_legacy.pkl'.format(threshold)
#         single_events_dic = T.load_dict_from_binary(single_f)
#         repetitive_events_dic = T.load_dict_from_binary(repetitive_f)
#         # print(events_dic)
#         # exit()
#         pix_list = []
#         recovery_time_list = []
#         drought_event_date_range_list = []
#         recovery_date_range_list = []
#         legacy_list = []
#         drought_type = []
#
#         for pix in tqdm(single_events_dic,desc='single events carbon loss'):
#             events = single_events_dic[pix]
#             for event in events:
#                 recovery_time = event['recovery_time']
#                 drought_event_date_range = event['drought_event_date_range']
#                 recovery_date_range = event['recovery_date_range']
#                 legacy = event['carbon_loss']
#
#                 drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(drought_event_date_range)
#                 recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(recovery_date_range)
#
#                 pix_list.append(pix)
#                 recovery_time_list.append(recovery_time)
#                 drought_event_date_range_list.append(tuple(drought_event_date_range))
#                 recovery_date_range_list.append(tuple(recovery_date_range))
#                 legacy_list.append(legacy)
#                 drought_type.append('single')
#
#         for pix in tqdm(repetitive_events_dic,desc='repetitive events carbon loss'):
#             events = repetitive_events_dic[pix]
#             if len(events) == 0:
#                 continue
#             for repetetive_event in events:
#                 initial_event = repetetive_event[0]
#
#                 initial_recovery_time = initial_event['recovery_time']
#                 initial_drought_event_date_range = initial_event['drought_event_date_range']
#                 initial_recovery_date_range = initial_event['recovery_date_range']
#                 initial_legacy = initial_event['carbon_loss']
#                 initial_drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(initial_drought_event_date_range)
#                 initial_recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(initial_recovery_date_range)
#
#                 pix_list.append(pix)
#                 recovery_time_list.append(initial_recovery_time)
#                 drought_event_date_range_list.append(tuple(initial_drought_event_date_range))
#                 recovery_date_range_list.append(tuple(initial_recovery_date_range))
#                 legacy_list.append(initial_legacy)
#                 drought_type.append('repetitive_initial')
#
#                 subsequential_event = repetetive_event[1]
#                 subsequential_recovery_time = subsequential_event['recovery_time']
#                 subsequential_drought_event_date_range = subsequential_event['drought_event_date_range']
#                 subsequential_recovery_date_range = subsequential_event['recovery_date_range']
#                 subsequential_legacy = subsequential_event['carbon_loss']
#                 subsequential_drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(
#                     subsequential_drought_event_date_range)
#                 subsequential_recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(subsequential_recovery_date_range)
#
#                 pix_list.append(pix)
#                 recovery_time_list.append(subsequential_recovery_time)
#                 drought_event_date_range_list.append(tuple(subsequential_drought_event_date_range))
#                 recovery_date_range_list.append(tuple(subsequential_recovery_date_range))
#                 legacy_list.append(subsequential_legacy)
#                 drought_type.append('repetitive_subsequential')
#
#
#         df['pix'] = pix_list
#         df['drought_type'] = drought_type
#         df['drought_event_date_range'] = drought_event_date_range_list
#         df['recovery_date_range'] = recovery_date_range_list
#         df['recovery_time'] = recovery_time_list
#         df['carbon_loss'] = legacy_list
#         # print(df)
#         # exit()
#         return df
#         pass
#
#     def add_isohydricity_to_df(self,df):
#         tif = data_root + 'Isohydricity/tif_all_year/ISO_Hydricity.tif'
#         dic = DIC_and_TIF().spatial_tif_to_dic(tif)
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
#     def add_landcover_to_df(self,df):
#         dic_f = data_root + 'landcover/gen_spatial_dic.npy'
#         dic = T.load_npy(dic_f)
#         lc_type_dic = {
#             1:'EBF',
#             2:'DBF',
#             3:'DBF',
#             4:'ENF',
#             5:'DNF',
#         }
#
#         forest_type_list = []
#         for i,row in tqdm(df.iterrows(),total=len(df)):
#             pix = row.pix
#             val = dic[pix]
#             forest_type = lc_type_dic[val]
#             forest_type_list.append(forest_type)
#
#         df['lc'] = forest_type_list
#         return df
#         pass
#
#     def landcover_compose(self,df):
#         lc_type_dic = {
#             'EBF':'Broadleaf',
#             'DBF':'Broadleaf',
#             'ENF':'Needleleaf',
#             'DNF':'Needleleaf',
#         }
#
#         lc_broad_needle_list = []
#         for i,row in tqdm(df.iterrows(),total=len(df)):
#             lc = row.lc
#             lc_broad_needle = lc_type_dic[lc]
#             lc_broad_needle_list.append(lc_broad_needle)
#
#         df['lc_broad_needle'] = lc_broad_needle_list
#         return df
#
#
#
#     def add_TWS_to_df(self,df):
#
#         fdir = data_root + 'TWS/GRACE/per_pix/'
#         tws_dic = T.load_npy_dir(fdir)
#         tws_list = []
#         for i,row in tqdm(df.iterrows(),total=len(df)):
#             recovery_date_range = row['recovery_date_range']
#             pix = row.pix
#             if not pix in tws_dic:
#                 tws_list.append(np.nan)
#                 continue
#             vals = tws_dic[pix]
#             picked_val = T.pick_vals_from_1darray(vals,recovery_date_range)
#             picked_val[picked_val<-999]=np.nan
#             mean = np.nanmean(picked_val)
#             tws_list.append(mean)
#         df['TWS_recovery_period'] = tws_list
#
#         # exit()
#         return df

class Tif:


    def __init__(self):
        self.this_class_tif = results_root_main_flow + 'tif/Tif/'
        Tools().mk_dir(self.this_class_tif, force=True)

    def run(self):

        # self.carbon_loss_single_events()
        # self.carbon_loss_repetitive_events_initial()
        # self.carbon_loss_repetitive_events_subsequential()
        # for var in ['Resilience_rs','Resistance_rt','Recovery_rc',]:
        #     print(var)
        #     self.rc_rs_rt(var)
        # self.drought_start()
        self.delta()
        pass


    def load_df(self):
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)

        return df,dff

        pass

    def carbon_loss_single_events(self):
        df ,dff = self.load_df()
        df = df[df['drought_type']=='single']
        spatial_dic = DIC_and_TIF(Global_vars().tif_template_7200_3600).void_spatial_dic()
        outdir = self.this_class_tif + 'carbon_loss/'
        T.mk_dir(outdir)
        outf = outdir + 'carbon_loss_single_events.tif'
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = row['carbon_loss']
            spatial_dic[pix].append(val)

        mean_arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr_mean(spatial_dic)
        DIC_and_TIF(Global_vars().tif_template_7200_3600).arr_to_tif(mean_arr,outf)


        pass
    def carbon_loss_repetitive_events_initial(self):
        df ,dff = self.load_df()
        df = df[df['drought_type']=='repetitive_initial']
        spatial_dic = DIC_and_TIF(Global_vars().tif_template_7200_3600).void_spatial_dic()
        outdir = self.this_class_tif + 'carbon_loss/'
        T.mk_dir(outdir)
        outf = outdir + 'carbon_loss_repetitive_events_initial.tif'
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = row['carbon_loss']
            spatial_dic[pix].append(val)

        mean_arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr_mean(spatial_dic)
        DIC_and_TIF(Global_vars().tif_template_7200_3600).arr_to_tif(mean_arr,outf)


        pass


    def carbon_loss_repetitive_events_subsequential(self):
        df ,dff = self.load_df()
        df = df[df['drought_type']=='repetitive_subsequential']
        spatial_dic = DIC_and_TIF(Global_vars().tif_template_7200_3600).void_spatial_dic()
        outdir = self.this_class_tif + 'carbon_loss/'
        T.mk_dir(outdir)
        outf = outdir + 'carbon_loss_repetitive_events_subsequential.tif'
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = row['carbon_loss']
            spatial_dic[pix].append(val)

        mean_arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr_mean(spatial_dic)
        DIC_and_TIF(Global_vars().tif_template_7200_3600).arr_to_tif(mean_arr,outf)


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

    def rc_rs_rt(self,var):
        outtifdir = self.this_class_tif + 'rc_rs_rt/'
        T.mk_dir(outtifdir)
        df,dff = self.load_df()
        print('loading df done')
        spatial_dic = DIC_and_TIF(Global_vars().tif_template_7200_3600).void_spatial_dic()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = row[var]
            spatial_dic[pix].append(val)
        arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr_mean(spatial_dic)
        DIC_and_TIF(Global_vars().tif_template_7200_3600).arr_to_tif(arr,outtifdir + '{}.tif'.format(var))

    def delta(self):
        outtifdir = self.this_class_tif + 'delta/'
        outf = outtifdir + 'subseq_init_delta.tif'
        T.mk_dir(outtifdir)
        df, dff = self.load_df()
        events_dic = {}
        pix_list = df['pix'].to_list()
        pix_list = set(pix_list)
        for pix in pix_list:
            events_dic[pix] = {}
            events_dic[pix]['repeatedly_initial_spei12'] = []
            events_dic[pix]['repeatedly_subsequential_spei12'] = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            CSIF_anomaly_loss = row.CSIF_anomaly_loss
            drought_type = row.drought_type
            if drought_type == 'repeatedly_initial_spei12':
                events_dic[pix]['repeatedly_initial_spei12'].append(CSIF_anomaly_loss)
            elif drought_type == 'repeatedly_subsequential_spei12':
                events_dic[pix]['repeatedly_subsequential_spei12'].append(CSIF_anomaly_loss)
            else:
                raise UserWarning('drought_type error')

        delta_spatial_dic = {}
        for pix in tqdm(events_dic,desc='cal delta...'):
            events = events_dic[pix]
            repeatedly_initial_spei12 = events['repeatedly_initial_spei12']
            repeatedly_subsequential_spei12 = events['repeatedly_subsequential_spei12']
            init_mean = np.mean(repeatedly_initial_spei12)
            subseq_mean = np.mean(repeatedly_subsequential_spei12)
            delta = subseq_mean - init_mean
            delta_spatial_dic[pix] = delta
        DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_tif(delta_spatial_dic,outf)

class Analysis:

    def __init__(self):
        self.this_class_png = results_root_main_flow + 'png/Analysis/'
        self.this_class_tif = results_root_main_flow + 'tif/Analysis/'
        Tools().mk_dir(self.this_class_png, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass

    def run(self):

        # self.run_Bins_scatter_line()
        # self.dominate_drought()
        # self.scatter_vpd_precip()
        # self.delta()
        # self.matrix()
        # self.bin_scatter()
        self.bin_correlation()


        pass

    def run_Bins_scatter_line(self):
        x_var_list = [
            'min_precip_anomaly_in_drought_range',
            'max_vpd_in_drought_range',
            'Aridity_Index',
            'zr',
            'Rplant',
            'Aridity_Index',
        ]
        y_var_list = [
            'Recovery_rc',
            'Resilience_rs',
            'Resistance_rt',
            'CSIF_anomaly_loss',
        ]
        y1_var_list = [
            'min_precip_anomaly_in_drought_range',
            'max_vpd_in_drought_range',
            'Aridity_Index',
            'zr',
            'Rplant',
        ]
        flag = 0
        for x in x_var_list:
            for y in y_var_list:
                for y1 in y1_var_list:
                    flag += 1
                    print('{} / {}'.format(flag,len(x_var_list)*len(y_var_list)*len(y1_var_list)))
                    self.Bins_scatter_line(x, y, y1)

    def __load_df(self):

        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        return df,dff



    def __divide_bins_equal_interval(self,arr,min_v=None,max_v=None,step=None,n=None,round_=2,include_external=False):
        if min_v == None:
            min_v = np.min(arr)
        if max_v == None:
            max_v = np.max(arr)
        if n == None and step == None:
            raise UserWarning('step or n is required')
        if n == None:
            d = np.arange(start=min_v,step=step,stop=max_v)
            d = list(d)
            d.append(max_v)
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
        for i in range(len(d)):
            if i + 1 >= len(d):
                break
            d_str.append('{}-{}'.format(round(d[i], round_),round(d[i+1], round_)))
        d_mean = []
        for i in range(len(d)):
            if i + 1 >= len(d):
                break
            mean_i = (d[i + 1] + d[i]) / 2.
            d_mean.append(mean_i)
        return d,d_str,d_mean
        pass


    def __divide_bins_quantile(self,arr,min_v=None,max_v=None,n=10,round_=2,bins_mean_type='mean'):
        if min_v == None:
            min_v = np.min(arr)
        if max_v == None:
            max_v = np.max(arr)

        arr = np.array(arr)
        arr[arr<min_v]=np.nan
        arr[arr>max_v]=np.nan
        arr = T.remove_np_nan(arr)

        d_str = []
        d = []
        for i in range(n):
            q_i = float(i)/float(n)
            q = np.quantile(arr,q_i)
            d.append(q)
        d.append(max_v)
        d_str = []
        for i in range(len(d)):
            if i + 1 >= len(d):
                break
            d_str.append('{}-{}'.format(round(d[i], round_), round(d[i + 1], round_)))
        if bins_mean_type == 'mean':
            d_mean = []
            for i in range(len(d)):
                if i + 1 >= len(d):
                    break
                mean_i = (d[i+1] + d[i])/2.
                d_mean.append(mean_i)
        elif bins_mean_type == 'quantile':
            d_mean = []
            for i in range(len(d)):
                if i + 1 >= len(d):
                    break
                mean_i = i
                d_mean.append(mean_i)
        else:
            raise UserWarning('bins_mean_type error')
        return d,d_str,d_mean
        pass

    def __jenks_breaks(self,arr,min_v=0.,max_v=1.,n=10):

        if min_v == None:
            min_v = np.min(arr)
        if max_v == None:
            max_v = np.max(arr)

        arr = np.array(arr)
        arr[arr<min_v]=np.nan
        arr[arr>max_v]=np.nan
        arr = T.remove_np_nan(arr)
        imp_list = list(arr)
        if len(imp_list) > 10000:
            imp_list = random.sample(imp_list, 10000)
        jnb = JenksNaturalBreaks(nb_class=n)
        jnb.fit(imp_list)
        breaks = jnb.inner_breaks_
        breaks = list(breaks)
        breaks.insert(0,min_v)
        breaks.append(max_v)
        # print(breaks)
        # exit()
        breaks_str = [str(round(i,2)) for i in breaks]
        d_mean = []
        d = breaks
        for i in range(len(d)):
            if i + 1 >= len(d):
                break
            mean_i = (d[i + 1] + d[i]) / 2.
            d_mean.append(mean_i)
        return breaks,breaks_str

    def __unique_sort_list(self,inlist):

        inlist = list(inlist)
        inlist = set(inlist)
        inlist = list(inlist)
        inlist.sort()
        return inlist

    def __bin_min_max(self,bin_var):
        if 'precip' in bin_var:
            bin_min = -2
            bin_max = -0.
        elif 'vpd_anomaly' in bin_var:
            bin_min = 0
            bin_max = 2.5
        elif 'vpd_in' in bin_var:
            bin_min = 0
            bin_max = 3.
        elif 'Aridity' in bin_var:
            bin_min = 0
            bin_max = 2.
        elif 'zr' in bin_var:
            bin_min = 0
            bin_max = 5.
        elif 'Rplant' in bin_var:
            bin_min = 0
            bin_max = 0.06
        else:
            raise UserWarning('Y var error')
        pass
        return bin_min,bin_max

    def foo(self):
        df,dff = self.__load_df()

        drought_types_list = ['single','repeatedly_initial','repeatedly_subsequential',]
        bar_list = []
        for drought_type in drought_types_list:
            df_drought_type = df[df['drought_type']==drought_type]
            vals = []
            for i,row in tqdm(df_drought_type.iterrows(),total=len(df_drought_type),desc=drought_type):
                pix = row.pix
                r,c = pix
                if r > 1800:
                    continue
                carbon_loss = row.carbon_loss
                vals.append(carbon_loss)
            bar_list.append(vals)
        plt.boxplot(bar_list,showfliers=False)
        plt.show()


    def foo2(self):
        # threshold = '-2'
        # dff = Main_flow_Dataframe_NDVI_SPEI_legacy_threshold(threshold).dff
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        df = df.dropna()

        df = df[df['drought_type']!='single']
        # df = df[df['drought_type']=='single']
        # df = df[df['drought_type']=='repeatedly_initial']
        # df = df[df['drought_type']=='repeatedly_subsequential']
        # print(len(df))
        # exit()

        carbon_loss = df['carbon_loss']
        carbon_loss_ = -carbon_loss
        df['carbon_loss_'] = carbon_loss_
        # sns.catplot(x='lc_broad_needle',kind="bar",y='carbon_loss_',hue='drought_type',data=df,ci='sd')
        # sns.catplot(x='lc_broad_needle',kind="bar",y='carbon_loss_',hue='drought_type',data=df,ci=60)
        sns.catplot(x='lc_broad_needle',kind="bar",y='carbon_loss_',hue='drought_type',data=df)
        # sns.catplot(x='lc_broad_needle',kind="violin",y='carbon_loss_',hue='drought_type',data=df)
        # sns.catplot(x='lc_broad_needle',kind="swarm",y='carbon_loss_',hue='drought_type',data=df)
        plt.show()
        pass

    def foo3(self):
        # threshold = '-2'
        # dff = Main_flow_Dataframe_NDVI_SPEI_legacy_threshold(threshold).dff
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        df = df.dropna()
        lc_type = 'Needleleaf'

        # lc_type = 'Broadleaf'
        # class_var = 'min_precip_in_drought_range_bin_class'
        class_var = 'max_vpd_in_drought_range_bin_class'
        # df = df[df['lc_broad_needle']=='Needleleaf']
        df = df[df['lc_broad_needle']==lc_type]
        df = df[df['drought_type']!='single']
        # df = df[df['drought_type']=='single']
        # df = df[df['drought_type']=='repeatedly_initial']
        # df = df[df['drought_type']=='repeatedly_subsequential']
        # print(len(df))
        # exit()

        carbon_loss = df['carbon_loss']
        carbon_loss_ = -carbon_loss
        df['carbon_loss_'] = carbon_loss_
        order = df[class_var].tolist()
        order = list(set(order))
        order.sort()
        # print(hue_order)
        # exit()

        # sns.catplot(x='lc_broad_needle',kind="bar",y='carbon_loss_',hue='drought_type',data=df,ci='sd')
        # sns.catplot(x='lc_broad_needle',kind="bar",y='carbon_loss_',hue='drought_type',data=df,ci=60)
        sns.catplot(x=class_var,kind="bar",y='carbon_loss_',hue='drought_type',data=df,ci=None,order=order)
        # sns.catplot(x='lc_broad_needle',kind="violin",y='carbon_loss_',hue='drought_type',data=df)
        # sns.catplot(x='lc_broad_needle',kind="swarm",y='carbon_loss_',hue='drought_type',data=df)
        plt.title(lc_type + '_' + class_var)
        new_ticks = [round(i,2) for i in order]
        plt.xticks(range(len(order)),new_ticks,rotation=90)
        plt.ylim(0,3)
        plt.tight_layout()
        plt.show()
        pass


    def matrix(self):

        n = 100

        # vpd_var = 'mean_vpd_anomaly_in_drought_range'
        # vpd_var = 'max_vpd_anomaly_in_drought_range'
        vpd_var = 'max_vpd_in_drought_range'

        # precip_var = 'mean_precip_anomaly_in_drought_range'
        precip_var = 'min_precip_anomaly_in_drought_range'

        for lc_type in ['Needleleaf','Broadleaf']:
            for drought_type in ['repeatedly_initial_spei12', 'repeatedly_subsequential_spei12']:
                df, dff = self.__load_df()
                df = df.dropna()
                # df = df[df['recovery_time'] < 12]
                df = df[df['lc_broad_needle'] == lc_type]
                # df = df[df['drought_type'] != 'single']
                df = df[df['drought_type'] == drought_type]
                min_precip_in_drought_range = df[precip_var]
                max_vpd_in_drought_range = df[vpd_var]
                # print(df)
                # exit()
                d1,d1_str,d1_mean = self.__divide_bins_equal_interval(min_precip_in_drought_range,min_v=-2.5,max_v=0,n=n)
                d2,d2_str,d2_mean = self.__divide_bins_equal_interval(max_vpd_in_drought_range,min_v=0,max_v=3.5,n=n)
                # print(min_precip_in_drought_range)

                # precip_bins, precip_bins_str = self.__divide_bins_quantile(min_precip_in_drought_range,n=n)
                # vpd_bins,vpd_bins_str = self.__divide_bins_quantile(max_vpd_in_drought_range,n=n,min_v=1.,)
                # vpd_bins,vpd_bins_str = self.__divide_bins_equal_interval(max_vpd_in_drought_range,min_v=-2,max_v=2,n=n)
                # print(d1)
                # print(d2)
                # exit()
                matrix = []
                for i in tqdm(range(len(d1))):
                    if i+1 >= len(d1):
                        continue
                    # print(d1)
                    df_p_bin = df[df[precip_var]>d1[i]]
                    # print(df_p_bin)
                    # exit()
                    df_p_bin = df_p_bin[df_p_bin[precip_var]<d1[i+1]]
                    # print(df_p_bin)
                    # exit()
                    temp = []
                    for j in range(len(d2)):
                        if j + 1 >= len(d2):
                            continue
                        df_vpd_bin = df_p_bin[df_p_bin[vpd_var]>d2[j]]
                        df_vpd_bin = df_vpd_bin[df_vpd_bin[vpd_var]<d2[j+1]]
                        legacy_i = df_vpd_bin['CSIF_anomaly_loss']
                        # print(legacy_i)
                        if len(legacy_i)==0:
                            temp.append(np.nan)
                        else:
                            temp.append(np.nanmean(legacy_i))
                    matrix.append(temp)
                matrix = np.array(matrix)[::-1]
                plt.figure()
                # plt.imshow(matrix,cmap='OrRd')
                plt.imshow(matrix,vmin=1,vmax=6,cmap='jet')
                # plt.xticks(range(len(vpd_bins))[::10],vpd_bins_str[::10])
                precip_bins_str = d1_str[::-1]
                plt.yticks(range(len(d1_str))[::5],precip_bins_str[::5])
                plt.xticks(range(len(d2_str))[::5],d2_str[::5],rotation=90)
                plt.xlabel(vpd_var)
                plt.ylabel(precip_var)
                plt.colorbar()
                plt.title(lc_type+' '+drought_type)
                plt.tight_layout()
                # exit()
        plt.show()


    def decouple_precip_vpd(self):
        '''
        x: vpd
        y: legacy
        line: precip
        '''
        # lc_type = 'Needleleaf'
        # lc_type = 'Broadleaf'
        # drought_type = 'repeatedly_initial'
        # drought_type = 'repeatedly_subsequential'

        # min_precip_var = 'mean_precip_in_drought_range'
        # min_precip_var = 'mean_soil_in_drought_range'
        # min_precip_var = 'mean_soil_in_drought_range'
        min_precip_var = 'mean_precip_anomaly_in_drought_range'
        # max_vpd_var = 'max_vpd_in_drought_range'
        max_vpd_var = 'max_vpd_in_drought_range'
        # max_vpd_var = 'mean_vpd_anomaly_in_drought_range'
        # max_vpd_var = 'mean_vpd_in_drought_range'


        for lc_type in ['Needleleaf', 'Broadleaf']:
            for drought_type in ['repeatedly_initial', 'repeatedly_subsequential']:
                df, dff = self.__load_df()
                df = df.dropna()
                df = df[df['recovery_time'] < 12]
                df = df[df['lc_broad_needle'] == lc_type]
                # df = df[df['drought_type'] != 'single']
                df = df[df['drought_type'] == drought_type]
                min_precip_in_drought_range = df[min_precip_var]
                max_vpd_in_drought_range = df[max_vpd_var]
                min_precip_in_drought_range = self.__unique_sort_list(min_precip_in_drought_range)
                max_vpd_in_drought_range = self.__unique_sort_list(max_vpd_in_drought_range)
                # precip_bins, precip_bins_str = self.__divide_bins_equal_interval(min_precip_in_drought_range, min_v=-2, max_v=0, n=5)
                # precip_bins, precip_bins_str = self.__divide_bins_equal_interval(min_precip_in_drought_range, min_v=-2.5,max_v=2.6,step=0.5)
                precip_bins, precip_bins_str = self.__divide_bins_equal_interval(min_precip_in_drought_range, min_v=-2., max_v=2, n=6)
                # vpd_bins, vpd_bins_str = self.__divide_bins_equal_interval(max_vpd_in_drought_range, min_v=-2.5,max_v=2.6,step=0.5)
                vpd_bins, vpd_bins_str = self.__jenks_breaks(max_vpd_in_drought_range, min_v=0, max_v=3, n=10)
                # vpd_bins, vpd_bins_str = self.__divide_bins_quantile(max_vpd_in_drought_range, min_v=1, n=10)
                # vpd_bins, vpd_bins_str = self.__divide_bins_quantile(max_vpd_in_drought_range, n=20)
                print('vpd_bins', vpd_bins)
                print('precip_bins', precip_bins)
                plt.figure()
                count_matrix = []
                cmap = sns.color_palette("inferno", n_colors=len(precip_bins))
                for i in tqdm(range(len(precip_bins))):
                    if i + 1 >= len(precip_bins):
                        continue
                    df_p_bin = df[df[min_precip_var] > precip_bins[i]]
                    df_p_bin = df_p_bin[df_p_bin[min_precip_var] < precip_bins[i + 1]]
                    count_matrix_temp = []
                    x = []
                    y = []
                    for j in range(len(vpd_bins)):
                        if j + 1 >= len(vpd_bins):
                            continue
                        df_vpd_bin = df_p_bin[df_p_bin[max_vpd_var] > vpd_bins[j]]
                        df_vpd_bin = df_vpd_bin[df_vpd_bin[max_vpd_var] < vpd_bins[j + 1]]
                        count = len(df_vpd_bin)
                        # if count<=100:
                        #     count_matrix_temp.append(np.nan)
                        #     continue
                        count_matrix_temp.append(count)
                        legacy_i = df_vpd_bin['carbon_loss']
                        legacy_i = -legacy_i
                        mean_legacy = np.mean(legacy_i)
                        x.append(vpd_bins[j])
                        # x.append(j)
                        y.append(mean_legacy)
                        # print(vpd_bins[0])
                        # exit()
                    # print(len(x))
                    # print(x)
                    # print(y)
                    label = min_precip_var
                    plt.plot(x,y,label=label + ' '+precip_bins_str[i],c=cmap[i])
                    plt.scatter(x,y,color=cmap[i])
                    count_matrix.append(count_matrix_temp)

                plt.legend()
                plt.xlabel(max_vpd_var)
                plt.ylabel('legacy')
                plt.ylim(0,5)
                plt.title(lc_type+' '+drought_type)
                # plt.xticks(range(len(x))[::2], vpd_bins_str[::2])
                plt.figure()
                count_matrix = count_matrix
                plt.imshow(count_matrix,norm=LogNorm())
                plt.yticks(range(len(precip_bins))[::2], precip_bins_str[::2])
                plt.xticks(range(len(vpd_bins_str))[::2], vpd_bins_str[::2])
                plt.xlabel(max_vpd_var)
                plt.ylabel(min_precip_var)
                print('np.sum(count_matrix)',np.nansum(count_matrix))
                plt.colorbar()
        plt.show()

        pass


    def tree_types_of_drought_type_overview(self):
        df,dff = self.__load_df()
        # product = 'vpd'
        product = 'precip'
        # product = 'spei12'
        df_vpd_initial = df[df['drought_type']=='repeatedly_initial_{}'.format(product)]
        df_vpd_subseq = df[df['drought_type']=='repeatedly_subsequential_{}'.format(product)]

        df_vpd_initial_needle = df_vpd_initial[df_vpd_initial['lc_broad_needle']=='Needleleaf']
        df_vpd_initial_broad = df_vpd_initial[df_vpd_initial['lc_broad_needle']=='Broadleaf']

        df_vpd_subseq_needle = df_vpd_subseq[df_vpd_subseq['lc_broad_needle'] == 'Needleleaf']
        df_vpd_subseq_broad = df_vpd_subseq[df_vpd_subseq['lc_broad_needle'] == 'Broadleaf']

        carbonloss_df_vpd_initial_needle = df_vpd_initial_needle['carbon_loss'].tolist()
        carbonloss_df_vpd_initial_broad = df_vpd_initial_broad['carbon_loss'].tolist()
        carbonloss_df_vpd_subseq_needle = df_vpd_subseq_needle['carbon_loss'].tolist()
        carbonloss_df_vpd_subseq_broad = df_vpd_subseq_broad['carbon_loss'].tolist()

        boxes = [
            carbonloss_df_vpd_initial_broad,
            carbonloss_df_vpd_subseq_broad,
            carbonloss_df_vpd_initial_needle,
            carbonloss_df_vpd_subseq_needle,
                 ]
        boxes_labels = [
            'carbonloss_df_{}_initial_broad'.format(product),
            'carbonloss_df_{}_subseq_broad'.format(product),
            'carbonloss_df_{}_initial_needle'.format(product),
            'carbonloss_df_{}_subseq_needle'.format(product),
                 ]
        # plt.boxplot(boxes,labels=boxes_labels,showfliers=False)
        # plt.show()
        mean_list = []
        error_list = []
        for b in boxes:
            mean = np.mean(b)
            mean = -mean
            error = np.std(b)/2.
            error_list.append(error)
            mean_list.append(mean)
        # print(stats.f_oneway(carbonloss_df_vpd_initial_broad,
        #     carbonloss_df_vpd_subseq_broad,))
        plt.bar(boxes_labels,mean_list,yerr=error_list)
        plt.show()

        # print(len(df_vpd_initial))

        # for i,row in tqdm(df.iterrows(),total=len(df)):
        #     pass

        pass

    def dominate_drought(self):

        df,dff = self.__load_df()
        hue = 'dominate'
        drought_type = 'repeatedly_subsequential_spei12'
        # drought_type = 'repeatedly_initial_spei12'
        # hue = 'drought_type'
        df = Global_vars().clean_df(df)
        df = df[df['drought_type']==drought_type]
        CSIF_anomaly_loss = df['CSIF_anomaly_loss']
        df['CSIF_anomaly_loss'] = CSIF_anomaly_loss
        # dominate
        # lc_broad_needle
        # drought_type
        g = sns.catplot(
            data=df, kind="bar",
            x="lc_broad_needle", y="CSIF_anomaly_loss",hue=hue, hue_order=['supply','demand'],
            alpha=0.6,palette={"supply": "b", "demand": ".85"},)
        plt.ylim(1.5,3.5)

        # sns.violinplot(data=df, x="lc_broad_needle", y="carbon_loss_", hue=hue,
        #                split=True, inner="quart", linewidth=1,
        #                palette={"supply": "b", "demand": ".85"})
        sns.despine(left=True)
        plt.title(drought_type)
        plt.tight_layout()
        plt.show()


    def __normalize(self,vals):
        min_v = np.min(vals)
        max_v = np.max(vals)
        normalized_list = []
        for i in vals:
            norm = (i - min_v)/max_v
            normalized_list.append(norm)
        return normalized_list

    def Bins_line(self):
        color_list = sns.color_palette('muted')
        color_list_line = sns.color_palette('colorblind')

        # min_precip_var = 'mean_precip_anomaly_in_drought_range'
        # max_vpd_var = 'max_vpd_in_drought_range'
        # var_ = 'max_vpd_in_drought_range'
        var_ = 'water_balance'
        # var_ = 'mean_precip_anomaly_in_drought_range'
        # max_vpd_var = 'mean_vpd_anomaly_in_drought_range'
        # max_vpd_var = 'mean_vpd_in_drought_range'

        color_flag = -1
        for lc_broad_needle in ['Needleleaf','Broadleaf']:
            for drought_type in ['repeatedly_initial_spei12','repeatedly_subsequential_spei12']:
                color_flag += 1
                title = '{}\n{}'.format(lc_broad_needle,drought_type)
                print(title)
                df,dff = self.__load_df()
                df = df[df['lat']>23]
                df = df[df['recovery_time']<10]
                df = df[df['lc_broad_needle']==lc_broad_needle]
                # df = df[df['dominate']=='supply']
                # df = df[df['dominate']=='demand']
                # df = df[df['lc']==lc_broad_needle]
                df = df[df['drought_type']==drought_type]
                vals = df[var_]
                # plt.hist(waterbalance,bins=80)
                # plt.show()
                bins,bins_str,bins_mean = self.__divide_bins_equal_interval(vals,min_v=-0,max_v=3,n=30)
                # bins,bins_str,bins_mean = self.__divide_bins_quantile(waterbalance,min_v=0.,max_v=2,n=100)
                # print(bins)
                # print(bins_str)
                # exit()
                mean_list = []
                events_number = []
                boxes = []
                err_list = []
                for i in tqdm(range(len(bins))):
                    if i + 1 >= len(bins):
                        break
                    df_wb = df[df[var_]>bins[i]]
                    df_wb = df_wb[df_wb[var_]<bins[i+1]]
                    carbonloss = df_wb['carbon_loss_'].tolist()
                    events_number.append(len(carbonloss))
                    bar = np.mean(carbonloss)
                    err = np.std(carbonloss)/4.
                    mean_list.append(bar)
                    err_list.append(err)
                    boxes.append(carbonloss)
                mean_list = np.array(mean_list)
                err_list = np.array(err_list)
                window = 5
                err_list = SMOOTH().smooth_convolve(err_list,window_len=window)
                mean_list_smooth = SMOOTH().smooth_convolve(mean_list,window_len=window)[:-1]
                # mean_list_smooth = mean_list
                plt.plot(bins_mean,mean_list_smooth,label=title,c=color_list_line[color_flag],linewidth=3,zorder=99)
                # plt.scatter(bins_mean,mean_list_smooth,c='w',zorder=100)
                Plot_line().plot_line_with_gradient_error_band(bins_mean,mean_list_smooth,err_list,
                                                               c=color_list[color_flag],color_gradient_n=200)
        plt.xlabel(var_)
        plt.ylabel('CSIF Anomaly')
        plt.legend()
        plt.show()


        pass

    def Bins_scatter_line(self,x_var,y_var,y1_var):
        outpngdir = self.this_class_png + 'Bins_scatter_line_equal_interval/'
        # outpngdir = self.this_class_png + 'Bins_scatter_line_quantile/'

        T.mk_dir(outpngdir)
        # color_list = sns.color_palette('muted')
        color_list = []
        color_list2 = sns.color_palette('Reds_r',n_colors=10)
        # color_list2 = sns.color_palette('YlOrBr_r',n_colors=10)
        color_list1 = sns.color_palette('Greens_r',n_colors=10)
        # color_list_line = sns.color_palette('colorblind')
        color_list_line = []
        color_list.append(color_list1[1])
        color_list.append(color_list1[1])
        color_list.append(color_list2[1])
        color_list.append(color_list2[1])

        color_list_line.append(color_list1[6])
        color_list_line.append(color_list1[3])
        color_list_line.append(color_list2[6])
        color_list_line.append(color_list2[3])

        color_list_line.append(color_list1[6])
        color_list_line.append(color_list1[3])
        color_list_line.append(color_list2[6])
        color_list_line.append(color_list2[3])

        # color_list_line = color_list
        # exit()
        bin_n = 7
        # bin_n = 20
        if 'Re' in y_var:
            y_var_max = 1.3
            y_var_min = 0.7
        else:
            y_var_max = 9999
            y_var_min = -9999
        color_flag = -1
        # plt.figure(figsize=(10,8))
        for drought_type in ['repeatedly_initial_spei12', 'repeatedly_subsequential_spei12']:
            fig,ax0 = plt.subplots(figsize=(8,8))
            ax1 = ax0.twinx()

            for lc_broad_needle in ['Needleleaf','Broadleaf']:
                color_flag += 1
                title = '{}\n{}'.format(lc_broad_needle,drought_type)
                print(title)
                df,dff = self.__load_df()
                df = df[df['lat']>23]
                df = df[df['Rplant']<0.06]
                df = df[df['lc_broad_needle']==lc_broad_needle]
                df = df[df['drought_type']==drought_type]
                # df = df[df['dominate']=='demand']
                # df = df[df['dominate']=='supply']
                df = df[df[y_var]>y_var_min]
                df = df[df[y_var]<y_var_max]
                vals = df[x_var]
                # plt.hist(vals,bins=80)
                # plt.show()
                if 'precip' in x_var:
                    bin_min = -2
                    bin_max = -0.
                elif 'vpd_anomaly' in x_var:
                    bin_min = 0
                    bin_max = 2.5
                elif 'vpd_in' in x_var:
                    bin_min = 0
                    bin_max = 3.
                elif 'Aridity' in x_var:
                    bin_min = 0
                    bin_max = 2.
                elif 'zr' in x_var:
                    bin_min = 0
                    bin_max = 8.
                elif 'Rplant' in x_var:
                    bin_min = 0
                    bin_max = 0.06
                else:
                    raise UserWarning('Y var error')
                print('bin_max', bin_max)
                print('bin_min', bin_min)
                bins,bins_str,bins_mean = self.__divide_bins_equal_interval(vals,min_v=bin_min,max_v=bin_max,n=bin_n,include_external=False)
                # bins,bins_str,bins_mean = self.__divide_bins_quantile(vals,min_v=bin_min,max_v=bin_max,n=bin_n,
                #                                                       bins_mean_type='quantile')
                mean_list = []
                events_number = []
                boxes = []
                err_list = []
                x_err_list = []

                mean1_list = []
                err_list1 = []
                for i in tqdm(range(len(bins))):
                    if i + 1 >= len(bins):
                        break
                    df_wb = df[df[x_var]>bins[i]]
                    df_wb = df_wb[df_wb[x_var]<bins[i+1]]
                    wb = df_wb[x_var]
                    carbonloss = df_wb[y_var].tolist()
                    y1 = df_wb[y1_var].tolist()
                    y1_mean = np.nanmean(y1)
                    y1_err = np.nanstd(y1) / 4.
                    mean1_list.append(y1_mean)
                    err_list1.append(y1_err)
                    events_number.append(len(carbonloss))
                    bar = np.nanmean(carbonloss)
                    err = np.nanstd(carbonloss)/4.
                    xerr = np.nanstd(wb)
                    mean_list.append(bar)
                    err_list.append(err)
                    x_err_list.append(xerr)
                    boxes.append(carbonloss)
                mean_list = np.array(mean_list)
                err_list = np.array(err_list)
                mean1_list = np.array(mean1_list)
                err_list1 = np.array(err_list1)
                window = 5
                # err_list = SMOOTH().smooth_convolve(err_list,window_len=window)
                # mean_list_smooth = SMOOTH().smooth_convolve(mean_list,window_len=window)[:-1]
                mean_list_smooth = mean_list
                # bins_mean = range(len(bins_mean))
                ax0.plot(bins_mean,mean_list_smooth,label=title,color=color_list_line[color_flag],linewidth=3,zorder=99)
                ax0.scatter(bins_mean,mean_list_smooth,color=color_list_line[color_flag],s=80,zorder=100)
                ax0.errorbar(bins_mean, mean_list_smooth, yerr=err_list,
                             capsize=4,ecolor=color_list_line[color_flag],
                             ls='none')
                # Plot_line().plot_line_with_gradient_error_band(bins_mean,mean_list_smooth,err_list,
                #                                                c=color_list[color_flag],color_gradient_n=200)
                ax0.set_xlabel(x_var)
                ax0.set_ylabel(y_var)
                # ax0.set_ylim(0,4)
                # ax_hist = plt.twinx()
                # plt.plot(bins_mean,events_number,label=title+' events number')
                ax1.plot(bins_mean,mean1_list,label=title+' '+y1_var,color=color_list_line[color_flag+2])
                ax1.scatter(bins_mean,mean1_list,color=color_list_line[color_flag+2])
                ax1.set_ylabel(y1_var)
                plt.tight_layout()
                ax0.legend(loc='upper right')
                ax1.legend(loc='lower left')

            # plt.show()

            outf = outpngdir + '{}__{}__{}__{}.png'.format(drought_type,x_var, y_var,y1_var)
            plt.savefig(outf, dpi=300)
            plt.close()
        # plt.figure(figsize=(10,8))
        # for lc_broad_needle in ['Needleleaf','Broadleaf']:
        #     for drought_type in ['repeatedly_initial_spei12','repeatedly_subsequential_spei12']:
        #         color_flag += 1
        #         title = '{}\n{}'.format(lc_broad_needle,drought_type)
        #         print(title)
        #         df,dff = self.__load_df()
        #         # for i in df:
        #             # print(i)
        #         # T.print_head_n(df)
        #         # exit()
        #         df = df[df['lat']>23]
        #         # plt.hist(var_all_vals,bins=100)
        #         # plt.show()
        #         # df = df[df['recovery_time']<10]
        #         df = df[df['lc_broad_needle']==lc_broad_needle]
        #         df = df[df['drought_type']==drought_type]
        #         # df = df[df['dominate']=='demand']
        #         # df = df[df['dominate']=='supply']
        #         df = df[df[y_var]>y_var_min]
        #         df = df[df[y_var]<y_var_max]
        #         vals = df[x_var]
        #         if 'precip' in x_var:
        #             bin_min = -2.
        #             bin_max = -0.
        #         elif 'vpd' in x_var:
        #             bin_min = 0
        #             bin_max = 2.5
        #         elif 'Aridity' in x_var:
        #             bin_min = 0
        #             bin_max = 2.
        #         else:
        #             raise UserWarning('Y var error')
        #         print('bin_max', bin_max)
        #         print('bin_min', bin_min)
        #         bins, bins_str, bins_mean = self.__divide_bins_equal_interval(vals, min_v=bin_min, max_v=bin_max,
        #                                                                       n=bin_n,include_external=False)
        #         events_number = []
        #
        #         for i in tqdm(range(len(bins))):
        #             if i + 1 >= len(bins):
        #                 break
        #             df_wb = df[df[x_var]>bins[i]]
        #             df_wb = df_wb[df_wb[x_var]<bins[i+1]]
        #             carbonloss = df_wb[y_var].tolist()
        #             events_number.append(len(carbonloss))
        #         plt.plot(bins_mean,events_number,label=title+' events number')
        #         plt.scatter(bins_mean,events_number,s=80)
        #         # plt.yscale('log', base=10)
        #         plt.tight_layout()
        #         plt.legend()
        # # plt.show()
        # plt.savefig(outf.replace('.png','')+'_events_number.png', dpi=300)
        # plt.close()




        pass


    def scatter_vpd_precip(self):
        color_list1 = sns.color_palette('Reds_r',n_colors=10)
        color_list2 = sns.color_palette('Greens_r', n_colors=10)
        color_list_line = []

        color_list_line.append(color_list1[6])
        color_list_line.append(color_list1[3])
        color_list_line.append(color_list2[6])
        color_list_line.append(color_list2[3])

        df, dff = self.__load_df()
        df = Global_vars().clean_df(df)
        df = df[df['max_vpd_in_drought_range']<3]
        sns.JointGrid(x='max_vpd_in_drought_range',
                      y='min_precip_anomaly_in_drought_range',
                      data=df )
        plt.show()
        # vpd = df['max_vpd_in_drought_range']
        # pre = df['min_precip_anomaly_in_drought_range']
        # vpd = np.array(vpd)
        # pre = np.array(pre)

        # color_flag = -1
        # for lc_broad_needle in ['Needleleaf','Broadleaf']:
        #     for drought_type in ['repeatedly_initial_spei12','repeatedly_subsequential_spei12']:
        #         title = '{}\n{}'.format(lc_broad_needle,drought_type)
        #         print(title)
        #         color_flag += 1
        #         df,dff = self.__load_df()
        #         df = Global_vars().clean_df(df)
        #         df = df[df['lc_broad_needle']==lc_broad_needle]
        #         df = df[df['drought_type']==drought_type]
        #         vpd = df['max_vpd_in_drought_range']
        #         pre = df['min_precip_anomaly_in_drought_range']
        #         vpd = np.array(vpd)
        #         pre = np.array(pre)
        #
        #         vpd[vpd>3] = np.nan
        #         # vpd[vpd>3] = np.nan
        #
        #         sns.jointplot(x=vpd, y=pre, kind="hex", color=color_list_line[color_flag], dropna=True, )
        plt.show()


    def delta(self):
        # for i,row in tqdm(df.iterrows(),total=len(df)):
        #     delta_loss = row['subseq-init_csif_anomaly_loss']
        x_var_list = [
            'min_precip_anomaly_in_drought_range',
            'max_vpd_in_drought_range',
            'max_vpd_anomaly_in_drought_range',
            'Aridity_Index',
        ]
        # x_var = 'Aridity_Index'
        x_var = x_var_list[1]
        y_var = 'subseq-init_csif_anomaly_loss'
        bin_n = 11
        df, dff = self.__load_df()
        lc_list = df['lc_broad_needle'].tolist()
        lc_list = list(set(lc_list))
        lc_list.sort()
        # for lc_broad_needle in ['Needleleaf','Broadleaf']:
        for lc in lc_list:
            # title = '{}'.format(lc_broad_needle)
            print(lc)
            df,dff = self.__load_df()
            df = df[df['lat']>23]
            # plt.hist(var_all_vals,bins=100)
            # plt.show()
            # df = df[df['recovery_time']<10]
            # df = df[df['lc_broad_needle']==lc_broad_needle]
            df = df[df['lc_broad_needle']==lc]

            # boxes.append(val)
            # df = df[df['dominate']=='demand']
            # df = df[df['dominate']=='supply']
            # df = df[df[y_var]>-10]
            df = df[df[y_var]!=0]
            # df = df[df[y_var]<10]

            x_vals = df[x_var]
            if 'precip' in x_var:
                bin_min = -2.5
                bin_max = -0.
            elif 'vpd' in x_var:
                bin_min = 0
                bin_max = 4
            elif 'Aridity' in x_var:
                bin_min = 0
                bin_max = 2.
            else:
                raise UserWarning('Y var error')
            print('bin_max', bin_max)
            print('bin_min', bin_min)
            # plt.hist(carbonloss_all,bins=80)
            # plt.show()
            bins, bins_str, bins_mean = self.__divide_bins_equal_interval(x_vals, min_v=bin_min, max_v=bin_max, n=bin_n,
                                                                          include_external=False)
            mean_list = []
            events_number = []
            boxes = []
            err_list = []
            x_err_list = []

            for i in tqdm(range(len(bins))):
                if i + 1 >= len(bins):
                    break
                df_wb = df[df[x_var] > bins[i]]
                df_wb = df_wb[df_wb[x_var] < bins[i + 1]]
                wb = df_wb[x_var]
                carbonloss = df_wb[y_var].tolist()
                events_number.append(len(carbonloss))
                bar = np.nanmean(carbonloss)
                err = np.nanstd(carbonloss) / 4.
                xerr = np.nanstd(wb)
                mean_list.append(bar)
                err_list.append(err)
                x_err_list.append(xerr)
                boxes.append(carbonloss)
            mean_list = np.array(mean_list)
            err_list = np.array(err_list)
            window = 5
            # err_list = SMOOTH().smooth_convolve(err_list,window_len=window)
            # mean_list_smooth = SMOOTH().smooth_convolve(mean_list,window_len=window)[:-1]
            mean_list_smooth = mean_list
            # bins_mean = range(len(bins_mean))
            plt.plot(bins_mean, mean_list_smooth, linewidth=3, zorder=99,label=lc)
            plt.scatter(bins_mean, mean_list_smooth, s=80, zorder=100)
            plt.errorbar(bins_mean, mean_list_smooth, yerr=err_list,
                         capsize=4,
                         ls='none')
            # Plot_line().plot_line_with_gradient_error_band(bins_mean,mean_list_smooth,err_list,
            #                                                c=color_list[color_flag],color_gradient_n=200)
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            # ax_hist = plt.twinx()
            # plt.plot(bins_mean,events_number,label=title+' events number')
            plt.tight_layout()
            plt.legend()
        plt.show()

        pass

    def bin_scatter(self):
        df,dff = self.__load_df()
        x_var_list = [
            'min_precip_anomaly_in_drought_range',
            'max_vpd_in_drought_range',
            'Aridity_Index',
            'zr',
            'Rplant',
            'Aridity_Index',
        ]
        y_var_list = [
            'Recovery_rc',
            'Resilience_rs',
            'Resistance_rt',
            'CSIF_anomaly_loss',
        ]
        Y_var = 'CSIF_anomaly_loss'
        X_var = 'max_vpd_in_drought_range'
        # X_var = 'max_vpd_anomaly_in_drought_range'
        # X_var = 'min_precip_anomaly_in_drought_range'
        # bin_var = 'Rplant'
        bin_var = 'zr'
        bin_min, bin_max = self.__bin_min_max(bin_var)
        bin_n = 10
        df, dff = self.__load_df()
        lc_list = df['lc_broad_needle'].tolist()
        lc_list = list(set(lc_list))
        lc_list.sort()

        for lc in lc_list:
            df, dff = self.__load_df()
            df = df[df['lc_broad_needle']==lc]
            df = Global_vars().clean_df(df)
            df = df[df['CSIF_anomaly_loss']<5]
            # df = df[df[X_var]<0]
            bin_vals = df[bin_var]
            bins, bins_str, bins_mean = self.__divide_bins_equal_interval(bin_vals, min_v=bin_min, max_v=bin_max, n=bin_n,
                                                                                      include_external=False)
            for i in tqdm(range(len(bins))):
                if i + 1 >= len(bins):
                    break
                df_bin = df[df[bin_var] > bins[i]]
                df_bin = df_bin[df_bin[bin_var] < bins[i + 1]]
                # x_val = df_bin[X_var].tolist()
                # y_val = df_bin[Y_var].tolist()
                # plt.figure()
                # plt.scatter(x_val,y_val)
                # plt.xlabel(X_var)
                # plt.ylabel(Y_var)
                # plot_kinds = ["scatter", "hist", "hex", "kde", "reg", "resid"]
                g = sns.jointplot(x=X_var, y=Y_var, kind="reg", dropna=True, data=df_bin)
                # sns.jointplot(x=X_var, y=Y_var, kind="reg", dropna=True, data=df_bin)
                reg = LinearRegression()
                sklearn.linear_model()
                xval = df[X_var]
                yval = df[Y_var]
                reg.fit(xval,yval)
                plt.title('{}\n{}\n'.format(lc,bin_var)+bins_str[i])
            plt.show()

    def bin_correlation(self):
        df,dff = self.__load_df()
        x_var_list = [
            'min_precip_anomaly_in_drought_range',
            'max_vpd_in_drought_range',
            'Aridity_Index',
            'zr',
            'Rplant',
            'Aridity_Index',
        ]
        y_var_list = [
            'Recovery_rc',
            'Resilience_rs',
            'Resistance_rt',
            'CSIF_anomaly_loss',
        ]
        Y_var = 'CSIF_anomaly_loss'
        X_var = 'max_vpd_in_drought_range'
        # X_var = 'max_vpd_anomaly_in_drought_range'
        # X_var = 'min_precip_anomaly_in_drought_range'
        bin_var = 'Rplant'
        # bin_var = 'zr'
        bin_min, bin_max = self.__bin_min_max(bin_var)
        bin_n = 10
        df, dff = self.__load_df()
        lc_list = df['lc_broad_needle'].tolist()
        # lc_list = df['drought_type'].tolist()
        # lc_list = df['dominate'].tolist()
        lc_list = list(set(lc_list))
        lc_list.sort()

        for lc in lc_list:
            df, dff = self.__load_df()
            df = df[df['lc_broad_needle']==lc]
            # df = df[df['drought_type']==lc]
            # df = df[df['dominate']==lc]
            df = Global_vars().clean_df(df)
            df = df[df['CSIF_anomaly_loss']<5]
            # df = df[df[X_var]<0]
            bin_vals = df[bin_var]
            bins, bins_str, bins_mean = self.__divide_bins_equal_interval(bin_vals, min_v=bin_min, max_v=bin_max, n=bin_n,
                                                                                      include_external=False)
            # bins, bins_str, bins_mean = self.__divide_bins_quantile(bin_vals, min_v=bin_min, max_v=bin_max,
            #                                                               n=bin_n,)
            coef_list = []
            r_list = []
            for i in tqdm(range(len(bins))):
                if i + 1 >= len(bins):
                    break
                df_bin = df[df[bin_var] > bins[i]]
                df_bin = df_bin[df_bin[bin_var] < bins[i + 1]]
                reg = LinearRegression()
                xval = df_bin[X_var]
                yval = df_bin[Y_var]
                xval = np.array(xval)
                r,p = stats.pearsonr(xval,yval)
                r_list.append(r)
                xval = xval.reshape(-1, 1)
                yval = np.array(yval)
                reg.fit(xval,yval)
                coef_ = reg.coef_[0]
                coef_list.append(coef_)
            # plt.plot(bins_mean,coef_list,label=lc)
            # plt.scatter(bins_mean,coef_list)

            plt.plot(bins_mean,r_list,label=lc)
            plt.scatter(bins_mean,r_list)

            # plt.xticks(bins_mean,bins_str,rotation=90)
            # plt.ylabel('{} {} K'.format(X_var,Y_var))
            plt.ylabel('{} {} r'.format(X_var,Y_var))
            plt.xlabel(bin_var)
        plt.legend()
        plt.tight_layout()
        plt.show()



class Check:

    def __init__(self):

        pass

    def run(self):
        self.foo()
        pass


    def foo(self):
        repetitive_f = Main_flow_Carbon_loss().this_class_arr + \
                       'gen_recovery_time_legacy_repetitive_events_VPD_auto/recovery_time_legacy.pkl'
        repetitive_events_dic = T.load_dict_from_binary(repetitive_f)

        pix_list = []
        recovery_time_list = []
        drought_event_date_range_list = []
        recovery_date_range_list = []
        legacy_list = []
        drought_type = []


        for pix in tqdm(repetitive_events_dic,desc='repetitive events carbon loss'):
            events = repetitive_events_dic[pix]
            if len(events) == 0:
                continue
            for repetetive_event in events:
                initial_event = repetetive_event[0]

                initial_recovery_time = initial_event['recovery_time']
                initial_drought_event_date_range = initial_event['drought_event_date_range']
                initial_recovery_date_range = initial_event['recovery_date_range']
                initial_legacy = initial_event['carbon_loss']
                initial_drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(initial_drought_event_date_range)
                initial_recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(initial_recovery_date_range)

                pix_list.append(pix)
                recovery_time_list.append(initial_recovery_time)
                drought_event_date_range_list.append(tuple(initial_drought_event_date_range))
                recovery_date_range_list.append(tuple(initial_recovery_date_range))
                legacy_list.append(initial_legacy)
                drought_type.append('repetitive_initial')

                subsequential_event = repetetive_event[1]
                subsequential_recovery_time = subsequential_event['recovery_time']
                subsequential_drought_event_date_range = subsequential_event['drought_event_date_range']
                subsequential_recovery_date_range = subsequential_event['recovery_date_range']
                subsequential_legacy = subsequential_event['carbon_loss']
                subsequential_drought_event_date_range = Global_vars().growing_season_indx_to_all_year_indx(
                    subsequential_drought_event_date_range)
                subsequential_recovery_date_range = Global_vars().growing_season_indx_to_all_year_indx(subsequential_recovery_date_range)

                pix_list.append(pix)
                recovery_time_list.append(subsequential_recovery_time)
                drought_event_date_range_list.append(tuple(subsequential_drought_event_date_range))
                recovery_date_range_list.append(tuple(subsequential_recovery_date_range))
                legacy_list.append(subsequential_legacy)
                drought_type.append('repetitive_subsequential')

        pass


def main():
    # kill_matplotlib()

    # Main_Flow_Pick_drought_events().run()
    # Main_Flow_Pick_drought_events_05().run()
    # Main_flow_Carbon_loss().run()
    # Main_flow_Dataframe_NDVI_SPEI_legacy().run()
    # for threshold in ['-1.2','-1.8','-2',]:
    #     print('threshold',threshold)
    #     Main_flow_Dataframe_NDVI_SPEI_legacy_threshold(threshold).run()
    # Tif().run()
    Analysis().run()
    # Check().run()
    pass

if __name__ == '__main__':

    main()