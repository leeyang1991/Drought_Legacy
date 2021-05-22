# coding=utf-8

from __init__ import *
# from Main_flow_csif_legacy_2002 import *

results_root_main_flow_high_res = this_root + 'results_root_main_flow_high_res/'
results_root_main_flow = results_root_main_flow_high_res

class Global_vars:
    def __init__(self):
        # self.growing_date_range = list(range(5,11))
        self.tif_template_7200_3600 = this_root + 'conf/tif_template_005.tif'
        self.growing_date_range = self.gs_mons()
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

    def growing_season_indx_to_all_year_indx(self,indexs):
        new_indexs = []
        for indx in indexs:
            n_year = indx // len(self.growing_date_range)
            res_mon = indx % len(self.growing_date_range)
            real_indx = n_year * 12 + res_mon + self.growing_date_range[0]
            new_indexs.append(real_indx)
        new_indexs = tuple(new_indexs)
        return new_indexs



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
        # self.single_events()
        # self.repetitive_events()
        # self.check_single_events()
        # self.check_repetitive_events()
        self.check_single_and_repetitive_events()
        pass



    def repetitive_events(self):

        outdir = self.this_class_arr + 'repetitive_events/'
        fdir = data_root + 'SPEI12/per_pix/'
        T.mk_dir(outdir, force=True)
        threshold = -1.5
        n = 4 * 12

        for f in tqdm(os.listdir(fdir)):
            gs_mons = Global_vars().gs_mons()
            single_event_dic = {}
            dic = T.load_npy(fdir + f)
            fname = f.split('/')[-1]
            for pix in dic:
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
                events_gs = [] # 只要生长季事件
                for i in event_list:
                    level, drought_range = i
                    is_gs = 0
                    mon = drought_range[0] % 12 + 1
                    if mon in gs_mons:
                        is_gs = 1
                    if is_gs:
                        events_gs.append(drought_range)
                if len(events_gs) <= 1: # 全时段只有一次以下的事件，则不存在repetitive事件
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
                    if i+1 >= len(events_gs):
                        continue
                    if events_gs[i+1][0] - n > events_gs[i][0]:
                        initial_drought.append(events_gs[i+1])
                        initial_drought_indx.append(i+1)
                if len(initial_drought) == 0:
                    continue

                repetitive_events = []
                for init_indx in initial_drought_indx:
                    init_date = events_gs[init_indx][0]
                    one_recovery_period = list(range(init_date,init_date + n))
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
        # arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # DIC_and_TIF(Global_vars().tif_template_7200_3600).plot_back_ground_arr()
        # plt.show()
    # def pick_repetitive_events(self):
    #
    #     pass


    def single_events(self):
        outdir = self.this_class_arr + 'single_events/'
        # fdir = data_root + 'SPEI/per_pix_2002/'
        # fdir = data_root + 'CWD/CWD/per_pix_anomaly/'
        fdir = data_root + 'SPEI12/per_pix/'
        params = []
        for f in os.listdir(fdir):
            params.append([fdir + f,outdir])
            # self.pick_events([fdir + f,outdir])
        MULTIPROCESS(self.pick_events,params).run()
            # self.pick_events(fdir + f,outdir)
        pass

    def pick_events(self,params):
        # 前n个月和后n个月无极端干旱事件
        f, outdir = params
        n = 4*12
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
            threshold = -1.5
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

        # self.single_events()
        self.repetitive_events()

        pass

    def single_events(self):
        # 1 cal recovery time
        event_dic,spei_dic,sif_dic = self.load_data_single_events()
        out_dir = self.this_class_arr + 'gen_recovery_time_legacy_single_events/'
        self.gen_recovery_time_legacy_single(event_dic,spei_dic, sif_dic,out_dir)
        pass

    def repetitive_events(self):
        event_dic, spei_dic, sif_dic = self.load_data_repetitive_events(condition='')

        out_dir = self.this_class_arr + 'gen_recovery_time_legacy_repetitive_events/'
        self.gen_recovery_time_legacy_repetitive(event_dic,spei_dic, sif_dic,out_dir)


        pass


    def load_data_repetitive_events(self,condition=''):
        # events_dir = results_root_main_flow + 'arr/SPEI_preprocess/drought_events/'
        # SPEI_dir = data_root + 'SPEI/per_pix_clean/'
        # SIF_dir = data_root + 'CSIF/per_pix_anomaly_detrend/'

        events_dir = Main_Flow_Pick_drought_events().this_class_arr + 'repetitive_events/'
        SPEI_dir = data_root + 'SPEI12/per_pix/'
        SIF_dir = data_root + 'CSIF005/per_pix_anomaly_detrend/'

        event_dic = T.load_npy_dir(events_dir,condition)
        spei_dic = T.load_npy_dir(SPEI_dir,condition)
        sif_dic = T.load_npy_dir(SIF_dir,condition)

        return event_dic,spei_dic,sif_dic
        pass


    def kernel_gen_recovery_time_legacy_repetitive(self,date_range,ndvi,spei,growing_date_range,):
        # print(date_range)
        # exit()
        # event_start_index = T.pick_min_indx_from_1darray(spei, date_range)
        event_start_index = date_range[0]
        event_start_index_trans = self.__drought_indx_to_gs_indx(event_start_index, growing_date_range, len(ndvi))
        # print(event_start_index_trans)
        # exit()
        if event_start_index_trans == None:
            # print('__drought_indx_to_gs_indx failed')
            return None
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
        search_end_indx = 3 * len(growing_date_range)
        recovery_range = self.search1(ndvi_gs, spei_gs, event_start_index_trans, search_end_indx)
        # continue
        # recovery_time, lag, recovery_start_gs, recovery_start, 'undefined'
        ###########################################
        ###########################################
        ###########################################
        # print('recovery_range',recovery_range)
        if recovery_range == None:
            # print('event_start_index+search_end_indx >= len(ndvi) '
            #       'event_start_index_trans, search_end_indx',
            #       event_start_index_trans, search_end_indx)
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
                        carbonloss1 = results1['carbon_loss']
                        carbonloss2 = results2['carbon_loss']
                        recovery_time_result.append([carbonloss1,carbonloss2])
                    # exit()
                recovery_time_dic[pix] = recovery_time_result
            else:
                recovery_time_dic[pix] = []
        T.save_dict_to_binary(recovery_time_dic, outf)
        pass


    def load_data_single_events(self,condition=''):
        # events_dir = results_root_main_flow + 'arr/SPEI_preprocess/drought_events/'
        # SPEI_dir = data_root + 'SPEI/per_pix_clean/'
        # SIF_dir = data_root + 'CSIF/per_pix_anomaly_detrend/'

        events_dir = Main_Flow_Pick_drought_events().this_class_arr + 'single_events/'
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
            # print('search1')
            # print(event_start_index,search_end_indx,len(ndvi))
            return None
        selected_indx = []
        for i in range(event_start_index,event_start_index+search_end_indx):
            ndvi_i = ndvi[i]
            if ndvi_i < 0:
                selected_indx.append(i)
            else:
                selected_indx.append(999999)
        recovery_indx_gs = self.__split_999999(selected_indx)
        # if recovery_indx_gs == None:
            # print(' event_start_index,search_end_indx', event_start_index,search_end_indx)
            # plt.close()
            # plt.plot(ndvi,label='ndvi')
            # plt.plot(drought_indx,label='spei')
            # plt.legend()
            # plt.show()
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
        df = self.Carbon_loss_to_df(df)
        # print(df)
        # df = self.add_legacy_123_to_df(df)
        # df = self.add_greenness_loss_to_df(df)
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
        # df = self.add_TWS_to_df(df)
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
        # df = self.add_drought_year_SOS(df)
        # 21 add thaw data into df
        # df = self.add_thaw_date(df)
        # df = self.add_thaw_date_anomaly(df)
        # df = self.add_thaw_date_std_anomaly(df)
        # 22 add temp trend to df
        # df = self.add_temperature_trend_to_df(df)
        # 23 add MAT MAP to df
        # df = self.add_MAT_MAP_to_df(df)
        # -1 df to excel
        # add pre_drought vars
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


class Tif:


    def __init__(self):
        self.this_class_tif = results_root_main_flow + 'tif/Tif/'
        Tools().mk_dir(self.this_class_tif, force=True)

    def run(self):

        self.carbon_loss()
        # self.drought_start()
        pass


    def load_df(self):
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)

        return df,dff

        pass

    def carbon_loss(self):
        df ,dff = self.load_df()
        spatial_dic = DIC_and_TIF(Global_vars().tif_template_7200_3600).void_spatial_dic()
        outdir = self.this_class_tif + 'carbon_loss/'
        T.mk_dir(outdir)
        outf = outdir + 'carbon_loss.tif'
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


def main():
    # Main_Flow_Pick_drought_events().run()
    Main_flow_Carbon_loss().run()
    # Main_flow_Dataframe_NDVI_SPEI_legacy().run()
    # Tif().run()
    pass


if __name__ == '__main__':

    main()