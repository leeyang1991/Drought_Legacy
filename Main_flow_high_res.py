# coding=utf-8

from __init__ import *
from Main_flow_csif_legacy_2002 import *
results_root_main_flow_high_res = this_root + 'results_root_main_flow_high_res/'
results_root_main_flow = results_root_main_flow_high_res


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

def main():
    Main_Flow_Pick_drought_events().run()
    pass


if __name__ == '__main__':

    main()