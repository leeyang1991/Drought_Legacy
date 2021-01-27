# coding=gbk

from __init__ import *


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


class Main_flow_Legacy():
    '''
    based on CSIF SPEI linear regression
    '''
    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Main_flow_Legacy\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Main_flow_Legacy\\'
        self.this_class_png = results_root_main_flow + 'png\\Main_flow_Legacy\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)

        pass

    def run(self):
        # rf_model = T.load_npy(self.this_class_arr + 'rfmodel.npy')
        # rf_model_dic = self.cal_linear_reg()
        for legacy in [1,2,3]:
            print(legacy)
            # self.cal_legacy_annual(legacy)
            self.cal_legacy_monthly(legacy)

        # self.plot_legacy()
        # self.legacy_trend()
        # self.monthly_change()
        # self.different_legacy_sos()
        # self.legacy_and_sos_scatter()
        # self.plot_hist()
        # self.delta_legacy_and_isohydricity_linspace()
        # for legc in [1,2,3,4]:
        #     print legc
        # #     self.delta_legacy_per_pix(legc)
        #     self.delta_legacy_and_isohydricity_per_pix(legc)
        # self.delta_legacy_and_TWS_trend()
        # self.legacy_and_TWS_trend()
        pass


    def load_df(self):
        dff = Main_flow_Dataframe().dff
        df = T.load_df(dff)
        T.print_head_n(df)
        return df,dff

    def legacy_and_sos_scatter(self):
        df, dff = self.load_df()
        df = df[df['lat'] < 66.5]
        df = df[df['lat'] > 23.5]

        legacy = df['legacy_1']
        sos = df['dormant_length']

        new_df = pd.DataFrame()
        new_df['x']=legacy
        new_df['y']=sos
        new_df = new_df.dropna()

        x = new_df['x']
        y = new_df['y']
        KDE_plot().plot_scatter(x,y,plot_fit_line=False)
        # sns.jointplot(x=legacy, y=sos, kind="hex", color="#4CB391")

        # g = sns.JointGrid(data=new_df, x="x", y="y", space=0, ratio=17)
        # g.plot_joint(sns.scatterplot, sizes=(30, 120),
        #              color="g", alpha=.6, legend=False)
        # g.plot_marginals(sns.rugplot, height=1, color="g", alpha=.6)
        # plt.scatter(legacy,sos)
        plt.show()

        pass

    def different_legacy_sos(self):
        df, dff = self.load_df()
        df = df[df['lat'] < 66.5]
        df = df[df['lat'] > 23.5]
        for lc in Global_vars().landuse_list():
            df_lc = df[df['lc']==lc]
            # df_lc = df_lc[df_lc['correlation']<0]
            df_greater = df_lc[df_lc['legacy_1']>0]
            df_less = df_lc[df_lc['legacy_1']<-0]
            legacy = df_lc['legacy_1']
            legacy = legacy.dropna()
            sos_greater = df_greater['drought_year_sos_std_anomaly']
            sos_less = df_less['drought_year_sos_std_anomaly']
            # sos_greater = sos_greater.dropna()
            # sos_less = sos_less.dropna()
            plt.figure()
            plt.hist(sos_greater,bins=120,alpha=0.5,color='b',normed=1)
            plt.hist(sos_less,bins=120,alpha=0.5,color='r',normed=1)
            # plt.hist(legacy,bins=120,alpha=0.5,color='g',normed=1)
            # plt.hist(legacy,bins=120,alpha=0.5,normed=1)
            plt.title(lc)
        plt.show()

        pass

    def monthly_change(self):
        df, dff = self.load_df()
        df = df[df['lat'] < 66.5]
        df = df[df['lat'] > 23.5]

        for lc in Global_vars().landuse_list():
            df_lc = df[df['lc']==lc]
            # df_lc = df
            monthly_dic = {}
            for m in range(408):
                monthly_dic[m] = []
            for i, row in tqdm(df_lc.iterrows(), total=len(df_lc)):
                # pix = row.pix
                drought_event_date_range = row.drought_event_date_range
                drought_start = drought_event_date_range[0]
                # gs_mons = row.gs_mons
                drought_start_mon = drought_start % 12 + 1

                # print drought_start_year
                legacy = row.legacy_1
                legacy = -legacy
                # if not drought_start_mon in gs_mons:
                #     legacy = np.nan
                correlation = row.correlation
                # if legacy > 0:
                #     continue
                if correlation < 0:
                    continue
                if np.isnan(legacy):
                    continue
                monthly_dic[drought_start].append(legacy)
            mean_list = []
            upper = []
            lower = []
            for year in range(408):
                vals = monthly_dic[year]
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                mean_list.append(mean)
                upper.append(mean+std)
                lower.append(mean-std)

            mean_list_smooth = SMOOTH().mid_window_smooth(mean_list,window=59)
            plt.figure()
            plt.ylim(-0.7, 0.7)
            plt.plot(mean_list_smooth,c='gray')
            # plt.plot(upper)
            # plt.plot(lower)
            plt.fill_between(range(len(mean_list)), upper, lower,alpha=0.5)
            plt.title(lc)
        plt.show()
        pass
        pass


    def legacy_trend(self):
        df,dff = self.load_df()
        df = df[df['lat']<66.5]
        df = df[df['lat']>23.5]

        for lc in Global_vars().koppen_list():
            df_lc = df[df['lc']==lc]
            # df_lc = df
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            year_dic = {}
            for year in range(1982, 2016):
                year_dic[year] = []
            for i,row in tqdm(df_lc.iterrows(),total=len(df_lc)):
                # pix = row.pix
                drought_event_date_range = row.drought_event_date_range
                drought_start = drought_event_date_range[0]
                drought_start_year = drought_start // 12 + 1982
                # print drought_start_year
                legacy = row.legacy_year_1
                # legacy = -legacy
                correlation = row.correlation
                # if legacy > 0:
                #     continue
                if correlation < 0:
                    continue
                if np.isnan(legacy):
                    continue
                year_dic[drought_start_year].append(legacy)
            box = []
            for year in range(1982,2016):
                vals = year_dic[year]
                box.append(vals)
            plt.figure()
            plt.boxplot(box,positions=range(1982,2016),showfliers=False)
            # plt.ylim(-0.7,0.7)
            plt.grid()
            plt.title(lc)
        plt.show()
        pass

    def __pick_gs_vals(self,gs_mon,vals):
        gs_vals = []
        for i,val in enumerate(vals):
            mon = i % 12 + 1
            if mon in gs_mon:
                gs_vals.append(val)
        return gs_vals

    def cal_linear_reg(self):
        outf = self.this_class_arr + 'rfmodel'
        df,dff = self.load_df()
        # df = df.drop(columns=['legacy_1','legacy_2','legacy_3','legacy_4','linear_model_params_a_b_score'])
        # df = df.drop(columns=['legacy'])
        # T.save_df(df,dff)
        # exit()
        NDVIdir = data_root + 'NDVI\\per_pix_clean_anomaly_smooth\\'
        SPEIdir = data_root + 'SPEI\\per_pix_clean_anomaly_smooth\\'
        outdir = self.this_class_arr + 'cal_trend\\'
        T.mk_dir(outdir)
        NDVIdic = {}
        for f in tqdm(os.listdir(NDVIdir)):
            dic_i = T.load_npy(NDVIdir + f)
            NDVIdic.update(dic_i)

        SPEIdic = {}
        for f in tqdm(os.listdir(SPEIdir)):
            dic_i = T.load_npy(SPEIdir + f)
            SPEIdic.update(dic_i)
        rf_model_dic = {}
        # flag = 1
        for i,row in tqdm(df.iterrows(),total=len(df)):
            # flag += 1
            # if flag > 100:
            #     break
            pix = row.pix
            if pix in rf_model_dic:
                continue
            # if pix in linear_model_params_dic:
            #     continue
            gs_mons = row.gs_mons
            if not pix in NDVIdic:
                # rf_model_list.append(np.nan)
                continue
            if not pix in SPEIdic:
                # rf_model_list.append(np.nan)
                continue
            NDVIvals = NDVIdic[pix]
            SPEIvals = SPEIdic[pix]
            NDVI_GS_vals = self.__pick_gs_vals(gs_mons,NDVIvals)
            SPEI_GS_vals = self.__pick_gs_vals(gs_mons,SPEIvals)
            SPEIvals = np.array(SPEI_GS_vals)
            NDVIvals = np.array(NDVI_GS_vals)
            SPEIvals[SPEIvals<-2.]=np.nan
            df_temp = pd.DataFrame()
            df_temp['spei'] = SPEIvals
            df_temp['ndvi'] = NDVIvals
            df_temp = df_temp.dropna()
            x = np.array(df_temp['spei']).reshape(-1, 1)
            y = np.array(df_temp['ndvi']).reshape(-1, 1)
            # print x
            # reg = LinearRegression().fit(x,y)

            # rf = RandomForestRegressor()
            # rf.fit(x, y)

            lr = LinearRegression()
            lr.fit(x, y)

            # print rf
            # exit()
            # exit()
            # pred_ndvi = rf.predict(x)
            # a = reg.coef_[0]
            # b = reg.intercept_

            # pre_ndvi1 = a * df_temp['spei'] + b
            # plt.plot(x)
            # plt.plot(pred_ndvi)
            #
            # plt.figure()
            # plt.scatter(pred_ndvi,x)
            # plt.show()
            # r = reg.score(SPEIvals, NDVIvals)
            # a,b,r = KDE_plot().linefit(SPEIvals,NDVIvals)
            # rf_model_list.append(rf)
            # rf_model_dic[pix] = rf
            rf_model_dic[pix] = lr
        # df['rf_model_list'] = rf_model_list
        # dff = Main_flow_Prepare().this_class_arr + 'prepare\\data_frame_threshold_0.df'
        # T.save_df(df,dff)
        # np.save(outf,rf_model_dic)
        return rf_model_dic


    def cal_legacy_annual(self,legacy_year):

        SIFdir = data_root + 'CSIF\\per_pix_anomaly_detrend\\'
        SPEIdir = data_root + 'SPEI\\per_pix_clean\\'
        SIFdic = T.load_npy_dir(SIFdir)
        SPEIdic = T.load_npy_dir(SPEIdir)
        df,dff = self.load_df()
        legacy_list = []
        # rf_model_dic = self.cal_linear_reg()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            # linear_model_params_a_b_score = row['linear_model_params_a_b_score']
            # if not type(linear_model_params_a_b_score) == tuple:
            #     legacy_list.append(np.nan)
            #     continue
            # a,b,score = linear_model_params_a_b_score
            # print a,b,score
            pix = row.pix
            event = row.event
            drought_start = event[0]
            drought_start_year = event[0] // 12
            drought_start_year = int(drought_start_year)
            drought_mon = drought_start % 12 + 1
            drought_mon = int(drought_mon)
            gs_mons = list(range(1,13)) ################## Todo: Need to calculate Growing season via phenology
            if not drought_mon in gs_mons:
                legacy_list.append(np.nan)
                continue
            if not pix in SPEIdic:
                legacy_list.append(np.nan)
                continue
            if not pix in SIFdic:
                legacy_list.append(np.nan)
                continue
            spei = SPEIdic[pix]
            spei = np.array(spei)
            sif = SIFdic[pix]
            sif = np.array(sif)

            # plt.plot(spei)
            # plt.plot(sif)
            # plt.show()

            gs_indx = []
            for m in range(len(sif)):
                mon = m % 12 + 1
                mon = int(mon)
                if mon in gs_mons:
                    gs_indx.append(m)
            sif_gs = T.pick_vals_from_1darray(sif,gs_indx)
            spei_gs = T.pick_vals_from_1darray(spei,gs_indx)
            # plt.plot(sif_gs)
            # plt.plot(spei_gs)
            # plt.show()
            sif_gs_reshape = np.reshape(sif_gs,(int(len(sif_gs)/len(gs_mons)),len(gs_mons)))
            spei_gs_reshape = np.reshape(spei_gs,(int(len(spei_gs)/len(gs_mons)),len(gs_mons)))
            sif_annual = []
            for sif_i in sif_gs_reshape:
                sif_annual.append(np.mean(sif_i))
            spei_annual = []
            for spei_i in spei_gs_reshape:
                spei_annual.append(np.mean(spei_i))
            sif_annual = np.array(sif_annual)
            spei_annual = np.array(spei_annual)
            # plt.figure()
            # plt.plot(sif_annual)
            # plt.plot(spei_annual)
            # plt.figure()
            # plt.scatter(spei_annual,sif_annual)
            # plt.show()
            x = spei_annual.reshape(-1,1)
            y = sif_annual
            lr = LinearRegression()
            lr.fit(x,y)
            sif_annual_pred = lr.predict(x)

            # legacy_months_start = (legacy_year - 1) * len(gs_mons)
            # legacy_months_end = legacy_year * 12

            if drought_start_year + legacy_year >= len(spei_annual):
                legacy_list.append(np.nan)
                continue
            legacy_duration = range(drought_start_year + legacy_year - 1, drought_start_year + legacy_year)
            # print legacy_duration
            # exit()
            selected_indx = list(legacy_duration)
            # sif_GS_vals = self.__pick_gs_vals(gs_mons,sifvals)
            # SPEI_GS_vals = self.__pick_gs_vals(gs_mons,SPEIvals)
            df_temp = pd.DataFrame()
            sif_obs_selected = T.pick_vals_from_1darray(sif_annual,selected_indx)
            sif_pred_selected = T.pick_vals_from_1darray(sif_annual_pred,selected_indx)
            legacy = sif_obs_selected - sif_pred_selected

            legacy_mean = np.mean(legacy)
            # legacy_mean = np.sum(legacy)
            legacy_list.append(legacy_mean)
            # print legacy_mean
            # legacy_dic[pix] = legacy_mean

        df['annual_legacy_year_{}'.format(legacy_year)] = legacy_list
        T.save_df(df,dff)


    def cal_legacy_monthly(self,legacy_year):

        SIFdir = data_root + 'CSIF\\per_pix_anomaly_detrend\\'
        SPEIdir = data_root + 'SPEI\\per_pix_clean\\'
        SIFdic = T.load_npy_dir(SIFdir)
        SPEIdic = T.load_npy_dir(SPEIdir)
        df,dff = self.load_df()
        legacy_list = []
        # rf_model_dic = self.cal_linear_reg()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            # linear_model_params_a_b_score = row['linear_model_params_a_b_score']
            # if not type(linear_model_params_a_b_score) == tuple:
            #     legacy_list.append(np.nan)
            #     continue
            # a,b,score = linear_model_params_a_b_score
            # print a,b,score
            pix = row.pix
            event = row.event
            drought_start = event[0]
            # drought_start_year = event[0] // 12
            # drought_start_year = int(drought_start_year)
            drought_mon = drought_start % 12 + 1
            drought_mon = int(drought_mon)
            gs_mons = list(range(1,13)) ################## Todo: Need to calculate Growing season via phenology
            if not drought_mon in gs_mons:
                legacy_list.append(np.nan)
                continue
            if not pix in SPEIdic:
                legacy_list.append(np.nan)
                continue
            if not pix in SIFdic:
                legacy_list.append(np.nan)
                continue
            spei = SPEIdic[pix]
            spei = np.array(spei)
            sif = SIFdic[pix]
            sif = np.array(sif)

            # plt.plot(spei)
            # plt.plot(sif)
            # plt.show()

            gs_indx = []
            for m in range(len(sif)):
                mon = m % 12 + 1
                mon = int(mon)
                if mon in gs_mons:
                    gs_indx.append(m)
            sif_gs = T.pick_vals_from_1darray(sif,gs_indx)
            spei_gs = T.pick_vals_from_1darray(spei,gs_indx)
            # plt.plot(sif_gs)
            # plt.plot(spei_gs)
            # plt.show()
            # plt.figure()
            # plt.plot(sif_annual)
            # plt.plot(spei_annual)
            # plt.figure()
            # plt.scatter(spei_annual,sif_annual)
            # plt.show()
            x = spei_gs.reshape(-1,1)
            y = sif_gs
            lr = LinearRegression()
            lr.fit(x,y)
            sif_pred = lr.predict(x)

            legacy_months_start = (legacy_year - 1) * len(gs_mons)
            legacy_months_end = legacy_year * 12

            if drought_start + legacy_months_end >= len(spei_gs):
                legacy_list.append(np.nan)
                continue
            legacy_duration = range(drought_start + legacy_months_start, drought_start + legacy_months_end)
            # legacy_duration = list(legacy_duration)
            # print(legacy_duration)
            # exit()
            selected_indx = list(legacy_duration)
            # sif_GS_vals = self.__pick_gs_vals(gs_mons,sifvals)
            # SPEI_GS_vals = self.__pick_gs_vals(gs_mons,SPEIvals)
            sif_obs_selected = T.pick_vals_from_1darray(sif_gs,selected_indx)
            sif_pred_selected = T.pick_vals_from_1darray(sif_pred,selected_indx)
            # plt.plot(sif_obs_selected)
            # plt.plot(sif_pred_selected)
            # plt.show()
            legacy = sif_obs_selected - sif_pred_selected

            legacy_mean = np.mean(legacy)
            # legacy_mean = np.sum(legacy)
            legacy_list.append(legacy_mean)
            # print(legacy_mean)
            # legacy_dic[pix] = legacy_mean

        df['monthly_legacy_year_{}'.format(legacy_year)] = legacy_list
        T.save_df(df,dff)



    def plot_legacy(self):
        df,dff = self.load_df()
        # exit()
        for year in [1,2,3,4]:
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            for i,row in tqdm(df.iterrows(),total=len(df)):
                pix = row.pix
                legacy = row['legacy_year_{}'.format(year)]
                spatial_dic[pix].append(legacy)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(spatial_dic)
            DIC_and_TIF().arr_to_tif(arr,self.this_class_tif + 'plot_legacy_{}_2.tif'.format(year))
            # plt.imshow(arr,vmax=0.2,vmin=-0.2)
            # plt.show()


    def plot_hist(self):
        df,dff = self.load_df()
        df = df[df['lat'] < 66.5]
        df = df[df['lat'] > 23.5]

        for lc in Global_vars().landuse_list():
            df_lc = df[df['lc'] == lc]
            # df_lc = df
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            year_dic = {}
            for year in range(1982, 2016):
                year_dic[year] = []
            for i, row in tqdm(df_lc.iterrows(), total=len(df_lc)):
                # pix = row.pix
                drought_event_date_range = row.drought_event_date_range
                drought_start = drought_event_date_range[0]
                drought_start_year = drought_start // 12 + 1982
                # print drought_start_year
                legacy = row.legacy_2
                correlation = row.correlation
                # if legacy > 0:
                #     continue
                # if correlation < 0:
                #     continue
                if np.isnan(legacy):
                    continue
                year_dic[drought_start_year].append(legacy)
            hist_1 = []
            hist_2 = []
            hist_3 = []
            for year in range(1982, 2016):
                vals = year_dic[year]
                if year < 1992:
                    for val in vals:
                        hist_1.append(val)
                elif 1992<= year < 2002:
                    for val in vals:
                        hist_2.append(val)
                elif 2002 <= year <= 2015:
                    for val in vals:
                        hist_3.append(val)

            plt.figure()
            plt.hist(hist_1,bins=80,normed=1,alpha=0.5)
            plt.hist(hist_2,bins=80,normed=1,alpha=0.5)
            plt.hist(hist_3,bins=80,normed=1,alpha=0.5)
            plt.title(lc)

            plt.figure()
            plt.boxplot([hist_1,hist_2,hist_3],showfliers=False)
            # plt.ylim(-0.7, 0.7)
            plt.grid()
            plt.title(lc)
        plt.show()

    def delta_legacy_and_isohydricity_linspace(self):
        df,dff = self.load_df()
        legacy_arg = 'legacy_year_4'
        # print df.loc[[0,1,2,3,4]]
        # exit()
        iso_hyd = df['isohydricity']
        iso_std = np.nanstd(iso_hyd)
        iso_mean = np.nanmean(iso_hyd)
        # legacy = df['legacy_year_1']
        min_iso = iso_mean - 3*iso_std
        max_iso = iso_mean + 3*iso_std
        # print min_iso
        # print max_iso
        # exit()
        # min_iso = 0.0
        # max_iso = 1.5
        classes = 20
        iso_hyd_seperate = np.linspace(min_iso,max_iso,classes)
        # print len(iso_hyd_seperate)
        # exit()
        # print iso_hyd_seperate
        iso_class_dic = {}
        for key in range(classes):
            iso_class_dic[key] = []
        # iso_class_dic[-1]=[]
        # iso_class_dic[len(iso_hyd_seperate)]=[]

        # print len()

        for i,row in tqdm(df.iterrows(),total=len(df)):
            iso_hyd = row.isohydricity
            if np.isnan(iso_hyd):
                continue
            iso_class = np.nan
            if iso_hyd <= min(iso_hyd_seperate):
                # iso_class = -1
                iso_class = np.nan
            elif iso_hyd >= max(iso_hyd_seperate):
                # iso_class = len(iso_hyd_seperate)
                iso_class = np.nan
            else:
                for j in range(len(iso_hyd_seperate)):
                    if j+1 >= len(iso_hyd_seperate):
                        break
                    if iso_hyd > iso_hyd_seperate[j] and iso_hyd < iso_hyd_seperate[j+1]:
                        iso_class = j
            if np.isnan(iso_class):
                continue
                # print iso_hyd
                # print iso_hyd_seperate
            iso_class_dic[iso_class].append(i)

        y = []
        x = []
        for cls in iso_class_dic:
            indxs = iso_class_dic[cls]
            df_selected = df.loc[indxs]
            part1_vals = []
            part2_vals = []
            for i,row in tqdm(df_selected.iterrows(),total=len(df_selected),desc=str(cls)):
                legacy = row[legacy_arg]
                if np.isnan(legacy):
                    continue
                drought_event_date_range = row.drought_event_date_range
                start_year = drought_event_date_range[0] // 12 + 1
                if start_year <= 17:
                    part1_vals.append(legacy)
                else:
                    part2_vals.append(legacy)
            part1_mean = np.mean(part1_vals)
            part2_mean = np.mean(part2_vals)
            delta = part2_mean - part1_mean
            y.append(delta)
            x.append(cls)
        # print len(y)
        plt.scatter(iso_hyd_seperate,y)
        plt.title(legacy_arg)
        plt.show()

    def delta_legacy_per_pix(self,legacy_arg):
        df,dff = self.load_df()
        legacy_arg = 'legacy_year_{}'.format(legacy_arg)
        # print len()
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            legacy = row[legacy_arg]
            if np.isnan(legacy):
                continue
            drought_event_date_range = row['drought_event_date_range']
            drought_year = drought_event_date_range[0]//12 + 1
            vals = (drought_year,legacy)
            spatial_dic[pix].append(vals)

        delta_legacy_dic = {}
        for pix in tqdm(spatial_dic):
            vals = spatial_dic[pix]
            if len(vals) == 0:
                continue
            dic_i = {}
            for y,v in vals:
                dic_i[y] = v

            part1 = []
            part2 = []
            for y in dic_i:
                if y <= 17:
                    part1.append(dic_i[y])
                else:
                    part2.append(dic_i[y])
            delta_legacy = np.mean(part2) - np.mean(part1)
            delta_legacy_dic[pix] = delta_legacy

        delta_legacy_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            if not pix in delta_legacy_dic:
                delta_legacy_list.append(np.nan)
                continue
            delta_legacy = delta_legacy_dic[pix]
            delta_legacy_list.append(delta_legacy)
        df['delta_{}'.format(legacy_arg)] = delta_legacy_list
        T.save_df(df,dff)

        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(delta_legacy_dic)
        # temp_dir = temp_results_dir + 'Main_flow_Legacy_2\\'
        # T.mk_dir(temp_dir)
        # outf = temp_dir + 'delta_legacy.tif'
        # DIC_and_TIF().arr_to_tif(arr,outf)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()

    def delta_legacy_and_isohydricity_per_pix(self,legacy_num):

        df, dff = self.load_df()
        legacy_arg = 'legacy_year_{}'.format(legacy_num)
        # for i,row in tqdm(df.iterrows(),total=len(df)):
        #     pix = row.pix
        #     delta_legacy = row[legacy_arg]
        #     if np.isnan(delta_legacy):
        #         continue
        #     drought_event_date_range = row['drought_event_date_range']
        #     drought_year = drought_event_date_range[0]//12 + 1
        #     iso_hyd = row['isohydricity']
        iso_hyd = df['isohydricity']
        delta_legacy = df[legacy_arg]
        df_temp = pd.DataFrame()
        df_temp['iso_hyd'] = iso_hyd
        df_temp['delta_legacy'] = delta_legacy
        df_temp = df_temp.dropna()
        sns.pairplot(df_temp)
        plt.show()


    def delta_legacy_and_TWS_trend(self):
        tws_trend_tif = temp_results_dir + 'TWS_WATER_Gap\\anomaly_trend.tif'
        # tws_trend_tif = temp_results_dir + 'TWS_WATER_Gap\\trend.tif'
        tws_trend_arr = to_raster.raster2array(tws_trend_tif)[0]
        T.mask_999999_arr(tws_trend_arr)

        tws_trend_arr[tws_trend_arr<-3]=np.nan
        tws_trend_arr[tws_trend_arr>3]=np.nan
        # tws_std = np.nanstd(tws_trend_arr)
        # tws_mean = np.nanmean(tws_trend_arr)
        # print tws_std
        # print tws_mean-tws_std
        # print tws_mean+tws_std
        # exit()

        tws_dic = DIC_and_TIF().spatial_arr_to_dic(tws_trend_arr)
        df,dff = self.load_df()
        # delta_legacy_year_1
        legacy_dic = {}
        lc_dic = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            delta_legacy = row.delta_legacy_year_3
            lc = row.lc
            pix = row.pix
            legacy_dic[pix] = delta_legacy
            lc_dic[pix] = lc
        temp_df = pd.DataFrame()
        legacy_list = []
        tws_trend_list = []
        lc_list = []
        for pix in legacy_dic:
            legacy = legacy_dic[pix]
            if not pix in tws_dic:
                continue
            tws_trend = tws_dic[pix]
            lc = lc_dic[pix]
            legacy_list.append(legacy)
            tws_trend_list.append(tws_trend)
            lc_list.append(lc)
        temp_df['tws_trend'] = tws_trend_list
        temp_df['legacy'] = legacy_list
        temp_df['landcover'] = lc_list
        temp_df = temp_df.dropna()
        sns.pairplot(temp_df,hue='landcover')
        # sns.jointplot(data=temp_df,x='legacy',y='tws_trend',hue='landcover',kind='kde')
        plt.show()



    def legacy_and_TWS_trend(self):
        tws_trend_tif = temp_results_dir + 'TWS_WATER_Gap\\anomaly_trend.tif'
        # tws_trend_tif = temp_results_dir + 'TWS_WATER_Gap\\trend.tif'
        tws_trend_arr = to_raster.raster2array(tws_trend_tif)[0]
        T.mask_999999_arr(tws_trend_arr)

        tws_trend_arr[tws_trend_arr<-3]=np.nan
        tws_trend_arr[tws_trend_arr>3]=np.nan
        # tws_std = np.nanstd(tws_trend_arr)
        # tws_mean = np.nanmean(tws_trend_arr)
        # print tws_std
        # print tws_mean-tws_std
        # print tws_mean+tws_std
        # exit()

        tws_dic = DIC_and_TIF().spatial_arr_to_dic(tws_trend_arr)
        df,dff = self.load_df()
        # delta_legacy_year_1
        legacy_dic = {}
        lc_dic = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            delta_legacy = row.legacy_year_1
            lc = row.lc
            pix = row.pix
            legacy_dic[pix] = delta_legacy
            lc_dic[pix] = lc
        temp_df = pd.DataFrame()
        legacy_list = []
        tws_trend_list = []
        lc_list = []
        for pix in legacy_dic:
            legacy = legacy_dic[pix]
            if not pix in tws_dic:
                continue
            tws_trend = tws_dic[pix]
            lc = lc_dic[pix]
            legacy_list.append(legacy)
            tws_trend_list.append(tws_trend)
            lc_list.append(lc)
        temp_df['tws_trend'] = tws_trend_list
        temp_df['legacy'] = legacy_list
        temp_df['landcover'] = lc_list
        temp_df = temp_df.dropna()
        sns.pairplot(temp_df,hue='landcover')
        # sns.jointplot(data=temp_df,x='legacy',y='tws_trend',hue='landcover',kind='kde')
        plt.show()




class Main_flow_Dataframe:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Main_flow_Dataframe\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Main_flow_Dataframe\\'
        self.this_class_png = results_root_main_flow + 'png\\Main_flow_Dataframe\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)

        self.dff = self.this_class_arr + 'data_frame.df'

    def run(self):
        # 0 generate a void dataframe
        df = self.__gen_df_init()
        # 1 add drought event into df
        # df = self.events_to_df(df)



        # -1 df to excel
        # T.save_df(df,self.dff)
        self.__df_to_excel(df,self.dff)
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

    def events_to_df(self,df):
        fdir = Main_Flow_Pick_drought_events().this_class_arr + 'drought_events\\'
        events_dic = T.load_npy_dir(fdir)
        pix_list = []
        event_list = []
        for pix in tqdm(events_dic,desc='1. events_to_df'):
            events = events_dic[pix]
            for event in events:
                pix_list.append(pix)
                event_list.append(event)
        df['pix'] = pix_list
        df['event'] = event_list
        return df
        pass

def main():
    # Main_Flow_Pick_drought_events().run()
    Main_flow_Dataframe().run()
    # Main_flow_Legacy().run()
    pass


if __name__ == '__main__':
    main()