# coding=utf-8

from Main_flow import *
from CSIF_legacy import *

class Correlation_CSIF_SPEI:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Correlation_CSIF_SPEI\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Correlation_CSIF_SPEI\\'
        self.this_class_png = results_root_main_flow + 'png\\Correlation_CSIF_SPEI\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        # self.correlation()
        self.tif_correlation()

        pass


    def tif_correlation(self):
        corr_dic = self.correlation()
        corr_dic_r = {}
        for key in corr_dic:
            corr_dic_r[key] = corr_dic[key][0]
        outtif = self.this_class_tif + 'Correlation_CSIF_SPEI.tif'
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(corr_dic_r)
        DIC_and_TIF().arr_to_tif(arr,outtif)
        pass

    def correlation(self):
        CSIF_dir = data_root + r'CSIF\per_pix_anomaly_detrend\\'
        SPEI_dir = data_root + r'SPEI\per_pix_clean\\'

        csif_dic = T.load_npy_dir(CSIF_dir)
        spei_dic = T.load_npy_dir(SPEI_dir)

        corr_dic = {}
        gs_mon = list(range(4,11))

        for pix in tqdm(csif_dic):
            csif = csif_dic[pix]
            if not pix in spei_dic:
                continue
            spei = spei_dic[pix]
            gs_indx = []
            for m in range(len(csif)):
                mon = m % 12 + 1
                mon = int(mon)
                if mon in gs_mon:
                    gs_indx.append(m)
            csif_gs = T.pick_vals_from_1darray(csif,gs_indx)
            spei_gs = T.pick_vals_from_1darray(spei,gs_indx)
            r,p = stats.pearsonr(csif_gs,spei_gs)
            # r,p = stats.pearsonr(csif,spei)
            corr_dic[pix] = (r,p)
        return corr_dic
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(corr_dic)
        # plt.imshow(arr,vmin=-0.6,vmax=0.6)
        # plt.colorbar()
        # plt.show()

        pass



class Statistic:

    def __init__(self):
        self.this_class_arr = results_root_main_flow + 'arr\\Statistic\\'
        self.this_class_tif = results_root_main_flow + 'tif\\Statistic\\'
        self.this_class_png = results_root_main_flow + 'png\\Statistic\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        # self.vars_pairplot()
        # self.legacy_change()
        # self.vars_pairplot()
        self.plot_scatter()
        # self.legacy_trend_box()
        pass



    def load_df(self):
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        T.print_head_n(df)
        return df,dff




    def vars_pairplot(self):

        df,dff = self.load_df()
        lc_list = [
            'Forest',
            'Shrublands',
            'Grasslands',
        ]
        # lc = lc_list[0]
        # lc = lc_list[1]
        lc = lc_list[2]
        outf = lc + '.png'
        df = df[df['lc'] == lc]
        df = df[df['lat'] > 30]
        df = df[df['lat'] < 60]
        df = df[df['canopy_height'] > 0]
        # df = df[df['rooting_depth'] > 0]
        df = df[df['gs_sif_spei_corr'] > 0]
        # df = df[df['gs_sif_spei_corr_p'] < 0.05]
        # df = df[df['rooting_depth']<30]

        # df = df[df['isohydricity']<0.6]
        # df = df[df['isohydricity']>0.9]
        spatial_dic = {}
        for i,row in df.iterrows():
            pix = row.pix
            spatial_dic[pix] = 1
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(arr)
        # plt.figure()
        new_df = pd.DataFrame()
        # rooting_depth = df['rooting_depth']
        canopy_height = df['canopy_height']
        legacy = df['delta_legacy']
        isohydricity = df['isohydricity']
        waterbalance = df['waterbalance']
        PRE_cv = df['PRE_cv']
        TMP_cv = df['TMP_cv']
        # new_df['rooting_depth'] = rooting_depth
        new_df['canopy height'] = canopy_height
        new_df['isohydricity'] = isohydricity
        new_df['waterbalance'] = waterbalance
        new_df['PRE_cv'] = PRE_cv
        new_df['TMP_cv'] = TMP_cv
        new_df['legacy'] = legacy

        # new_df['TWS_-1'] = tws_minus_1
        # new_df['TWS_1'] = tws1
        # new_df['TWS_2'] = tws2
        # new_df['TWS_3'] = tws3
        # new_df['is_gs'] = is_gs

        # new_df = new_df.dropna()
        sns.pairplot(new_df,markers='.',kind='reg',diag_kind='kde')
        plt.suptitle(lc)
        plt.tight_layout()
        # plt.show()
        plt.savefig(outf)


    def legacy_change(self):
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        df = Main_flow_Dataframe_NDVI_SPEI_legacy().drop_duplicated_sample(df)

        df = df[df['lat'] > 30]
        df = df[df['lat'] < 60]
        # lc = 'Grasslands'
        # lc = 'Forest'
        xticks = []
        xticks_num = []
        flag = 0
        flag1 = 0
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        # for lc in Global_vars().koppen_landuse():
        # for lc in Global_vars().koppen_list():
        for lc in Global_vars().landuse_list():
            flag1 += 2
            xticks_num.append(flag1)
            xticks.append(lc)
            # xticks.append(lc)
            flag += 2
            df_selected = df[df['lc'] == lc]
            # df_selected = df[df['climate_zone'] == lc]
            # df_selected = df[df['kp'] == lc]
            # df = df[df['gs_sif_spei_corr'] > 0]
            # df = df[df['gs_sif_spei_corr_p'] < 0.05]

            total_year = len(list(range(1982,2016))) * 6
            half_total_year = total_year / 2
            # print(total_year)
            # exit()
            part1 = []
            part2 = []
            for i,row in tqdm(df_selected.iterrows(),total=len(df_selected)):
                pix = row.pix
                drought_event_date_range = row.drought_event_date_range
                legacy = row.legacy
                if np.isnan(legacy):
                    continue
                spatial_dic[pix].append(legacy)
                # print(legacy)
                event_start = drought_event_date_range[0]
                if event_start<= half_total_year:
                    part1.append(legacy)
                else:
                    part2.append(legacy)
            plt.boxplot([part1,part2],positions=[flag,flag+1],showfliers=False)
            ff, pp = f_oneway(part1, part2)
            print(lc)
            print(ff)
            print(pp)
            print('*'*8)
            plt.ylim(-10,1)
            # plt.title(lc)
            plt.grid()
        print(xticks_num)
        print(xticks)
        plt.xticks(xticks_num, xticks,rotation=90)
        plt.tight_layout()
        plt.figure()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(spatial_dic)
        plt.imshow(arr)
        DIC_and_TIF().plot_back_ground_arr()
        plt.show()
        # plt.hist(part1)
        # plt.figure()
        # plt.hist(part2)
        # plt.show()
        # for hist in [part1,part2]:
        #     hist = np.array(hist)
        #     hist = T.remove_np_nan(hist)
        #     n, x, _ = plt.hist(hist, bins=120, alpha=1.0, density=True, histtype=u'step',
        #                        )
        #     # density = stats.gaussian_kde(hist)
        #     # plt.plot(x,density(x),label='legacy_{}'.format(year))
        #     # print(n)
        #     nn = SMOOTH().smooth_convolve(n, 21)
        #     plt.plot(x, nn)
        #
        # plt.show()


        pass


    def legacy_trend_box(self):
        df,dff = self.load_df()
        df = Global_vars().clean_df(df)
        box = []
        for lc in Global_vars().landuse_list():
            df_lc = df[df['lc']==lc]
            # spatial_dic = {}
            trend = df_lc.trend
            box.append(trend)
        plt.boxplot(box,labels=Global_vars().landuse_list())
        plt.show()

        pass


    def plot_scatter(self):
        x_vars, Y_var = Global_vars().variables()

        dest_var = Y_var
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        df = Global_vars().clean_df(df)
        print(df.columns)
        kl_list = list(set(list(df['lc'])))
        # kl_list = list(set(list(df['climate_zone'])))
        kl_list.remove(None)
        kl_list.sort()
        results_dic = {}
        for kl in kl_list:
            print(kl)
            vars_list = x_vars
            df_kl = df[df['lc'] == kl]
            # df_kl = df[df['climate_zone'] == kl]
            df_kl = df_kl.replace([np.inf, -np.inf], np.nan)
            all_vars_list = copy.copy(vars_list)
            all_vars_list.append(dest_var)
            XXX = df_kl[vars_list]
            if len(XXX) < 100:
                print('{} sample number < 100'.format(kl))
                continue
            spatial_dic = {}
            for i, row in df_kl.iterrows():
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
            # print(XX)
            flag = 0
            plt.figure(figsize=(12, 8))
            for x in X:
                # print(x)
                flag += 1
                ax = plt.subplot(3, 4, flag)
                x_val = X[x]
                std = np.std(x_val)
                mean = np.mean(x_val)
                up = mean + std
                down = mean - std
                x_val_new = []
                for v in x_val:
                    if v > up:
                        x_val_new.append(np.nan)
                        continue
                    if v < down:
                        x_val_new.append(np.nan)
                        continue
                    x_val_new.append(v)
                temp_df = pd.DataFrame()
                temp_df['x'] = x_val_new
                temp_df['y'] = np.array(Y)
                temp_df = temp_df.dropna()
                x_ = np.array(temp_df.x)
                y_ = np.array(temp_df.y)
                if x in ['rooting_depth','sand','canopy_height']:
                    KDE_plot().plot_scatter(x_, y_, ax=ax, plot_fit_line=True, s=1.2)
                else:
                    KDE_plot().plot_scatter(X[x], Y, ax=ax, plot_fit_line=True, s=1.2)
                # plt.scatter(X[x],Y)
                plt.xlabel(x)
                plt.ylabel('legacy trend')
                plt.title(' ')
                plt.subplots_adjust(
                    top=0.956,
                    bottom=0.068,
                    left=0.064,
                    right=0.987,
                    hspace=0.402,
                    wspace=0.461
                )
            # plt.suptitle('{} no drop'.format(kl))
            # plt.tight_layout()
            plt.suptitle('{}'.format(kl))
            plt.show()

        pass

class Tif:

    def __init__(self):
        self.this_class_tif = results_root_main_flow + 'tif\\Tif\\'
        Tools().mk_dir(self.this_class_tif, force=True)
        pass

    def run(self):
        # self.tif_legacy()
        # self.tif_delta_legacy()
        self.tif_legacy_trend()
        # self.shp_legacy_trend_sig_star()
        pass


    def tif_legacy(self):
        outtifdir = self.this_class_tif + 'tif_legacy\\'
        T.mk_dir(outtifdir)
        # outtif = outtifdir + 'tif_legacy.tif'
        outtif = outtifdir + 'tif_legacy_reg_sig.tif'
        f = Recovery_time_Legacy().this_class_arr + 'Recovery_time_Legacy\\recovery_time_legacy_reg_sig.pkl'
        # f = Recovery_time_Legacy().this_class_arr + 'Recovery_time_Legacy\\recovery_time_legacy.pkl'
        dic = T.load_dict_from_binary(f)
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for pix in dic:
            vals = dic[pix]
            if len(vals) == 0:
                continue
            for dic_i in vals:
                legacy = dic_i['legacy']
                spatial_dic[pix].append(legacy)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(spatial_dic)
        DIC_and_TIF().arr_to_tif(arr,outtif)
        pass

    def tif_delta_legacy(self):
        outtifdir = self.this_class_tif + 'tif_delta_legacy\\'
        T.mk_dir(outtifdir)
        # outtif = outtifdir + 'tif_legacy.tif'
        outtif = outtifdir + 'delta_legacy.tif'
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        spatial_dic = {}
        for i,row in df.iterrows():
            pix = row.pix

            val = row.delta_legacy
            spatial_dic[pix] = val
        DIC_and_TIF().pix_dic_to_tif(spatial_dic,outtif)
        pass


    def tif_legacy_trend(self):
        outtifdir = self.this_class_tif + 'tif_legacy_trend\\'
        T.mk_dir(outtifdir)
        outtif = outtifdir + 'trend_legacy.tif'
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        spatial_dic = {}
        for i, row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = row.trend
            score = row.trend_score
            corr = row.gs_sif_spei_corr
            if corr < 0:
                continue
            # if score > 0.5:
            spatial_dic[pix] = val
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)

        DIC_and_TIF().arr_to_tif(arr,outtif)

    def tif_legacy_trend_byte(self):
        outtifdir = self.this_class_tif + 'tif_legacy_trend\\'
        T.mk_dir(outtifdir)
        # outtif = outtifdir + 'tif_legacy.tif'
        outtif = outtifdir + 'trend_legacy_sig_byte.tif'
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        spatial_dic = {}
        for i, row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = row.trend
            score = row.trend_score
            corr = row.gs_sif_spei_corr
            if corr < 0:
                continue
            if score > 0.5:
                spatial_dic[pix] = val
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        new_arr = []
        for i in range(len(arr)):
            temp = []
            for j in range(len(arr[0])):
                val = arr[i][j]
                if np.isnan(val):
                    temp.append(255)
                else:
                    temp.append(1)
            new_arr.append(temp)
        new_arr = np.array(new_arr)

        tif_template = DIC_and_TIF().tif_template
        arr_template, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        to_raster.array2raster_GDT_Byte(outtif,originX,originY,pixelWidth,pixelHeight,new_arr)


    def shp_legacy_trend_sig_star(self):
        outtifdir = self.this_class_tif + 'tif_legacy_trend\\'
        T.mk_dir(outtifdir)
        # outtif = outtifdir + 'tif_legacy.tif'
        outtif = outtifdir + 'trend_legacy_sig.tif'
        dff = Main_flow_Dataframe_NDVI_SPEI_legacy().dff
        df = T.load_df(dff)
        spatial_dic = {}
        for i, row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = row.trend
            score = row.trend_score
            if score > 0.7:
                spatial_dic[pix] = val
        outf = self.this_class_tif + 'shp_legacy_trend_sig_star'
        DIC_and_TIF().pix_dic_to_shp(spatial_dic,outf)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # DIC_and_TIF().plot_back_ground_arr()
        # plt.show()
        # DIC_and_TIF().pix_dic_to_tif(spatial_dic, outtif)


class Climate_Vars_delta_change:

    def __init__(self):

        pass

    def run(self):
        # self.delta()
        # self.CV()
        self.CV_delta()
        # self.spei_delta()
        # self.trend()
        # self.check()
        pass

    def __pick_gs_vals(self,vals,gs_mons):
        picked_vals = []
        for i in range(len(vals)):
            mon = i % 12 + 1
            if mon in gs_mons:
                picked_vals.append(vals[i])
        picked_vals = np.array(picked_vals)
        return picked_vals

    def delta(self):
        gs_mons = list(range(4,10))
        fdir = data_root + 'Climate_408\\'
        for climate_var in os.listdir(fdir):
            # npy_dir = os.path.join(fdir, climate_var,'per_pix_clean_anomaly_smooth') + '\\'
            npy_dir = os.path.join(fdir, climate_var,'per_pix_clean') + '\\'
            outdir = fdir + climate_var + '\\' + 'delta\\'
            T.mk_dir(outdir)
            outf = outdir + 'delta_origin_val'
            dic = T.load_npy_dir(npy_dir)
            delta_dic = {}
            for pix in tqdm(dic,desc=climate_var):
                vals = dic[pix]
                gs_vals = self.__pick_gs_vals(vals,gs_mons)
                half = int(len(gs_vals)/2)
                part1 = gs_vals[half:]
                part2 = gs_vals[:half]
                delta = np.mean(part2) - np.mean(part1)
                delta_dic[pix] = delta
            np.save(outf,delta_dic)
        pass


    def CV(self):
        '''
        Coefficient of variation
        :return:
        '''
        gs_mons = list(range(4, 10))
        fdir = data_root + 'Climate_408\\'
        for climate_var in os.listdir(fdir):
            # if climate_var != 'VPD':
            #     continue
            # npy_dir = os.path.join(fdir, climate_var,'per_pix_clean_anomaly_smooth') + '\\'
            npy_dir = os.path.join(fdir, climate_var, 'per_pix_clean') + '\\'
            outdir = fdir + climate_var + '\\' + 'CV\\'
            T.mk_dir(outdir)
            outf = outdir + 'CV'
            dic = T.load_npy_dir(npy_dir,)
            CV_dic = {}
            for pix in tqdm(dic, desc=climate_var):
                vals = dic[pix]
                gs_vals = self.__pick_gs_vals(vals, gs_mons)
                # gs_vals = vals
                std = np.std(gs_vals)
                # print(std)
                # plt.plot(gs_vals)
                # plt.figure()
                # plt.hist(gs_vals,bins=20)
                # plt.show()
                # mean = np.mean(gs_vals)
                # cv = std/mean
                CV_dic[pix] = std
            np.save(outf, CV_dic)

        pass



    def CV_delta(self):
        '''
        Coefficient of variation
        :return:
        '''
        gs_mons = list(range(4, 10))
        fdir = data_root + 'Climate_408\\'
        for climate_var in os.listdir(fdir):
            # if climate_var != 'VPD':
            #     continue
            npy_dir = os.path.join(fdir, climate_var,'per_pix_clean_anomaly_smooth') + '\\'
            # npy_dir = os.path.join(fdir, climate_var, 'per_pix_clean') + '\\'
            outdir = fdir + climate_var + '\\' + 'CV\\'
            T.mk_dir(outdir)
            outf = outdir + 'CV_delta'
            dic = T.load_npy_dir(npy_dir,)
            delta_CV_dic = {}
            for pix in tqdm(dic, desc=climate_var):
                vals = dic[pix]
                gs_vals = self.__pick_gs_vals(vals, gs_mons)
                gs_vals = signal.detrend(gs_vals)
                gs_vals_reshape = np.reshape(gs_vals,(2,-1))
                # exit()
                # gs_vals = vals
                std1 = np.std(gs_vals_reshape[0])
                std2 = np.std(gs_vals_reshape[1])
                delta_CV = std2 - std1
                # print(delta_CV)
                # plt.plot(gs_vals)
                # plt.figure()
                # plt.hist(gs_vals,bins=20)
                # plt.show()
                # mean = np.mean(gs_vals)
                # cv = std/mean
                delta_CV_dic[pix] = delta_CV
            np.save(outf, delta_CV_dic)

        pass

    def check(self):
        fdir = data_root + 'Climate_408\\'
        for climate_var in os.listdir(fdir):
            # f = fdir + climate_var + '\\' + 'delta\\delta.npy'
            # f = fdir + climate_var + '\\' + 'CV\\CV.npy'
            f = fdir + climate_var + '\\' + 'trend\\trend.npy'
            # f = fdir + climate_var + '\\' + 'delta\\delta_origin_val.npy'
            dic = T.load_npy(f)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic)
            DIC_and_TIF().plot_back_ground_arr()
            # plt.imshow(arr,vmin=0,vmax=100)
            # plt.imshow(arr,vmin=0,vmax=30)
            # plt.imshow(arr,vmin=0,vmax=1)
            plt.imshow(arr)
            plt.colorbar()
            plt.title(climate_var)
            plt.show()
        pass

    def spei_delta(self):
        gs_mons = list(range(4, 10))
        # npy_dir = os.path.join(fdir, climate_var,'per_pix_clean_anomaly_smooth') + '\\'
        npy_dir = data_root + 'SPEI\\per_pix_408\\'
        outdir = data_root + 'SPEI\\' + 'delta\\'
        T.mk_dir(outdir)
        outf = outdir + 'delta'
        dic = T.load_npy_dir(npy_dir)
        delta_dic = {}
        for pix in tqdm(dic, desc='spei delta'):
            vals = dic[pix]
            gs_vals = self.__pick_gs_vals(vals, gs_mons)
            half = int(len(gs_vals) / 2)
            part1 = gs_vals[half:]
            part2 = gs_vals[:half]
            # print(len(part1))
            # print(len(part2))
            # exit()
            delta = np.mean(part2) - np.mean(part1)
            # print(delta)
            delta_dic[pix] = delta
        np.save(outf, delta_dic)
        pass

        pass


    def trend(self):
        gs_mons = list(range(4, 10))
        fdir = data_root + 'Climate_408\\'
        for climate_var in os.listdir(fdir):
            npy_dir = os.path.join(fdir, climate_var,'per_pix_clean_anomaly_smooth') + '\\'
            # npy_dir = os.path.join(fdir, climate_var, 'per_pix_clean') + '\\'
            outdir = fdir + climate_var + '\\' + 'trend\\'
            T.mk_dir(outdir)
            outf = outdir + 'trend'
            dic = T.load_npy_dir(npy_dir)
            trend_dic = {}
            for pix in tqdm(dic, desc=climate_var):
                vals = dic[pix]
                gs_vals = self.__pick_gs_vals(vals, gs_mons)
                x_list = list(range(len(gs_vals)))
                x_list = np.array(x_list)
                x_list = x_list.reshape((-1,1))
                reg = LinearRegression()
                reg.fit(x_list,gs_vals)
                coef = reg.coef_[0]
                trend_dic[pix] = coef
                # score = reg.score(x_list,gs_vals)
                # if score > 0.2 and score < 0.9:
                #     print(coef)
                #     print(score)
                #     print(len(gs_vals))
                #     plt.plot(gs_vals)
                #     plt.show()
                # exit()
            np.save(outf, trend_dic)
        pass

class Constant_Vars:

    def __init__(self):
        pass

    def run(self):
        pass


def main():
    # Correlation_CSIF_SPEI().run()
    Statistic().run()
    # Tif().run()
    # Climate_Vars_delta_change().run()
    pass


if __name__ == '__main__':
    main()