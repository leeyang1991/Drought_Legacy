# coding=utf-8

from __init__ import *

class Tree_Ring_preprocess:

    def __init__(self):
        self.this_class_arr = results_root + 'arr\\Tree_Ring_preprocess\\'
        self.this_class_tif = results_root + 'tif\\Tree_Ring_preprocess\\'
        self.this_class_png = results_root + 'png\\Tree_Ring_preprocess\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        self.dff = self.this_class_arr + 'DataFrame.df'
        pass

    def run(self):
        # self.load_Scientific_Name()
        # self.load_wood_density()
        # self.write_wood_to_excel()
        # self.load_traits()
        self.write_traits_to_df()

        pass

    def write_wood_to_excel(self):

        Scientific_Name_dic = self.load_Scientific_Name()
        wood_density_dic = self.load_wood_density()

        f = data_root + 'Traits\\tri_NH(1).csv'
        df = pd.read_csv(f)

        wood_density_list = []
        Scientific_Name_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            speciesCode = row['speciesCode']
            Scientific_Name = Scientific_Name_dic[speciesCode]
            Scientific_Name_list.append(Scientific_Name)
            if not Scientific_Name in wood_density_dic:
                wood_density_list.append(np.nan)
                continue
            wood_density = wood_density_dic[Scientific_Name]
            wood_density_list.append(wood_density)
            # except:
            #     wood_density_list.append(np.nan)
            # print(speciesCode)
            # print(Scientific_Name)
            # print(wood_density)
            # pause()
            # pass
        df['wood_density'] = wood_density_list
        df['Scientific_Name'] = Scientific_Name_list
        T.save_df(df,self.dff)
        # df.to_excel(self.this_class_arr + 'dataframe.xlsx')
        pass

    def load_Scientific_Name(self):
        f = data_root + 'Traits\\tree\\tree-species-code.txt'
        fr = open(f,'r')
        fr.readline()
        lines = fr.readlines()
        fr.close()
        Scientific_Name_dic = {}
        for line in lines:
            # pause()
            line = line.split('\n')[0]
            # print([line])
            line_split = line.split()
            abbr = line_split[0]
            full_name = line_split[1:3]
            full_name = ' '.join(full_name)
            Scientific_Name_dic[abbr] = full_name
            # print(abbr)
            # print(full_name)
            # print('*'*8)
            # pause()
        return Scientific_Name_dic
        pass

    def load_wood_density(self):
        f = data_root + 'Traits\\tree\\GlobalWoodDensityDatabase.xlsx'
        df = pd.read_excel(f,sheet_name='Data')
        # T.print_head_n(df)
        Binomial = df['Binomial']
        Binomial_dic = {}
        for b in Binomial:
            Binomial_dic[b] = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            val = row['Wood density (g/cm^3), oven dry mass/fresh volume']
            Binomial = row['Binomial']
            Binomial_dic[Binomial].append(val)
        Binomial_mean_dic = {}
        for b in Binomial_dic:
            Binomial_mean_dic[b] = np.mean(Binomial_dic[b])
        # for b in Binomial_mean_dic:
        #     print(b,Binomial_mean_dic[b])
        return Binomial_mean_dic

    def load_traits(self):
        f = data_root + 'Traits\\tree\\41586_2012_BFnature11688_MOESM527_ESM.xlsx'
        df = pd.read_excel(f)
        psi50_dic = {}
        safety_margin_dic = {}
        Species = df['Species']
        for s in Species:
            psi50_dic[s] = np.nan
            safety_margin_dic[s] = np.nan
        for i,row in tqdm(df.iterrows(),total=len(df)):
            Species = row.Species
            psi50 = row['ψ50']
            safety_margin = row['ψ88 safety margin']

            psi50_dic[Species]=psi50
            safety_margin_dic[Species]=safety_margin

        return psi50_dic,safety_margin_dic

    def write_traits_to_df(self):
        psi50_dic, safety_margin_dic = self.load_traits()
        df = T.load_df(self.dff)
        psi50_list = []
        safety_margin_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            Scientific_Name = row.Scientific_Name
            if not Scientific_Name in psi50_dic:
                psi50_list.append(np.nan)
                safety_margin_list.append(np.nan)
                continue
            psi50 = psi50_dic[Scientific_Name]
            safety_margin = safety_margin_dic[Scientific_Name]
            psi50_list.append(psi50)
            safety_margin_list.append(safety_margin)

        df['psi50'] = psi50_list
        df['safety_margin'] = safety_margin_list

        T.save_df(df,self.dff)
        df.to_excel(self.this_class_arr + 'Dataframe.xlsx')
        pass


class Tree_Ring_Legacy:

    def __init__(self):
        self.this_class_arr = results_root + 'arr\\Tree_Ring_preprocess\\'
        self.this_class_tif = results_root + 'tif\\Tree_Ring_preprocess\\'
        self.this_class_png = results_root + 'png\\Tree_Ring_preprocess\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        self.dff = Tree_Ring_preprocess().this_class_arr + 'DataFrame.df'

    def run(self):
        # self.Tree_Ring_to_df()
        # self.load_Tree_Ring()
        # self.integrate_lon_lat()
        # for legacy_year in range(1,5):
        #     print('legacy_year',legacy_year)
        #     self.cal_legacy(legacy_year)
        # self.check_legacy()
        self.legacy_plot()


        pass

    def load_df(self):
        df = T.load_df(self.dff)
        return df,self.dff


    def Tree_Ring_to_df(self):
        f = data_root + 'TreeRing\\TRI.xlsx'
        df = pd.read_excel(f)
        tree_dic = {}
        for col in df:
            # print(col)
            tree_dic[col] = np.array(df[col])
        T.save_df(df,self.this_class_arr + 'TreeRing.df')

    def load_Tree_Ring_dic(self):
        dff = self.this_class_arr + 'TreeRing.df'
        df = T.load_df(dff)
        # T.print_head_n(df)
        # exit()
        # tr_num = []
        # for col in df:
        #     tr_num.append(col)
        tr_dic = {}
        for col in df:
            tr_dic[col] = np.array(df[col],dtype=float)
        # for n in tr_dic:
        #     print(n,tr_dic[n])
        #     pause()
        return tr_dic


    def load_drought_events(self):
        dff = self.this_class_arr + 'drought_events.df'
        df = T.load_df(dff)
        return df,dff

    def integrate_lon_lat(self):
        drought_events_df, _ = self.load_drought_events()
        Traits_df,_ = self.load_df()

        spatial_dic = {}
        for i,row in tqdm(Traits_df.iterrows(),total=len(Traits_df)):
            pix = (row.LAT,row.LON)
            spatial_dic[pix] = 1
        arr1 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)

        spatial_dic = {}
        for i, row in tqdm(drought_events_df.iterrows(), total=len(drought_events_df)):
            pix = row.pix
            spatial_dic[pix] = 1
        arr2 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)

        plt.imshow(arr2,cmap='jet')
        plt.imshow(arr1,zorder=99)
        plt.show()
        pass



    def cal_legacy(self,legacy_year):
        # legacy_year = 4
        outdf = self.this_class_arr + 'legacy.df'
        if os.path.isfile(outdf):
            df_void = T.load_df(outdf)
        else:
            df_void = pd.DataFrame()
            T.save_df(df_void,outdf)
        # legacy_year = 3
        spei_dir = data_root + 'SPEI\\per_pix_clean\\'
        spei_dic = {}
        for f in os.listdir(spei_dir):
            dic = T.load_npy(spei_dir + f)
            spei_dic.update(dic)
        drought_event_dir = SPEI_preprocess().this_class_arr + 'drought_events\\'
        event_dic = {}
        for f in os.listdir(drought_event_dir):
            dic = T.load_npy(drought_event_dir + f)
            event_dic.update(dic)
        Traits_df,_ = self.load_df()
        Tree_Ring_dic = self.load_Tree_Ring_dic()

        legacy_df = df_void
        pix_list = []
        legacy_list = []
        drought_event_list = []
        tree_num_list = []
        corr_list = []
        for i,row in tqdm(Traits_df.iterrows(),total=len(Traits_df)):
            pix = (row.LAT,row.LON)
            tree_num = row.NUM
            tree_ring = Tree_Ring_dic[tree_num]
            tree_ring = tree_ring[49:]
            if not pix in spei_dic:
                pix_list.append(np.nan)
                legacy_list.append(np.nan)
                drought_event_list.append(np.nan)
                tree_num_list.append(np.nan)
                corr_list.append(np.nan)
                continue
            spei = spei_dic[pix]
            spei = np.array(spei)
            spei = spei.reshape((int(len(spei)/12),12))
            annual_spei = []
            for s in spei:
                gs_indx = list(range(5,10))
                s_selected = T.pick_vals_from_1darray(s,gs_indx)
                annual_spei.append(np.mean(s_selected))
            # plt.plot(annual_spei,c='r')
            # plt.twinx()
            # plt.plot(tree_ring,c='g')
            # plt.show()
            temp_df = pd.DataFrame()
            temp_df['annual_spei'] = annual_spei
            temp_df['tree_ring'] = tree_ring
            temp_df = temp_df.dropna()
            x = np.array(temp_df['annual_spei']).reshape(-1, 1)
            y = np.array(temp_df['tree_ring'])
            corr_r,p = stats.pearsonr(x.flatten(),y)
            # plt.scatter(x,y)
            # print(corr_r)
            # plt.show()
            # rf = RandomForestRegressor()
            # rf.fit(x, y)

            lr = LinearRegression()
            lr.fit(x, y)

            spei_reshape = np.array(annual_spei).reshape(-1,1)
            # tree_ring_pred = rf.predict(spei_reshape)
            tree_ring_pred = lr.predict(spei_reshape)
            # tree_ring_pred = np.ones_like(spei_reshape)

            # plt.plot(tree_ring_pred)
            # plt.plot(tree_ring)
            # plt.show()

            temp_df1 = pd.DataFrame()
            temp_df1['annual_spei'] = annual_spei
            temp_df1['tree_ring'] = tree_ring
            temp_df1['tree_ring_pred'] = tree_ring_pred
            if not pix in event_dic:
                pix_list.append(np.nan)
                legacy_list.append(np.nan)
                drought_event_list.append(np.nan)
                tree_num_list.append(np.nan)
                corr_list.append(np.nan)
                continue
            events = event_dic[pix]
            for event in events:
                drought_mon = event[0] % 12 + 1
                if not drought_mon in list(range(5,10)):
                    pix_list.append(np.nan)
                    legacy_list.append(np.nan)
                    drought_event_list.append(np.nan)
                    tree_num_list.append(np.nan)
                    corr_list.append(np.nan)
                    continue
                drought_year = event[0]//12
                if drought_year + legacy_year >= len(tree_ring_pred):
                    pix_list.append(np.nan)
                    legacy_list.append(np.nan)
                    drought_event_list.append(np.nan)
                    tree_num_list.append(np.nan)
                    corr_list.append(np.nan)
                    continue
                legacy_start = legacy_year - 1
                legacy_end = legacy_year
                legacy_range = list(range(drought_year + legacy_start,drought_year + legacy_end))
                # print(legacy_range)
                tree_ring_pred_select = T.pick_vals_from_1darray(tree_ring_pred,legacy_range)
                tree_ring_obs_select = T.pick_vals_from_1darray(tree_ring,legacy_range)
                if True in np.isnan(tree_ring_obs_select):
                    legacy_list.append(np.nan)
                else:
                    legacy = np.mean(tree_ring_obs_select - tree_ring_pred_select)
                    legacy_list.append(legacy)
                pix_list.append(pix)
                drought_event_list.append(event)
                tree_num_list.append(tree_num)
                corr_list.append(corr_r)
        legacy_df['tree_num'] = tree_num_list
        legacy_df['pix'] = pix_list
        legacy_df['drought_event'] = drought_event_list
        legacy_df['legacy_year_{}_linear'.format(legacy_year)] = legacy_list
        legacy_df['correlation'.format(legacy_year)] = corr_list
        T.save_df(legacy_df,outdf)


    def check_legacy(self):
        dff = self.this_class_arr + 'legacy.df'
        df = T.load_df(dff)
        # df = df.dropna(how='all')
        df.to_excel(self.this_class_arr + 'legacy.xlsx')
        T.save_df(df,dff)
        # T.print_head_n(df)
        pass

    def legacy_plot(self):
        dff = self.this_class_arr + 'legacy.df'
        df = T.load_df(dff)
        df = df[df['correlation']>0]
        # T.print_head_n(df)

        # exit()

        for legacy_year_ in range(1,5):
            legacy_dic = {}
            for dr in df['drought_event']:
                # print()
                if type(dr) == float:
                    continue
                legacy_dic[dr[0]//12] = []
            mon_list = []
            for i,row in tqdm(df.iterrows(),total=len(df)):
                drought_event = row.drought_event
                if type(drought_event) == float:
                    continue
                drought_start = drought_event[0]
                mon_list.append(drought_start//12)
                legacy = row['legacy_year_{}_linear'.format(legacy_year_)]
                legacy_dic[drought_start//12].append(legacy)

            mon_list = list(set(mon_list))
            mon_list.sort()
            mon_list = list(range(min(mon_list),max(mon_list)))
            # print(mon_list)
            # exit()
            mon_mean = []
            yerr = []
            box = []
            for mon in mon_list:
                if not mon in legacy_dic:
                    mon_mean.append(np.nan)
                    yerr.append(np.nan)
                    box.append(np.nan)
                else:
                    vals = legacy_dic[mon]
                    vals = np.array(vals)
                    mon_mean.append(np.nanmean(vals))
                    yerr.append(np.nanstd(vals))
                    vals_mask = T.remove_np_nan(vals)
                    box.append(vals_mask)
            # print(mon_mean)
            # exit()
            date_list = []
            for i in mon_list:
                mon = i % 12 + 1
                year = i // 12 + 1950
                date_list.append('{}_{}'.format(year,mon))
            # plt.bar(mon_list,mon_mean,label='legacy_year_{}'.format(legacy_year_),alpha=0.5,yerr=yerr)
            plt.figure()
            plt.boxplot(box,showfliers=False)
            mon_mean = SMOOTH().mid_window_smooth(mon_mean,window=3)
            plt.plot(mon_mean)
            # plt.xticks(mon_list[::5],date_list[::5],rotation=90)
            # plt.legend()
            plt.title('legacy_year_{}'.format(legacy_year_))
        plt.show()


        pass


class SPEI_preprocess:

    def __init__(self):
        self.this_class_arr = results_root + 'arr\\SPEI_preprocess\\'
        self.this_class_tif = results_root + 'tif\\SPEI_preprocess\\'
        self.this_class_png = results_root + 'png\\SPEI_preprocess\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
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
        outdir = data_root + 'SPEI\\spei_tif\\'
        T.mk_dir(outdir)
        f = data_root + 'SPEI\\spei03.nc'
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
            year = str(date.year)
            if not int(year) >= 1950:
                continue
            # exit()
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
        fdir = data_root + 'SPEI\\spei_tif\\'
        outdir = data_root + 'SPEI\\per_pix\\'
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
        fdir = data_root + 'SPEI\\per_pix\\'
        outdir = data_root + 'SPEI\\per_pix_clean\\'
        T.mk_dir(outdir)
        for f in tqdm(os.listdir(fdir)):
            dic = T.load_npy(fdir+f)
            clean_dic = {}
            for pix in dic:
                val = dic[pix]
                val = np.array(val,dtype=np.float)
                val[val<-9999]=np.nan
                new_val = T.interp_nan(val,kind='linear')
                if len(new_val) == 1:
                    continue
                # plt.plot(val)
                # plt.show()
                clean_dic[pix] = new_val
            np.save(outdir+f,clean_dic)
        pass
def main():
    # Tree_Ring_preprocess().run()
    Tree_Ring_Legacy().run()
    # SPEI_preprocess().run()
    pass


if __name__ == '__main__':

    main()