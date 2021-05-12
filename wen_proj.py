# coding=gbk

from __init__ import *

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

        gs = list(range(4,10))

        return gs

    def variables(self,n=3):

        X = [
            'isohydricity',
            'NDVI_pre_{}'.format(n),
            'CSIF_pre_{}'.format(n),
            'VPD_previous_{}'.format(n),
            'TMP_previous_{}'.format(n),
            'PRE_previous_{}'.format(n),
        ]
        Y = 'greenness_loss'
        # Y = 'carbon_loss'
        # Y = 'legacy_1'
        # Y = 'legacy_2'
        # Y = 'legacy_3'

        return X,Y

        pass

    def clean_df(self,df):
        ndvi_valid_f = '/Users/wenzhang/project/drought_legacy_new/results_root_main_flow_2002/arr/NDVI/NDVI_invalid_mask.npy'
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
        print(len(df))
        # exit()
        spatial_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            val = 1
            spatial_dic[pix] = val

        return df,spatial_dic

    def mask_arr_with_NDVI(self, inarr):
        ndvi_valid_f = results_root_main_flow_2002 + 'arr/NDVI/NDVI_invalid_mask.npy'
        ndvi_valid_arr = np.load(ndvi_valid_f)
        grid = np.isnan(ndvi_valid_arr)
        inarr[grid] = np.nan

        pass


        pass

class Make_Dataframe:

    def __init__(self):
        self.this_class_arr = '/Users/wenzhang/project/wen_proj/result/Make_Dataframe/'
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'data_frame.df'

    def run(self):
        # 0 generate a void dataframe
        df = self.__gen_df_init()

        df = self.add_previous_conditions(df)
        df = self.add_MAT_MAP_to_df(df)
        df = self.add_lon_lat_to_df(df)
        df = self.select_max_val_and_pre_length(df)
        df = self.select_max_product(df)
        T.save_df(df,self.dff)
        df = df.dropna()
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


    def add_previous_conditions(self,df):
        fdir = '/Users/wenzhang/project/wen_proj/result/climate_constrain_greenning_SOS 2/'
        for folder in os.listdir(fdir):
            if folder.startswith('.'):
                continue
            for f in os.listdir(os.path.join(fdir,folder)):
                if f.startswith('.'):
                    continue
                arr = to_raster.raster2array(os.path.join(fdir,folder,f))[0]
                arr = np.array(arr)
                arr[arr<-9999]=np.nan
                spatial_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
                var = '{}_{}'.format(folder,f).replace('.tif','')
                val_list = []
                pix_list = []
                for pix in tqdm(spatial_dic,desc=var):
                    pix_list.append(pix)
                    val = spatial_dic[pix]
                    val_list.append(val)
                df['pix'] = pix_list
                df[var] = val_list

                # plt.imshow(arr)
                # plt.show()
        return df

    def add_MAT_MAP_to_df(self,df):
        tif = data_root + 'Climate_408/PRE/MAPRE.tif'
        arr = to_raster.raster2array(tif)[0]
        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if pix in dic:
                tmp_trend = dic[pix]
                if tmp_trend > -999:
                    val_list.append(tmp_trend)
                else:
                    val_list.append(np.nan)

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
                if tmp_trend > -999:
                    val_list.append(tmp_trend)
                else:
                    val_list.append(np.nan)
            else:
                val_list.append(np.nan)
        df['MAT'] = val_list

        return df

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


    def select_max_val_and_pre_length(self,df):
        product_list = [
            'GPCP',
            'LST_mean',
            'PAR',
                        ]
        pre_length_list = [
            15,
            30,
            60,
            90,
        ]

        for product in product_list:
            maxv_list = []
            maxind_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=product):
                pix = row.pix
                val_list = []
                for length in pre_length_list:
                    var = '{}_SOS_pre{}_p'.format(product, length)
                    val = row[var]
                    val_list.append(val)
                if True in np.isnan(val_list):
                    maxv_list.append(np.nan)
                    maxind_list.append(np.nan)
                    continue
                max_v = np.max(val_list)
                max_arg = np.argmax(val_list)
                max_indx = pre_length_list[max_arg]
                maxv_list.append(max_v)
                maxind_list.append(max_indx)
            df['{}_max_value'.format(product)] = maxv_list
            df['{}_max_pre_length'.format(product)] = maxind_list
        return df


    def select_max_product(self,df):
        product_list = [
            'GPCP',
            'LST_mean',
            'PAR',
        ]
        max_product_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            max_var_list = []
            for product in product_list:
                max_var = '{}_max_value'.format(product)
                val = row[max_var]
                max_var_list.append(val)
            if True in max_var_list:
                print(max_var_list)
                exit()
            max_arg = np.argmax(max_var_list)
            max_product = product_list[max_arg]
            max_product_list.append(max_product)
        df['max_corr_product'] = max_product_list
        return df

class Main_flow_shui_re:

    def __init__(self):

        pass

    def run(self):
        # self.plot_MAP()
        # self.plot_matrix()
        # self.plot_scatter()
        self.plot_scatter_pre_n()

        pass

    def __divide_MA(self,arr,min_v=None,max_v=None,step=None,n=None):
        if min_v == None:
            min_v = np.min(arr)
        if max_v == None:
            max_v = np.max(arr)
        if n == None:
            d = np.arange(start=min_v,step=step,stop=max_v)
        if step == None:
            d = np.linspace(min_v,max_v,num=n)

        # print d
        # exit()
        # if step >= 10:
        #     d_str = []
        #     for i in d:
        #         d_str.append('{}'.format(int(round(i*12.,0))))
        # else:
        d_str = []
        for i in d:
            d_str.append('{}'.format(int(round(i, 0))))
        # print d_str
        # exit()
        return d,d_str
        pass


    def plot_matrix(self):
        var_name = 'GPCP_SOS_pre30_p'
        dff = Make_Dataframe().dff
        df = T.load_df(dff)
        df,valid_spatial_dic = Global_vars().clean_df(df)
        # df = df.drop_duplicates(subset=['pix'])
        vals_dic = DIC_and_TIF().void_spatial_dic()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            # sens = row.lag
            var = row[var_name]
            vals_dic[pix].append(var)
        MAT_series = df.MAT
        MAP_series = df.MAP * 12.
        df['MAP'] = MAP_series
        d_mat,mat_str = self.__divide_MA(MAT_series,step=1)
        d_map,map_str = self.__divide_MA(MAP_series,min_v=0,max_v=2001,step=100)
        # print map_str
        # print d_map
        # exit()

        shuire_matrix = []
        x = []
        y = []
        z = []
        for t in tqdm(range(len(d_mat))):
            if t + 1 >= len(d_mat):
                continue
            df_t = df[df['MAT']>d_mat[t]]
            df_t = df_t[df_t['MAT']<d_mat[t+1]]
            temp = []
            for p in range(len(d_map)):
                if p + 1 >= len(d_map):
                    continue
                df_p = df_t[df_t['MAP']>d_map[p]]
                df_p = df_p[df_p['MAP']<d_map[p+1]]
                pixs = df_p.pix

                if len(pixs) != 0:
                    vals = []
                    for pix in pixs:
                        val = vals_dic[pix]
                        val = np.nanmean(val)
                        vals.append(val)
                    val_mean = np.nanmean(vals)
                else:
                    val_mean = np.nan
                temp.append(val_mean)
                x.append(d_map[p])
                y.append(d_mat[t])
                z.append(val_mean)
            shuire_matrix.append(temp)
        # plt.imshow(shuire_matrix,vmin=-0.3,vmax=0.3)
        # plt.imshow(shuire_matrix)
        # plt.xticks(range(len(shuire_matrix[0])),map_str,rotation=90)
        # plt.yticks(range(len(shuire_matrix)),mat_str,rotation=0)

        plt.figure(figsize=(4, 6))
        cmap = 'RdBu_r'
        plt.scatter(x, y, c=z, marker='s', cmap=cmap, norm=None,vmin=-0.3,vmax=0.3)
        plt.gca().invert_yaxis()
        plt.subplots_adjust(
            top=0.88,
            bottom=0.11,
            left=0.12,
            right=0.90,
            hspace=0.2,
            wspace=0.2
        )
        plt.title('Lag (months)')
        plt.colorbar()
        plt.xlabel('MAP (mm)')
        plt.ylabel('MAT (��C)')
        plt.title(var_name)
        plt.show()


    def plot_scatter(self):
        dff = Make_Dataframe().dff
        df = T.load_df(dff)
        df,_ = Global_vars().clean_df(df)

        # max_corr_product
        map = df['MAP']
        mat = df['MAT']
        max_corr_product = df['max_corr_product']
        color_dic = {
            'PAR':'g',
            'LST_mean':'r',
            'GPCP':'b',
        }
        # x,y,z = []
        flag = 0
        xlist = []
        ylist = []
        clist = []
        for x,y,z in tqdm(zip(map,mat,max_corr_product),total=len(map)):
            if np.isnan(x):
                continue
            if np.isnan(y):
                continue
            # print(z)
            if not z in color_dic:
                print(z)
                exit()
                continue
            # flag += 1
            # if not flag % 3 == 0:
            #     continue
            # print(x,y,z)
            # plt.scatter(x,y,c=color_dic[z],s=1,alpha=0.5,marker='+')
            xlist.append(x)
            ylist.append(y)
            clist.append(color_dic[z])
            # plt.show()
            # pause()
        plt.scatter(xlist,ylist,c=clist,s=2,alpha=0.5,marker='+')
        plt.show()

        # print(flag)


        pass
    def plot_scatter_pre_n(self,n=30):
        dff = Make_Dataframe().dff
        df = T.load_df(dff)
        # df,_ = Global_vars().clean_df(df)
        product_list = [
            'GPCP',
            'LST_mean',
            'PAR',
        ]
        color_dic = {
            'PAR': 'g',
            'LST_mean': 'r',
            'GPCP': 'b',
        }
        # max_corr_product
        # map = df['MAP']
        # mat = df['MAT']
        x_list = []
        y_list = []
        c_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            map = row['MAP']
            mat = row['MAT']

            val_list = []
            for product in product_list:
                var_name = '{}_SOS_pre{}_p'.format(product,n)
                val = row[var_name]
                val_list.append(val)
            max_arg = np.argmax(val_list)
            max_product = product_list[max_arg]
            # if max_product == 'GPCP':
            #     continue
            x_list.append(map)
            y_list.append(mat)
            c_list.append(color_dic[max_product])
        plt.scatter(x_list, y_list, c=c_list, s=2, alpha=0.2, marker='+')
        plt.show()

        pass




    def plot_MAP(self):
        dff = Make_Dataframe().dff
        df = T.load_df(dff)
        # df,spatial_dic = Global_vars().clean_df(df)

        map_dic = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            map = row.MAP
            map_dic[pix] = map
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(map_dic)
        # arr = arr * 12.
        arr = arr
        plt.imshow(arr)
        plt.colorbar()
        plt.show()
        pass


class Analysis:

    def __init__(self):

        pass

    def run(self):

        pass

    # def

def main():

    # Make_Dataframe().run()
    Main_flow_shui_re().run()
    pass


if __name__ == '__main__':
    main()


