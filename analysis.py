# coding=utf-8

from Main_flow import *


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
        self.vars_pairplot()
        pass



    def load_df(self):
        dff = Main_flow_Dataframe().dff
        df = T.load_df(dff)
        T.print_head_n(df)
        return df,dff

    def vars_pairplot(self):

        df,dff = self.load_df()

        df = df[df['lat'] > 30]
        df = df[df['lat'] < 60]
        df = df[df['canopy_height'] > 0]
        # df = df[df['rooting_depth'] > 0]
        df = df[df['gs_sif_spei_corr'] > 0]
        df = df[df['gs_sif_spei_corr_p'] < 0.05]
        # df = df[df['rooting_depth']<30]

        # df = df[df['isohydricity']<0.6]
        df = df[df['isohydricity']>0.9]
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
        legacy1 = df['monthly_legacy_decrease_year_1']
        legacy2 = df['monthly_legacy_decrease_year_2']
        legacy3 = df['monthly_legacy_decrease_year_3']
        isohydricity = df['isohydricity']
        tws_minus_1 = df['TWS_-1']
        tws1 = df['TWS_1']
        tws2 = df['TWS_2']
        tws3 = df['TWS_3']
        is_gs = df['is_gs']
        # new_df['rooting_depth'] = rooting_depth
        new_df['canopy height'] = canopy_height
        new_df['isohydricity'] = isohydricity
        new_df['legacy1'] = legacy1
        new_df['legacy2'] = legacy2
        new_df['legacy3'] = legacy3

        new_df['TWS_-1'] = tws_minus_1
        new_df['TWS_1'] = tws1
        new_df['TWS_2'] = tws2
        new_df['TWS_3'] = tws3
        new_df['is_gs'] = is_gs

        new_df = new_df.dropna()
        sns.pairplot(new_df,markers='.',kind='reg',diag_kind='kde',hue='is_gs')
        # plt.show()
        plt.savefig('test.png')





def main():
    # Correlation_CSIF_SPEI().run()
    Statistic().run()
    pass


if __name__ == '__main__':
    main()