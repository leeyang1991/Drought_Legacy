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

def main():
    Tree_Ring_preprocess().run()
    pass


if __name__ == '__main__':

    main()