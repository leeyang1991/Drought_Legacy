# coding=gbk

# coding='utf-8'
import sys
version = sys.version_info.major
assert version == 3, 'Python Version Error'
import time
import os
from tqdm import tqdm
import requests
import re
import copyreg
import multiprocessing
import types
from multiprocessing.pool import ThreadPool as TPool




class MULTIPROCESS:
    '''
    可对类内的函数进行多进程并行
    由于GIL，多线程无法跑满CPU，对于不占用CPU的计算函数可用多线程
    并行计算加入进度条
    '''

    def __init__(self, func, params):
        self.func = func
        self.params = params
        copyreg.pickle(types.MethodType, self._pickle_method)
        pass

    def _pickle_method(self, m):
        if m.__self__ is None:
            return getattr, (m.__self__.__class__, m.__func__.__name__)
        else:
            return getattr, (m.__self__, m.__func__.__name__)

    def run(self, process=-9999, process_or_thread='p', **kwargs):
        '''
        # 并行计算加进度条
        :param func: input a kenel_function
        :param params: para1,para2,para3... = params
        :param process: number of cpu
        :param thread_or_process: multi-thread or multi-process,'p' or 't'
        :param kwargs: tqdm kwargs
        :return:
        '''

        if process > 0:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool(process)
            elif process_or_thread == 't':
                pool = TPool(process)
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
            return results
        else:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool()
            elif process_or_thread == 't':
                pool = TPool()
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
            return results


def mk_dir(dir, force=False):
    if not os.path.isdir(dir):
        if force == True:
            os.makedirs(dir)
        else:
            os.mkdir(dir)

def gen_folder_urls(url,root_dir):
    # req = urllib2.Request(url)
    req = requests.request('GET',url)
    html = req.text
    p = re.findall('href=".*?/">', html)
    outdir = root_dir + 'urls/'
    mk_dir(outdir,force=True)
    fw = open(outdir + 'folder_urls.txt','w')
    for pi in p:
        if not 'tile' in pi:
            continue
        tile_name = pi.split('"')[-2]
        url_i = url + tile_name
        fw.write(url_i+'\n')
    fw.close()

    pass


def kernel_gen_tif_urls(params):
    line, out_url_dir = params
    fw = out_url_dir + line.split('/')[-2] + '.txt'
    if os.path.isfile(fw):
        return None
    success = 0
    attempt = 0
    while 1:
        try:
            req = requests.request('GET', line)
            html = req.text
            p = re.findall('href=".*?.tif">', html)
            urls = []
            for pi in p:
                tif_name = pi.split('"')[-2]
                url_i = line + tif_name
                urls.append(url_i)
            success = 1
        except Exception as e:
            attempt += 1
            urls = []
            print(e,'sleep 5s')
            time.sleep(5)
        if success == 1:
            fw = out_url_dir + line.split('/')[-2] + '.txt'
            fw = open(fw,'w')
            content = '\n'.join(urls)+'\n'
            fw.write(content)
            fw.close()
            return None
        if attempt >= 10:
            return None
    pass

def gen_tif_urls(root_dir):
    folder_urls_f = root_dir + 'urls/folder_urls.txt'
    out_url_dir = root_dir + 'urls/tif_urls/'
    mk_dir(out_url_dir,force=True)
    fr = open(folder_urls_f,'r')
    lines = fr.readlines()
    all_url = []
    params = []
    for line in lines:
        line = line.split('\n')[0]
        # kernel_gen_tif_urls([line,out_url_dir])
        params.append([line,out_url_dir])
    MULTIPROCESS(kernel_gen_tif_urls,params).run(process=10,process_or_thread='t')


    pass


def download_i(url,outdir_i):

    # fname = url.split('/')[-1]
    # req = requests.request('GET',url)
    # content = req.content
    # fw = open(outdir_i + fname,'wb')
    # fw.write(content)
    # fw.close()

    #################
    attempt = 0
    while 1:
        try:
            fname = url.split('/')[-1]
            if os.path.isfile(outdir_i + fname):
                print(outdir_i + fname,' is existed')
                return None
            req = requests.request('GET', url)
            content = req.content
            fw = open(outdir_i + fname, 'wb')
            fw.write(content)
            fw.close()
            return None

        except Exception as e:
            print(url, 'error sleep 5s')
            time.sleep(5)
            attempt += 1
        if attempt >= 10:
            return None

def kernel_download_tifs(params):
    outdir,tile,url_dir = params
    outdir_i = outdir + tile.replace('.txt', '') + '/'
    mk_dir(outdir_i)
    fr = open(url_dir + tile, 'r')
    lines = fr.readlines()
    for line in lines:
        line = line.split('\n')[0]
        download_i(line, outdir_i)


def download_tifs(root_dir):
    url_dir = root_dir + 'urls/tif_urls/'
    outdir = root_dir + 'tifs/'
    mk_dir(outdir)
    params = []
    for tile in os.listdir(url_dir):
        params.append([outdir,tile,url_dir])
        # kernel_download_tifs([outdir,tile,url_dir])
    MULTIPROCESS(kernel_download_tifs,params).run(process=50,process_or_thread='t')


def download_invalid_tiles():
    invalid_tiles_txt = '/Volumes/SSD/drought_legacy_new/data/Soilgrids/invalid_tiles.txt'
    outdir = '/Volumes/SSD/drought_legacy_new/data/Soilgrids/tiles/'
    fr = open(invalid_tiles_txt,'r')
    lines = fr.readlines()
    for line in tqdm(lines):
        # print(line)
        line = line.split('\n')[0]
        line_split = line.split('/')
        tif_name = line_split[-1]
        tile_folder = line_split[-2]
        url = 'https://files.isric.org/soilgrids/latest/data/nitrogen/nitrogen_0-5cm_mean/{}/{}'.format(tile_folder,tif_name)
        outdir_i = outdir + tile_folder + '/'
        download_i(url,outdir_i)
    pass


def main():
    # product_list = ['nitrogen','ocs','ocd']
    product_list = ['ocs']
    # product_list = ['bdod','cec','phh2o','sand','soc','clay',]
    layers = ['0-30cm_mean']
    # layers = [
    #     '0-5cm_mean',
    #     '5-15cm_mean',
    #     '15-30cm_mean',
    #     '30-60cm_mean',
    #     '60-100cm_mean',
    #     '100-200cm_mean',
    # ]

    url_list = []
    for p in product_list:
        for l in layers:
            url = 'https://files.isric.org/soilgrids/latest/data/{}/{}_{}/'.format(p,p,l)
            # print(url)
            # print('https://files.isric.org/soilgrids/latest/data/ocs/ocs_0-30cm_mean/')
            url_list.append(url)
    # exit()
    for url in url_list:
        product = url.split('/')[-2]
        root_dir = '/Volumes/SSD/soil_test/{}/'.format(product)
        # root_dir = '/volume5/4T25/soilgrids/{}/'.format(product)
        mk_dir(root_dir,force=True)
        # 1 generate tiles
        gen_folder_urls(url,root_dir)
        # # 2 generate tifs in each tiles
        gen_tif_urls(root_dir)
        # # 3 download tifs via multi-thread
        download_tifs(root_dir)
        # download_invalid_tiles()
        pass


if __name__ == '__main__':

    main()