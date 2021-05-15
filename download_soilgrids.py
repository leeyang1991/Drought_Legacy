# coding=gbk

import re
import requests
from __init__ import *

root_dir = '/Users/wenzhang/project/soilgrids/'


def mk_dir(dir, force=False):
    if not os.path.isdir(dir):
        if force == True:
            os.makedirs(dir)
        else:
            os.mkdir(dir)

def gen_folder_urls():
    # req = urllib2.Request(url)
    url = 'https://files.isric.org/soilgrids/latest/data/nitrogen/nitrogen_0-5cm_mean/'
    # url = 'https://detail.zol.com.cn/'
    req = requests.request('GET',url)
    html = req.text
    p = re.findall('href=".*?/">', html)
    outdir = root_dir + 'urls/'
    mk_dir(outdir)
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
    success = 0
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
    pass

def gen_tif_urls():
    folder_urls_f = root_dir + 'urls/folder_urls.txt'
    out_url_dir = root_dir + 'urls/tif_urls/'
    mk_dir(out_url_dir)
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
    while 1:
        try:
            fname = url.split('/')[-1]
            req = requests.request('GET', url)
            content = req.content
            fw = open(outdir_i + fname, 'wb')
            fw.write(content)
            fw.close()
            return None

        except Exception as e:
            print(url, 'error sleep 5s')
            time.sleep(5)

def kernel_download_tifs(params):
    outdir,tile,url_dir = params
    outdir_i = outdir + tile.replace('.txt', '') + '/'
    T.mk_dir(outdir_i)
    fr = open(url_dir + tile, 'r')
    lines = fr.readlines()
    for line in lines:
        line = line.split('\n')[0]
        download_i(line, outdir_i)


def download_tifs():
    url_dir = root_dir + 'urls/tif_urls/'
    outdir = root_dir + 'tifs/'
    T.mk_dir(outdir)
    params = []
    for tile in os.listdir(url_dir):
        params.append([outdir,tile,url_dir])
        # kernel_download_tifs([outdir,tile,url_dir])
    MULTIPROCESS(kernel_download_tifs,params).run(process=50,process_or_thread='t')


def main():
    # 1 generate tiles
    # gen_folder_urls()
    # 2 generate tifs in each tiles
    # gen_tif_urls()
    # 3 download tifs via multi-thread
    download_tifs()
    pass


if __name__ == '__main__':

    main()