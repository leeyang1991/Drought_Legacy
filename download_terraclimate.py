# coding=utf-8
import requests
import os
import codecs
import time
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import analysis
from __init__ import *

year = []
for y in range(2009,2019):
    year.append(str(y))

def download(y):
    # outdir = r'D:\project05\PET\download\\'
    outdir = '/Users/wenzhang/Desktop/PET_terra/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_pet_{}.nc'.format(y)
    print(url)
    while 1:
        try:
            req = requests.request('GET', url)
            content = req.content
            fw = open(outdir+'pet_{}.nc'.format(y), 'wb')
            fw.write(content)
            return None
        except Exception as e:
            print(url, 'error sleep 5s')
            time.sleep(5)


MULTIPROCESS(download,year).run(process=5,process_or_thread='t')