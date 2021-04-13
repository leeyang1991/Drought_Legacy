# coding='utf-8'
import sys
version = sys.version_info.major
assert version == 3, 'Python Version Error'
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import to_raster
import os
# import gdal
# import ogr, osr
from tqdm import tqdm
import datetime
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import copyreg
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import types
from scipy.stats import gamma as gam
import math
import copy
import scipy
import sklearn
import random
import h5py
from netCDF4 import Dataset
import shutil
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from operator import itemgetter
from itertools import groupby
# import RegscorePy
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from mpl_toolkits.mplot3d import Axes3D
import pickle
from dateutil import relativedelta
from sklearn.inspection import permutation_importance
# from statsmodels.stats.outliers_influence import variance_inflation_factor

np.seterr('ignore')

def sleep(t=1):
    time.sleep(t)
def pause():
    input('\33[7m'+"PRESS ENTER TO CONTINUE."+'\33[0m')
# this_root = 'G:\Drought_legacy\\'
# data_root = 'G:\Drought_legacy\\data\\'
# results_root = 'G:\Drought_legacy\\results\\'
# results_root_main_flow = 'G:\Drought_legacy\\main_flow_results\\'

this_root = '/Users/wenzhang/project/Drought_legacy/'
data_root = this_root + 'data/'
results_root = this_root + 'results/'
results_root_main_flow = this_root + 'main_flow_results/'
# from HANTS import *

# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from LY_Tools import *
T = Tools()
D = DIC_and_TIF()
S = SMOOTH()
M = MULTIPROCESS
