import numpy as np
import pandas as pd
import tables
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import lightgbm as lgb
import time
import datetime

from ipdb import set_trace

import gc
import warnings
warnings.filterwarnings("ignore")

from libs.feature_utils import *

print("Load andrew's features...")
file_folder = '../input/andrew_yukawa_features'
andrew_train = pd.read_hdf(file_folder+'/andrew_yukawa_train.h5','df',engine='python')
#andrew_test = pd.read_hdf(file_folder+'/andrew_yukawa_test.h5','df',engine='python')

structures = pd.read_hdf(file_folder+'/ultimate_structures.h5','df',engine='python')

giba_columns, qm9_columns, label_columns, index_columns, diff_columns = get_columns()

print("Load Giba's features...")
train = pd.read_csv('../input/giba/train_giba.csv',
                   usecols=index_columns+giba_columns, engine='python')

#test = pd.read_csv('../input/giba/test_giba.csv',
#                  usecols=index_columns+giba_columns, engine='python')

train = pd.merge(train, andrew_train, how='left', on=index_columns)
#test = pd.merge(test, andrew_test, how='left', on=index_columns)

train = get_features(train, structures)
#test = get_features(test, structures)

print("Load parallel geometory features...")
geo_train = pd.read_csv('../input/geo_features/train_geometric_features.csv',engine='python')
#geo_test = pd.read_csv('../input/geo_features/test_geometric_features.csv',engine='python')

train = pd.concat([train, geo_train],axis=1)
#test = pd.concat([test, geo_test],axis=1)

set_trace()

print("Save Ultimate Dataset...")
train.to_hdf('../input/ultimate/ultimate_train.h5', 'df')
#test.to_hdf('../input/ultimate/ultimate_test.h5', 'df')
