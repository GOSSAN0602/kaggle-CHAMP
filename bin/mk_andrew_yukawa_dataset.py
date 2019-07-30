import os
import time
import datetime
import json
import gc
from numba import jit
import sys
sys.path.append('./')
from libs.andrew_utils import *
from libs.brute_force_utils import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import altair as alt
from altair.vega import v5

import tables

file_folder = '../input/champsdata'
#train = pd.read_csv(file_folder + '/train.csv', engine='python')
test = pd.read_csv(file_folder + '/test.csv', engine='python')
structures = pd.read_csv(file_folder + '/structures.csv', engine='python')
yukawa_str = pd.read_csv('../input/yukawa/structures_yukawa.csv', engine='python')

# concat structures
structures = pd.concat([structures, yukawa_str], axis=1)
del yukawa_str

# Basic feature engineering
print("basic feature engineering")
#train = basic_feature_engineering(train, structures)
test = basic_feature_engineering(test, structures)

#distances
#train_p_0 = train[['x_0', 'y_0', 'z_0']].values
#train_p_1 = train[['x_1', 'y_1', 'z_1']].values
test_p_0 = test[['x_0', 'y_0', 'z_0']].values
test_p_1 = test[['x_1', 'y_1', 'z_1']].values

#train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
#train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
#train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
#train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

#train['type_0'] = train['type'].apply(lambda x: x[0])
test['type_0'] = test['type'].apply(lambda x: x[0])

#print("create full features --train--")
#train = create_features_full(train)
print("create full features --test--")
test = create_features_full(test)

print("saving... ")
#train.to_hdf('andrew_yukawa_train.h5','df')
test.to_hdf('andrew_yukawa_test.h5','df')
#structures.to_hdf('ultimate_structures.h5','df')
