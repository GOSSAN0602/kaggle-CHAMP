import os
import time
import datetime
import json
import gc
from numba import jit
import sys

sys.path.append('./')
from libs.andrew_utils import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import altair as alt
from altair.vega import v5


def map_atom_info(df_1, df_2, atom_idx):
    df = pd.merge(df_1, df_2, how = 'left',
                  left_on  = ['molecule_name', 'atom_index_'+str(atom_idx)],
                  right_on = ['molecule_name',  'atom_index'])

    df = df.drop('atom_index', axis=1)
    return df

def basic_feature_engineering(df, struct):
    for atom_idx in [0,1]:
        df = map_atom_info(df, struct, atom_idx)
        df = df.rename(columns={'atom': 'atom_'+str(atom_idx),
                            'x': 'x_'+str(atom_idx),
                            'y': 'y_'+str(atom_idx),
                            'z': 'z_'+str(atom_idx)})
        struct['c_x'] = struct.groupby('molecule_name')['x'].transform('mean')
        struct['c_y'] = struct.groupby('molecule_name')['y'].transform('mean')
        struct['c_z'] = struct.groupby('molecule_name')['z'].transform('mean')
    return df

def create_features_full(df):
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['molecule_dist_std'] = df.groupby('molecule_name')['dist'].transform('std')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    num_cols = ['x_1', 'y_1', 'z_1', 'dist', 'dist_x', 'dist_y', 'dist_z']
    cat_cols = ['atom_index_0', 'atom_index_1', 'type', 'atom_1', 'type_0']
    aggs = ['mean', 'max', 'std', 'min']
    for col in cat_cols:
        df['molecule_'+str(col)+'_count'] = df.groupby('molecule_name')[col].transform('count')

    for cat_col in tqdm(cat_cols):
        for num_col in num_cols:
            for agg in aggs:
                df['molecule_'+str(cat_col)+'_'+str(num_col)+'_'+str(agg)] = df.groupby(['molecule_name', cat_col])[num_col].transform(agg)
                df['molecule_'+str(cat_col)+'_'+str(num_col)+'_'+str(agg)+'_diff'] = df['molecule_'+str(cat_col)+'_'+str(num_col)+'_'+str(agg)] - df[num_col]
                df['molecule_'+str(cat_col)+'_'+str(num_col)+'_'+str(agg)+'_div'] = df['molecule_'+str(cat_col)+'_'+str(num_col)+'_'+str(agg)] / df[num_col]

    return df
