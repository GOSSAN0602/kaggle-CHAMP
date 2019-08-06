import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import lightgbm as lgb
import time
import datetime

import gc
import warnings
warnings.filterwarnings("ignore")


#def select_feature():
    

def map_atom_info(df_1, df_2, atom_idx):
    df = pd.merge(df_1, df_2, how = 'left',
                  left_on  = ['molecule_name', 'atom_index_'+str(atom_idx)],
                  right_on = ['molecule_name',  'atom_index'])

    df = df.drop('atom_index', axis=1)
    return df


def find_dist(df):
    df_p_0 = df[['x_0', 'y_0', 'z_0']].values
    df_p_1 = df[['x_1', 'y_1', 'z_1']].values

    df['dist'] = np.linalg.norm(df_p_0 - df_p_1, axis=1)
    df['dist_inv2'] = 1/df['dist']**2
    df['dist_x'] = (df['x_0'] - df['x_1']) ** 2
    df['dist_y'] = (df['y_0'] - df['y_1']) ** 2
    df['dist_z'] = (df['z_0'] - df['z_1']) ** 2
    return df

def find_closest_atom(df):
    df_temp = df.loc[:,["molecule_name",
                      "atom_index_0","atom_index_1",
                      "dist","x_0","y_0","z_0","x_1","y_1","z_1"]].copy()
    df_temp_ = df_temp.copy()
    df_temp_ = df_temp_.rename(columns={'atom_index_0': 'atom_index_1',
                                       'atom_index_1': 'atom_index_0',
                                       'x_0': 'x_1',
                                       'y_0': 'y_1',
                                       'z_0': 'z_1',
                                       'x_1': 'x_0',
                                       'y_1': 'y_0',
                                       'z_1': 'z_0'})
    df_temp_all = pd.concat((df_temp,df_temp_),axis=0)

    df_temp_all["min_distance"]=df_temp_all.groupby(['molecule_name',
                                                     'atom_index_0'])['dist'].transform('min')
    df_temp_all["max_distance"]=df_temp_all.groupby(['molecule_name',
                                                     'atom_index_0'])['dist'].transform('max')

    df_temp = df_temp_all[df_temp_all["min_distance"]==df_temp_all["dist"]].copy()
    df_temp = df_temp.drop(['x_0','y_0','z_0','min_distance'], axis=1)
    df_temp = df_temp.rename(columns={'atom_index_0': 'atom_index',
                                         'atom_index_1': 'atom_index_closest',
                                         'dist': 'distance_closest',
                                         'x_1': 'x_closest',
                                         'y_1': 'y_closest',
                                         'z_1': 'z_closest'})
    df_temp = df_temp.drop_duplicates(subset=['molecule_name', 'atom_index'])

    for atom_idx in [0,1]:
        df = map_atom_info(df,df_temp, atom_idx)
        df = df.rename(columns={'atom_index_closest': 'atom_index_closest_'+str(atom_idx),
                                        'distance_closest': 'distance_closest_'+str(atom_idx),
                                        'x_closest': 'x_closest_'+str(atom_idx),
                                        'y_closest': 'y_closest_'+str(atom_idx),
                                        'z_closest': 'z_closest_'+str(atom_idx)})

    df_temp= df_temp_all[df_temp_all["max_distance"]==df_temp_all["dist"]].copy()
    df_temp = df_temp.drop(['x_0','y_0','z_0','max_distance'], axis=1)
    df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',
                                         'atom_index_1': 'atom_index_farthest',
                                         'dist': 'distance_farthest',
                                         'x_1': 'x_farthest',
                                         'y_1': 'y_farthest',
                                         'z_1': 'z_farthest'})
    df_temp = df_temp.drop_duplicates(subset=['molecule_name', 'atom_index'])

    for atom_idx in [0,1]:
        df = map_atom_info(df,df_temp, atom_idx)
        df = df.rename(columns={'atom_index_farthest': f'atom_index_farthest_'+str(atom_idx),
                                        'distance_farthest': 'distance_farthest_'+str(atom_idx),
                                        'x_farthest': 'x_farthest_'+str(atom_idx),
                                        'y_farthest': 'y_farthest_'+str(atom_idx),
                                        'z_farthest': 'z_farthest_'+str(atom_idx)})
    return df


def add_cos_features(df):

    df["distance_center0"] = np.sqrt((df['x_0']-df['c_x'])**2 \
                                   + (df['y_0']-df['c_y'])**2 \
                                   + (df['z_0']-df['c_z'])**2)
    df["distance_center1"] = np.sqrt((df['x_1']-df['c_x'])**2 \
                                   + (df['y_1']-df['c_y'])**2 \
                                   + (df['z_1']-df['c_z'])**2)

    df['distance_c0'] = np.sqrt((df['x_0']-df['x_closest_0'])**2 + \
                                (df['y_0']-df['y_closest_0'])**2 + \
                                (df['z_0']-df['z_closest_0'])**2)
    df['distance_c1'] = np.sqrt((df['x_1']-df['x_closest_1'])**2 + \
                                (df['y_1']-df['y_closest_1'])**2 + \
                                (df['z_1']-df['z_closest_1'])**2)

    df["distance_f0"] = np.sqrt((df['x_0']-df['x_farthest_0'])**2 + \
                                (df['y_0']-df['y_farthest_0'])**2 + \
                                (df['z_0']-df['z_farthest_0'])**2)
    df["distance_f1"] = np.sqrt((df['x_1']-df['x_farthest_1'])**2 + \
                                (df['y_1']-df['y_farthest_1'])**2 + \
                                (df['z_1']-df['z_farthest_1'])**2)

    vec_center0_x = (df['x_0']-df['c_x'])/(df["distance_center0"]+1e-10)
    vec_center0_y = (df['y_0']-df['c_y'])/(df["distance_center0"]+1e-10)
    vec_center0_z = (df['z_0']-df['c_z'])/(df["distance_center0"]+1e-10)

    vec_center1_x = (df['x_1']-df['c_x'])/(df["distance_center1"]+1e-10)
    vec_center1_y = (df['y_1']-df['c_y'])/(df["distance_center1"]+1e-10)
    vec_center1_z = (df['z_1']-df['c_z'])/(df["distance_center1"]+1e-10)

    vec_c0_x = (df['x_0']-df['x_closest_0'])/(df["distance_c0"]+1e-10)
    vec_c0_y = (df['y_0']-df['y_closest_0'])/(df["distance_c0"]+1e-10)
    vec_c0_z = (df['z_0']-df['z_closest_0'])/(df["distance_c0"]+1e-10)

    vec_c1_x = (df['x_1']-df['x_closest_1'])/(df["distance_c1"]+1e-10)
    vec_c1_y = (df['y_1']-df['y_closest_1'])/(df["distance_c1"]+1e-10)
    vec_c1_z = (df['z_1']-df['z_closest_1'])/(df["distance_c1"]+1e-10)

    vec_f0_x = (df['x_0']-df['x_farthest_0'])/(df["distance_f0"]+1e-10)
    vec_f0_y = (df['y_0']-df['y_farthest_0'])/(df["distance_f0"]+1e-10)
    vec_f0_z = (df['z_0']-df['z_farthest_0'])/(df["distance_f0"]+1e-10)

    vec_f1_x = (df['x_1']-df['x_farthest_1'])/(df["distance_f1"]+1e-10)
    vec_f1_y = (df['y_1']-df['y_farthest_1'])/(df["distance_f1"]+1e-10)
    vec_f1_z = (df['z_1']-df['z_farthest_1'])/(df["distance_f1"]+1e-10)

    vec_x = (df['x_1']-df['x_0'])/df['dist']
    vec_y = (df['y_1']-df['y_0'])/df['dist']
    vec_z = (df['z_1']-df['z_0'])/df['dist']

    df["cos_c0_c1"] = vec_c0_x*vec_c1_x + vec_c0_y*vec_c1_y + vec_c0_z*vec_c1_z
    df["cos_f0_f1"] = vec_f0_x*vec_f1_x + vec_f0_y*vec_f1_y + vec_f0_z*vec_f1_z

    df["cos_c0_f0"] = vec_c0_x*vec_f0_x + vec_c0_y*vec_f0_y + vec_c0_z*vec_f0_z
    df["cos_c1_f1"] = vec_c1_x*vec_f1_x + vec_c1_y*vec_f1_y + vec_c1_z*vec_f1_z

    df["cos_center0_center1"] = vec_center0_x*vec_center1_x \
                              + vec_center0_y*vec_center1_y \
                              + vec_center0_z*vec_center1_z

    df["cos_c0"] = vec_c0_x*vec_x + vec_c0_y*vec_y + vec_c0_z*vec_z
    df["cos_c1"] = vec_c1_x*vec_x + vec_c1_y*vec_y + vec_c1_z*vec_z

    df["cos_f0"] = vec_f0_x*vec_x + vec_f0_y*vec_y + vec_f0_z*vec_z
    df["cos_f1"] = vec_f1_x*vec_x + vec_f1_y*vec_y + vec_f1_z*vec_z

    df["cos_center0"] = vec_center0_x*vec_x + vec_center0_y*vec_y + vec_center0_z*vec_z
    df["cos_center1"] = vec_center1_x*vec_x + vec_center1_y*vec_y + vec_center1_z*vec_z

    return df


def dummies(df, list_cols):
    for col in list_cols:
        df_dummies = pd.get_dummies(df[col], drop_first=True,
                                    prefix=(str(col)))
        df = pd.concat([df, df_dummies], axis=1)
    return df


def add_qm9_features(df):
    data_qm9 = pd.read_pickle('../input/qm9/data.covs.pickle')
    to_drop = ['type',
               'linear',
               'atom_index_0',
               'atom_index_1',
               'scalar_coupling_constant',
               'U', 'G', 'H',
               'mulliken_mean', 'r2', 'U0']
    data_qm9 = data_qm9.drop(columns = to_drop, axis=1)
    df = pd.merge(df, data_qm9, how='left', on=['molecule_name','id'])
    del data_qm9

    df = dummies(df, ['type', 'atom_1'])
    return df

def get_features(df, struct):
    print("find dist...")
    df = find_dist(df)
    print("find closest atom...")
    df = find_closest_atom(df)
    print("add cos features")
    df = add_cos_features(df)
    print("add qm9")
    df = add_qm9_features(df)
    return df

def get_columns():
    giba_columns = ['inv_dist0', 'inv_dist1', 'inv_distP', 'inv_dist0R', 'inv_dist1R', 'inv_distPR',
    'inv_dist0E', 'inv_dist1E', 'inv_distPE', 'linkM0', 'linkM1',
    'min_molecule_atom_0_dist_xyz',
    'mean_molecule_atom_0_dist_xyz',
    'max_molecule_atom_0_dist_xyz',
    'sd_molecule_atom_0_dist_xyz',
    'min_molecule_atom_1_dist_xyz',
    'mean_molecule_atom_1_dist_xyz',
    'max_molecule_atom_1_dist_xyz',
    'sd_molecule_atom_1_dist_xyz',
    'coulomb_C.x', 'coulomb_F.x', 'coulomb_H.x', 'coulomb_N.x', 'coulomb_O.x',
    'yukawa_C.x', 'yukawa_F.x', 'yukawa_H.x', 'yukawa_N.x', 'yukawa_O.x',
    'vander_C.x', 'vander_F.x', 'vander_H.x', 'vander_N.x', 'vander_O.x',
    'coulomb_C.y', 'coulomb_F.y', 'coulomb_H.y', 'coulomb_N.y', 'coulomb_O.y',
    'yukawa_C.y', 'yukawa_F.y', 'yukawa_H.y', 'yukawa_N.y', 'yukawa_O.y',
    'vander_C.y', 'vander_F.y', 'vander_H.y', 'vander_N.y', 'vander_O.y',
    'distC0', 'distH0', 'distN0', 'distC1', 'distH1', 'distN1',
    'adH1', 'adH2', 'adH3', 'adH4', 'adC1', 'adC2', 'adC3', 'adC4',
    'adN1', 'adN2', 'adN3', 'adN4',
    'NC', 'NH', 'NN', 'NF', 'NO']

    qm9_columns = [
    'rc_A', 'rc_B', 'rc_C',
    'mu', 'alpha',
    'homo','lumo', 'gap',
    'zpve', 'Cv',
    'freqs_min', 'freqs_max', 'freqs_mean',
    'mulliken_min', 'mulliken_max',
    'mulliken_atom_0', 'mulliken_atom_1'
    ]

    label_columns = ['molecule_name',
    'atom_index_0', 'atom_index_1','atom_0','atom_1', 'type_0',
    'structure_atom_0','structure_atom_1','type']

    index_columns = ['type','molecule_name','id']

    diff_columns = ['Cv',
    'alpha', 'freqs_max', 'freqs_mean', 'freqs_min',
    'gap', 'homo', 'linkM0',
    'lumo', 'mu', 'mulliken_atom_0', 'mulliken_max', 'mulliken_min',
    'rc_A', 'rc_B', 'rc_C', 'sd_molecule_atom_1_dist_xyz', 'zpve']

    return giba_columns, qm9_columns, label_columns, index_columns, diff_columns
