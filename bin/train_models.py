import numpy as np
import pandas as pd
import os
import sys
sys.path.append('./')

import subprocess

from libs.andrew_utils import *
from libs.feature_utils import get_columns, get_feature_columns
from libs.get_dictionaries import *

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn import metrics
from sklearn import linear_model
import gc
import warnings
warnings.filterwarnings("ignore")
from ipdb import set_trace

import argparse


parser = argparse.ArgumentParser(description='Training Mol Models')

parser.add_argument('tag')
parser.add_argument('--model_type', type=str, default = 'lgb', help='choose model')

args = parser.parse_args()

os.mkdir('log/'+args.tag)

feature_columns = get_feature_columns(0.1)
#cmd = "script ./log/"+args.tag+"/loss.log"
#subprocess.call(cmd.split())

file_folder = '../input/ultimate'
print('Load Train data')
train = pd.read_hdf(file_folder+'/ultimate_train.h5', 'df', engine='python')
print('Load Test data')
test = pd.read_hdf(file_folder+'/ultimate_test.h5', 'df', engine='python')
sub = pd.read_csv('../input/champsdata/sample_submission.csv')


y = train['scalar_coupling_constant']
train = train.drop(columns = ['scalar_coupling_constant'])

giba_columns, qm9_columns, label_columns, index_columns, diff_columns = get_columns()

print("Encoding label features...")
for f in label_columns:
    # 'type' has to be the last one
    # since the this label encoder is used later
    if f in train.columns:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
train = train.drop(['id','molecule_name'],axis=1)
test = test.drop(['id','molecule_name'],axis=1)
n_fold = 2
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

param_dict = get_params_dict()
n_estimators_dict = get_estimators_dict()
f = open('log/'+args.tag+'/param.txt', 'w')
for key,value in sorted(param_dict.items()):
    f.write(f'{key} {value}\n')
f.close()

f = open('log/'+args.tag+'/n_estimators.txt', 'w')
for key,value in sorted(n_estimators_dict.items()):
    f.write(f'{key} {value}\n')
f.close()

X_short = pd.DataFrame({'ind': list(train.index), 'type': train['type'].values, 'oof': [0] * len(train), 'target': y.values})
X_short_test = pd.DataFrame({'ind': list(test.index), 'type': test['type'].values, 'prediction': [0] * len(test)})

for t in train['type'].unique():
    type_ = lbl.inverse_transform([t])[0]
    print('Training of type '+str(type_))
    X_t = train.loc[train['type'] == t]
    X_test_t = test.loc[test['type'] == t]
    y_t = X_short.loc[X_short['type'] == t, 'target']
    params = param_dict[str(type_)]
    n_estimators = n_estimators_dict[type_]

    # select features for each type
    X_t = X_t.loc[:,feature_columns[str(type_)]]
    X_test_t = X_test_t.loc[:,feature_columns[str(type_)]]
    result_dict_lgb = train_model_regression(args.tag, type_, X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type=args.model_type, eval_metric='group_mae', plot_feature_importance=True,
                                                      verbose=500, early_stopping_rounds=200, n_estimators=n_estimators)
    X_short.loc[X_short['type'] == t, 'oof'] = result_dict_lgb['oof']
    X_short_test.loc[X_short_test['type'] == t, 'prediction'] = result_dict_lgb['prediction']
    result_dict_lgb['feature_importance'].to_csv('log/'+args.tag+'/'+str(type_)+'feature_importance.csv')
sub['scalar_coupling_constant'] = X_short_test['prediction']
sub.to_csv('log/' + args.tag +'/submission.csv', index=False)

plot_data = pd.DataFrame(y)
plot_data.index.name = 'id'
plot_data['yhat'] = X_short['oof']
plot_data['type'] = lbl.inverse_transform(X_short['type'])

def plot_oof_preds(ctype, llim, ulim):
        plt.figure(figsize=(6,6))
        sns.scatterplot(x='scalar_coupling_constant',y='yhat',
                        data=plot_data.loc[plot_data['type']==ctype,
                        ['scalar_coupling_constant', 'yhat']]);
        plt.xlim((llim, ulim))
        plt.ylim((llim, ulim))
        plt.plot([llim, ulim], [llim, ulim])
        plt.xlabel('scalar_coupling_constant')
        plt.ylabel('predicted')
        plt.title(str(ctype), fontsize=18)
        plt.savefig('log/'+args.tag+'/'+str(ctype)+'.png')
plot_oof_preds('1JHC', 0, 250)
plot_oof_preds('1JHN', 0, 100)
plot_oof_preds('2JHC', -50, 50)
plot_oof_preds('2JHH', -50, 50)
plot_oof_preds('2JHN', -25, 25)
plot_oof_preds('3JHC', -25, 100)
plot_oof_preds('3JHH', -20, 20)
plot_oof_preds('3JHN', -15, 15)

