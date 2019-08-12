import numpy as np
import pandas as pd
from libs.loss_utils import logmae

def get_estimators_dict():
    types=['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']
    #estimators=[7000,4000,4000,4000,5000,4000,7000,4000]
    estimators=[7000,7000,7000,7000,7000,7000,7000,7000]
    return dict(zip(types,estimators))

def get_params_dict():
    types=['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']
    p_1JHC = {'num_leaves': 500,
              'objective': 'huber',
              'max_depth': 9,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.9,
              "metric": 'logmae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.1,
              'colsample_bytree': 0.9,
             }
 
    p_2JHH = {'num_leaves': 300,
              'objective': 'huber',
              'max_depth': 7,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 1.0,
              "metric": 'logmae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.1,
              'colsample_bytree': 0.9,
             }
 
    p_1JHN = {'num_leaves': 300,
              'objective': 'huber',
              'max_depth': 7,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 1.0,
              "metric": 'logmae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.1,
              'colsample_bytree': 0.9,
             }
 
    p_2JHN = {'num_leaves': 300,
              'objective': 'huber',
              'max_depth': 7,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 1.0,
              "metric": 'logmae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.1,
              'colsample_bytree': 0.9,
             }

    p_2JHC = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 8,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 1.0,
              "metric": 'logmae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.1,
              'colsample_bytree': 0.9,
             }

    p_3JHH = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 8,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 1.0,
              "metric": 'logmae',
              'reg_alpha': 0.1,
              'reg_lambda': 0.1,
              'colsample_bytree': 0.9,
             }

    p_3JHC = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 7,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 1.0,
              "metric": 'logmae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.1,
              'colsample_bytree': 0.9,
             }

    p_3JHN = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 7,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 1.0,
              "metric": 'logmae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.1,
              'colsample_bytree': 0.9,
             }

    params = [p_1JHC, p_2JHH, p_1JHN, p_2JHN, p_2JHC, p_3JHH, p_3JHC, p_3JHN]

    return dict(zip(types,params))

