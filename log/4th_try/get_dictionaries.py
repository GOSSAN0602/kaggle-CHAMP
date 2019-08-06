import numpy as np
import pandas as pd


def get_estimators_dict():
    types=['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']
    estimators=[7000,2500,2500,2500,5000,3500,3000,2500]
    return dict(zip(types,estimators))


def get_params_dict():
    types=['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']
    p_1JHC = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 7,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.8,
              "metric": 'mae',
              "verbosity": -1,
              'lambda_l1': 0.8,
              'lambda_l2': 0.2,
              'feature_fraction': 0.6,
             }
 
    p_2JHH = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 6,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.8,
              "metric": 'mae',
              "verbosity": -1,
              'lambda_l1': 0.8,
              'lambda_l2': 0.2,
              'feature_fraction': 0.6,
             }
 
    p_1JHN = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 5,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.8,
              "metric": 'mae',
              "verbosity": -1,
              'lambda_l1': 0.8,
              'lambda_l2': 0.2,
              'feature_fraction': 0.6,
             }
 
    p_2JHN = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 5,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.8,
              "metric": 'mae',
              "verbosity": -1,
              'lambda_l1': 0.8,
              'lambda_l2': 0.2,
              'feature_fraction': 0.6,
             }

    p_2JHC = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 7,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.8,
              "metric": 'mae',
              "verbosity": -1,
              'lambda_l1': 0.8,
              'lambda_l2': 0.2,
              'feature_fraction': 0.6,
             }

    p_3JHH = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 7,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.8,
              "metric": 'mae',
              "verbosity": -1,
              'lambda_l1': 0.8,
              'lambda_l2': 0.2,
              'feature_fraction': 0.6,
             }

    p_3JHC = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 7,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.8,
              "metric": 'mae',
              "verbosity": -1,
              'lambda_l1': 0.8,
              'lambda_l2': 0.2,
              'feature_fraction': 0.6,
             }

    p_3JHN = {'num_leaves': 400,
              'objective': 'huber',
              'max_depth': 7,
              'learning_rate': 0.12,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.8,
              "metric": 'mae',
              "verbosity": -1,
              'lambda_l1': 0.8,
              'lambda_l2': 0.2,
              'feature_fraction': 0.6,
             }

    params = [p_1JHC, p_2JHH, p_1JHN, p_2JHN, p_2JHC, p_3JHH, p_3JHC, p_3JHN]

    return dict(zip(types,params))

