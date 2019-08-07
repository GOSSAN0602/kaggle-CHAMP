import pandas as pd
import numpy as np
from ipdb import set_trace

cut_rate = 0.7

log_path = 'log/feature_search2/'
ctype_list = ['1JHC','2JHC','2JHN','1JHN','2JHH','3JHC','3JHH','3JHN']

f_list_1 = []
f_list_2 = []

for t in list(ctype_list):
    print('read '+t+' feature')
    f_ = pd.read_csv(log_path+t+'feature_importance.csv')
    threshold = int(f_.shape[0]/2*0.7)
    f_1 = f_[f_['fold']==1].drop(['Unnamed: 0','fold'],axis=1).sort_values(by='importance',ascending=False)[:threshold]
    f_2 = f_[f_['fold']==2].drop(['Unnamed: 0','fold'],axis=1).sort_values(by='importance',ascending=False)[:threshold]
    f_list_1.append(f_1)
    f_list_2.append(f_2)

columns_list = []
for i in range(len(f_list_1)):
    use_columns = f_list_1[i]["features"].values.tolist()
    columns_list.append(use_columns)

dict(zip(ctype_list, columns_list))

