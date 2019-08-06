import numpy as np

def logmae(y_pred, y_true):
    return np.log(np.max(np.abs(y_true - y_pred),1e-9))
    
