import numpy as np
from XGboost_model import XGboost_model

myfile = open('./result/XGBoost_score.txt', 'a')

minute_list = sorted(list(np.arange(3, 33, step=3)) + list(np.arange(1.5, 31.5, step=3)))

for min in minute_list:
    XGboost_model(min, myfile)

myfile.close()
