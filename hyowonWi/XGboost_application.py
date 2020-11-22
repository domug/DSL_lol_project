# import numpy as np
# from XGboost_model import XGboost_model
#
# myfile_test = open('result/XGBoost_score_test.txt', 'w')
# myfile_validation = open('result/XGBoost_score_validation.txt', 'w')
#
# minute_list = sorted(list(np.arange(3, 33, step=3)) + list(np.arange(1.5, 31.5, step=3)))
#
# for min in minute_list:
#      XGboost_model(min, myfile_test, myfile_validation)
#
# myfile_test.close()
# myfile_validation.close()
