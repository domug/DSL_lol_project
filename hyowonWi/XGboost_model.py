import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import xgboost as xgb


def XGboost_model(minute, file_test, file_val):
    data_train = pd.read_csv("./final_data_processed/Final_Train_Dataset_processed/Train_" + str(minute) + "min.csv").drop(['blueChamps', 'redChamps'], axis=1)
    data_test = pd.read_csv("./final_data_processed/Final_Test_Dataset_processed/Test_" + str(minute) + "min.csv").drop(['blueChamps', 'redChamps'], axis=1)
    data_validation = pd.read_csv("./final_data_processed/Final_Validation_Dataset_processed/Validation_" + str(minute) + "min.csv").drop(['blueChamps', 'redChamps'], axis=1)

    # Win or lose
    X_train = data_train.iloc[:, 2:]
    y_train = data_train.iloc[:, 1]

    X_test = data_test.iloc[:, 2:]
    y_test = data_test.iloc[:,1]

    X_val = data_validation.iloc[:, 2:]
    y_val = data_validation.iloc[:, 1]

    xgb_clf = xgb.XGBClassifier(n_estimators=100)

    # hyper-parameter tuning
    xgb_params = {'min_child_weight': [1, 5], 'colsample_bytree': [0.6, 0.8], 'max_depth': [5, 7]}

    xgb_gridCV = GridSearchCV(estimator=xgb_clf, param_grid=xgb_params, cv=5)
    xgb_gridCV.fit(X_train, y_train)

    xgb_bt_param = xgb_gridCV.best_params_

    # fitting
    xgb_clf = xgb.XGBClassifier(colsample_bytree=xgb_bt_param['colsample_bytree'], max_depth=xgb_bt_param['max_depth'],
                                min_child_weight=xgb_bt_param['min_child_weight'])

    # train
    xgb_clf.fit(X_train, y_train)

    # file.write(str(minute) + 'min trial - XGBoost model cv score: ' + str(xgb_cv_score) + '\n')

    # test
    pred_test = xgb_clf.predict(X_test)
    file_test.write(str(minute) + 'min trial - XGBoost model score wtih test set: ' + str(accuracy_score(y_test, pred_test)) + '\n')


    # validation
    pred_validation = xgb_clf.predict(X_val)
    file_val.write(str(minute) + 'min trial - XGBoost model score with validation set: ' + str(accuracy_score(y_val, pred_validation)) + '\n')


    # feature importance
    # fig = xgb.plot_importance(xgb_clf, max_num_features=20)
    # fig.figure.savefig("./result/feature_importance/" + str(minute) + "min.png", bbox_inches='tight')

    xgb_clf.save_model("./result/fitted_models/" + str(minute) + "min_model.bst")

