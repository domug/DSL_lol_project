import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import xgboost as xgb

def XGboost_model(minute):
    data = pd.read_csv("C:/Users/위효원/PycharmProjects/DSL_lol_project/jaeyong-song/data_processed/challenger_" + str(minute) + ".csv")

    # Win or lose
    X_features = data.iloc[:,1:]
    y_labels = data.iloc[:,0]


    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=0)

    train_cnt = y_train.count()
    test_cnt = y_test.count()

    xgb_clf = xgb.XGBClassifier(n_estimators=100)

    # hyper-parameter tuning
    xgb_params = {'min_child_weight': [1, 5],
                  'colsample_bytree': [0.6, 0.8],
                  'max_depth': [5, 7]}
    xgb_gridCV = GridSearchCV(estimator=xgb_clf, param_grid=xgb_params, cv=5)
    xgb_gridCV.fit(X_train, y_train)
    xgb_bt_param = xgb_gridCV.best_params_

    # fitting
    xgb_clf = xgb.XGBClassifier(colsample_bytree=xgb_bt_param['colsample_bytree'], max_depth=xgb_bt_param['max_depth'], min_child_weight=xgb_bt_param['min_child_weight'])
    xgb_clf.fit(X_train, y_train)
    xgb_cv_score = cross_val_score(xgb_clf, X_train, y_train, cv=5).mean()
    print(str(minute) + 'trial - XGBoost model cv score: ', xgb_cv_score)