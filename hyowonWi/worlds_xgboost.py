import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

worlds = pd.read_csv('./worlds.csv')
worlds = worlds.drop([i for i in worlds if 'Champs' in i], axis=1)

minute_list = sorted(list(np.arange(3, 36, step=3)) + list(np.arange(1.5, 34.5, step=3)))

X_worlds = worlds.iloc[:, 1:]  # exclude 0 minute
y_worlds = worlds.iloc[:, 0]

result = []

xgb_worlds_winning_rate = open('./result/xgb_worlds_winning_rate.txt', 'w')

for index, minute in enumerate(minute_list):
     bst = xgb.XGBClassifier()
     if minute <= 30:
         bst.load_model("./result/fitted_models/" + str(minute) + "min_model.bst")
     else:
         bst.load_model("./result/fitted_models/30min_model.bst")

     pred_worlds = bst.predict_proba(X_worlds)[index][1]
     result.append(float(pred_worlds))

     xgb_worlds_winning_rate.write(str(minute) + "min blue team winning rate: " + str(pred_worlds)+ '\n')


xgb_worlds_winning_rate.close()

plt.plot(minute_list, result)
plt.show()