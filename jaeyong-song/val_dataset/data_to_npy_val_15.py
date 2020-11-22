import numpy as np
import pandas as pd

# 각 index들에 대해서 시계열 형태로 붙여야함
df_015 = pd.read_csv("../final_data_processed/Val/challenger_1.5.csv")


print("dataset_shape: ", df_015.shape)
val_set_len = int(df_015.shape[0])

print("Valset creation...")

# valX의 각 element가 각 게임의 시간 순으로 feature가 들어가도록 조작해야함...
dataX = []
dataY = []
for i in range(0, val_set_len):
    if i % int(val_set_len/10) == 0:
        print("n/10 processed...")
    temp_df = df_015.loc[[i]].copy()

    dataX.append(temp_df.loc[:, temp_df.columns != 'blueWins'].to_numpy())
    _y = df_015.loc[[i]].copy()
    dataY.append(_y.loc[:, temp_df.columns == 'blueWins'].to_numpy())

valX = np.array(dataX)
valY = np.array(dataY)

np.save('./valX_15', valX)
np.save('./valY_15', valY)

