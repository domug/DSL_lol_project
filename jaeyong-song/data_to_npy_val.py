import numpy as np
import pandas as pd

# 각 index들에 대해서 시계열 형태로 붙여야함
df_015 = pd.read_csv("./final_data_processed/Val/challenger_1.5.csv")
df_030 = pd.read_csv("./final_data_processed/Val/challenger_3.csv")
df_045 = pd.read_csv("./final_data_processed/Val/challenger_4.5.csv")
df_060 = pd.read_csv("./final_data_processed/Val/challenger_6.csv")
df_075 = pd.read_csv("./final_data_processed/Val/challenger_7.5.csv")
df_090 = pd.read_csv("./final_data_processed/Val/challenger_9.csv")
df_105 = pd.read_csv("./final_data_processed/Val/challenger_10.5.csv")
df_120 = pd.read_csv("./final_data_processed/Val/challenger_12.csv")
df_135 = pd.read_csv("./final_data_processed/Val/challenger_13.5.csv")
df_150 = pd.read_csv("./final_data_processed/Val/challenger_15.csv")
df_165 = pd.read_csv("./final_data_processed/Val/challenger_16.5.csv")
df_180 = pd.read_csv("./final_data_processed/Val/challenger_18.csv")
df_195 = pd.read_csv("./final_data_processed/Val/challenger_19.5.csv")
df_210 = pd.read_csv("./final_data_processed/Val/challenger_21.csv")
df_225 = pd.read_csv("./final_data_processed/Val/challenger_22.5.csv")
df_240 = pd.read_csv("./final_data_processed/Val/challenger_24.csv")
df_255 = pd.read_csv("./final_data_processed/Val/challenger_25.5.csv")
df_270 = pd.read_csv("./final_data_processed/Val/challenger_27.csv")
df_285 = pd.read_csv("./final_data_processed/Val/challenger_28.5.csv")
df_300 = pd.read_csv("./final_data_processed/Val/challenger_30.csv")


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
    temp_df = pd.concat([temp_df, df_030.loc[[i]]])
    temp_df = pd.concat([temp_df, df_045.loc[[i]]])
    temp_df = pd.concat([temp_df, df_060.loc[[i]]])
    temp_df = pd.concat([temp_df, df_075.loc[[i]]])
    temp_df = pd.concat([temp_df, df_090.loc[[i]]])
    temp_df = pd.concat([temp_df, df_105.loc[[i]]])
    temp_df = pd.concat([temp_df, df_120.loc[[i]]])
    temp_df = pd.concat([temp_df, df_135.loc[[i]]])
    temp_df = pd.concat([temp_df, df_150.loc[[i]]])
    temp_df = pd.concat([temp_df, df_165.loc[[i]]])
    temp_df = pd.concat([temp_df, df_180.loc[[i]]])
    temp_df = pd.concat([temp_df, df_195.loc[[i]]])
    temp_df = pd.concat([temp_df, df_210.loc[[i]]])
    temp_df = pd.concat([temp_df, df_225.loc[[i]]])
    temp_df = pd.concat([temp_df, df_240.loc[[i]]])
    temp_df = pd.concat([temp_df, df_255.loc[[i]]])
    temp_df = pd.concat([temp_df, df_270.loc[[i]]])
    temp_df = pd.concat([temp_df, df_285.loc[[i]]])
    temp_df = pd.concat([temp_df, df_300.loc[[i]]])

    dataX.append(temp_df.loc[:, temp_df.columns != 'blueWins'].to_numpy())
    _y = df_015.loc[[i]].copy()
    dataY.append(_y.loc[:, temp_df.columns == 'blueWins'].to_numpy())

valX = np.array(dataX)
valY = np.array(dataY)

np.save('./valX', valX)
np.save('./valY', valY)

