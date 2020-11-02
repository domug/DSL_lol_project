import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Random seed to make results deterministic and reproducible
torch.manual_seed(0)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)


# not for lol dataset...
# def minmax_scaler(data):
#     numerator = data - np.min(data, 0)
#     denominator = np.max(data, 0) - np.min(data, 0)
#     return numerator / (denominator + 1e-7)

seq_length = 5 # 30분 경과 후의 예측
data_dim = 56 # feature 갯수
hidden_dim = 20 # 은닉층 길이
output_dim = 1 # true/false
learning_rate = 0.0001
iterations = 10000000

# 우리는 일반적인 시계열 형태가 아닌
# 각 시계열에 해당하는 것이 많은 경기가 존재하는
# 배치가 존재하는 시계열임에 주의하여 진행

# 각 index들에 대해서 시계열 형태로 붙여야함
df_03 = pd.read_csv("./data_processed/challenger_3.csv")
df_06 = pd.read_csv("./data_processed/challenger_6.csv")
df_09 = pd.read_csv("./data_processed/challenger_9.csv")
df_12 = pd.read_csv("./data_processed/challenger_12.csv")
df_15 = pd.read_csv("./data_processed/challenger_15.csv")
# df_18 = pd.read_csv("./data_processed/challenger_18.csv")
# df_21 = pd.read_csv("./data_processed/challenger_21.csv")
# df_24 = pd.read_csv("./data_processed/challenger_24.csv")
# df_27 = pd.read_csv("./data_processed/challenger_27.csv")
# df_30 = pd.read_csv("./data_processed/challenger_30.csv")

print("dataset_shape: ", df_03.shape)
train_set_len = int(df_03.shape[0]*0.7)
test_set_len = df_03.shape[0] - train_set_len

print("Trainset creation...")

# trainX의 각 element가 각 게임의 시간 순으로 feature가 들어가도록 조작해야함...
# dataX = []
# dataY = []
# for i in range(0, train_set_len):
#     if i % int(train_set_len/10) == 0:
#         print("n/10 processed...")
#     temp_df = df_03.loc[[i]].copy()
#     temp_df = pd.concat([temp_df, df_06.loc[[i]]])
#     temp_df = pd.concat([temp_df, df_09.loc[[i]]])
#     temp_df = pd.concat([temp_df, df_12.loc[[i]]])
#     temp_df = pd.concat([temp_df, df_15.loc[[i]]])
#     # temp_df = pd.concat([temp_df, df_18.loc[[i]]])
#     # temp_df = pd.concat([temp_df, df_21.loc[[i]]])
#     # temp_df = pd.concat([temp_df, df_24.loc[[i]]])
#     # temp_df = pd.concat([temp_df, df_27.loc[[i]]])
#     # temp_df = pd.concat([temp_df, df_30.loc[[i]]])
#     dataX.append(temp_df.loc[:, temp_df.columns != 'blueWins'].to_numpy())
#     _y = df_15.loc[[i]].copy()
#     dataY.append(_y.loc[:, temp_df.columns == 'blueWins'].to_numpy())
#
# trainX = np.array(dataX)
# trainY = np.array(dataY)
#
# np.save('/nfs/home/seonbinara/trainX', trainX)
# np.save('/nfs/home/seonbinara/trainY', trainY)

trainX = np.load('/nfs/home/seonbinara/trainX.npy')
trainY = np.load('/nfs/home/seonbinara/trainY.npy')

print("Testset creation...")

# todo - 나중에 데이터 인덱스 문제 반드시 수정!!!

# dataX = []
# dataY = []
# # for i in range(train_set_len, df_03.shape[0]):
# for i in range(train_set_len, 14106):
#     if i % int(test_set_len/10) == 0:
#         print("n/10 processed...")
#     temp_df = df_03.loc[[i]].copy()
#     temp_df = pd.concat([temp_df, df_06.loc[[i]]])
#     temp_df = pd.concat([temp_df, df_09.loc[[i]]])
#     temp_df = pd.concat([temp_df, df_12.loc[[i]]])
#     temp_df = pd.concat([temp_df, df_15.loc[[i]]])
#     # temp_df = pd.concat([temp_df, df_18.loc[[i]]])
#     # temp_df = pd.concat([temp_df, df_21.loc[[i]]])
#     # temp_df = pd.concat([temp_df, df_24.loc[[i]]])
#     # temp_df = pd.concat([temp_df, df_27.loc[[i]]])
#     # temp_df = pd.concat([temp_df, df_30.loc[[i]]])
#     dataX.append(temp_df.loc[:, temp_df.columns != 'blueWins'].to_numpy())
#     _y = df_15.loc[[i]].copy()
#     dataY.append(_y.loc[:, temp_df.columns == 'blueWins'].to_numpy())
#
# testX = np.array(dataX)
# testY = np.array(dataY)
#
# np.save('/nfs/home/seonbinara/testX', testX)
# np.save('/nfs/home/seonbinara/testY', testY)

testX = np.load('/nfs/home/seonbinara/testX.npy')
testY = np.load('/nfs/home/seonbinara/testY.npy')


trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)

print("trainX shape: ", trainX_tensor.shape)
print("trainY shape: ", trainY_tensor.shape)

testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

print("testX shape: ", testX_tensor.shape)
print("testY shape: ", testY_tensor.shape)

print("Data creation Finished....")

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:, -1])
        x = torch.tanh(x)
        return x

net = Net(data_dim, hidden_dim, output_dim, 1).to(device)

# loss & optimizer setting
criterion = torch.nn.BCELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

print("Training started!!!")

for i in range(iterations):

    optimizer.zero_grad()
    outputs = net(trainX_tensor.to(device))
    loss = criterion(outputs, trainY_tensor.to(device))
    loss.backward()
    optimizer.step()
    if i%1000 == 0:
        print(i, loss.item())

print("Training finished!!!")

# print(net(testX_tensor).data.numpy())
# print(testY_tensor.numpy())

predY = net(testX_tensor.to(device)).to('cpu').data.flatten()
for i in range(0, len(predY)):
    if predY[i] >= 0:
        predY[i] = 1
    else:
        predY[i] = 0

correct = int(sum(predY == testY_tensor.flatten()))
total = len(predY)
print("Total: ", total)
print("Correct: ", correct)
print("Accuracy: ", correct/total*100, "%")

# plt.plot(testY.flatten())
# plt.plot(predY)
# plt.legend(['original', 'prediction'])
# plt.show()