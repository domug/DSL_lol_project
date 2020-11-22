import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def runner(finalIdx):
    # Random seed to make results deterministic and reproducible
    torch.manual_seed(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)


    # not for lol dataset...
    # def minmax_scaler(data):
    #     numerator = data - np.min(data, 0)
    #     denominator = np.max(data, 0) - np.min(data, 0)
    #     return numerator / (denominator + 1e-7)

    seq_length = 10 # 30분 경과 후의 예측
    data_dim = 70 # feature 갯수
    hidden_dim = 32 # 은닉층 길이
    output_dim = 1 # true/false
    learning_rate = 0.0001
    #iterations = 1
    iterations = 50000

    # 우리는 일반적인 시계열 형태가 아닌
    # 각 시계열에 해당하는 것이 많은 경기가 존재하는
    # 배치가 존재하는 시계열임에 주의하여 진행

    print("Valset Loading...")

    # 각 index들에 대해서 시계열 형태로 붙여야함
    df = pd.read_csv("./final_data_processed/worlds.csv")

    print("dataset_shape: ", df.shape)

    print("Testset creation...")

    # testX의 각 element가 각 게임의 시간 순으로 feature가 들어가도록 조작해야함...
    dataX = []
    dataY = []
    dataX.append(df.loc[0:finalIdx, df.columns != 'blueWins'].to_numpy())
    _y = df.copy()
    dataY.append(_y.loc[0:finalIdx, df.columns == 'blueWins'].to_numpy())

    testX = np.array(dataX)
    testY = np.array(dataY)

    testX_tensor = torch.FloatTensor(testX)
    testY_tensor = torch.FloatTensor(testY)

    print("testX shape: ", testX_tensor.shape)
    print("testY shape: ", testY_tensor.shape)

    print("Data Loading Finished....")

    class Net(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, layers):
            super(Net, self).__init__()
            self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers=layers, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)

        def forward(self, x):
            x, _status = self.rnn(x)
            x = self.fc(x[:, -1])
            x = torch.sigmoid(x)
            return x

    net = Net(data_dim, hidden_dim, output_dim, 1).to(device)

    PATH = "./RNN_GRU_model.pth"
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint)

    net.eval()
    predY = net(testX_tensor.to(device)).to('cpu').data.flatten()
    print(predY)
    for i in range(0, len(predY)):
        if predY[i] >= 0.5:
            predY[i] = 1
        else:
            predY[i] = 0

    correct = int(sum(predY == testY_tensor.flatten()))
    print("isCorrect?: ", correct >= 1)

    # plt.plot(testY.flatten())
    # plt.plot(predY)
    # plt.legend(['original', 'prediction'])
    # plt.show()

if __name__ == "__main__":
    for i in range(0, 23):
        runner(i)