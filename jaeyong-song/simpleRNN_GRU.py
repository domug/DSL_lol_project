import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Random seed to make results deterministic and reproducible
torch.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# not for lol dataset...
# def minmax_scaler(data):
#     numerator = data - np.min(data, 0)
#     denominator = np.max(data, 0) - np.min(data, 0)
#     return numerator / (denominator + 1e-7)

seq_length = 20 # 30분 경과 후의 예측
data_dim = 70 # feature 갯수
hidden_dim = 32 # 은닉층 길이
output_dim = 1 # true/false
learning_rate = 0.0001
#iterations = 1
iterations = 50000

# 우리는 일반적인 시계열 형태가 아닌
# 각 시계열에 해당하는 것이 많은 경기가 존재하는
# 배치가 존재하는 시계열임에 주의하여 진행

print("Trainset Loading...")

trainX = np.load('./trainX.npy')
trainY = np.load('./trainY.npy')

print("Testset Loading...")

testX = np.load('./testX.npy')
testY = np.load('./testY.npy')


trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)

print("trainX shape: ", trainX_tensor.shape)
print("trainY shape: ", trainY_tensor.shape)

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

# loss & optimizer setting
criterion = torch.nn.BCELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

print("Training started!!!")

for i in range(iterations):
    net.train()
    optimizer.zero_grad()
    outputs = net(trainX_tensor.to(device))
    loss = criterion(outputs, trainY_tensor.to(device))
    loss.backward()
    optimizer.step()
    if i%1000 == 0:
        print(i, loss.item())

print("Training finished!!!")


net.eval()
predY = net(testX_tensor.to(device)).to('cpu').data.flatten()
for i in range(0, len(predY)):
    if predY[i] >= 0.5:
        predY[i] = 1
    else:
        predY[i] = 0

correct = int(sum(predY == testY_tensor.flatten()))
total = len(predY)
print("Total: ", total)
print("Correct: ", correct)
print("Accuracy: ", correct/total*100, "%")

PATH = "./RNN_GRU_model.pth"
torch.save(net.state_dict(), PATH)

# plt.plot(testY.flatten())
# plt.plot(predY)
# plt.legend(['original', 'prediction'])
# plt.show()
