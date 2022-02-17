import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils import data
from tqdm import tqdm


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False, dropout=0)

        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        per_out = []
        lstm_out, self.hidden_cell = self.lstm(x)
        per_out.append(lstm_out)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        per_out.append(out)
        score = self.sigmoid(out)
        return score, per_out


def get_Data(path):
    file = glob.glob(os.path.join(path, "test*.csv"))
    print(file)
    dl = []
    for f in file:
        dl.append(pd.read_csv(f, header=[0], index_col=None))
    df = pd.concat(dl)
    return df, dl


def data_preprocessing(df):
    y = df.attack
    x = df.drop(['attack', 'attack_P1', 'attack_P2', 'attack_P3', 'time'], axis=1)
    values = x
    values.replace([np.inf, -np.inf], np.nan, inplace=True)
    #normalization
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(values)
    # Divide data into training and validation subsets
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_normalized, y, train_size=0.9, test_size=0.1,random_state=0)
    TIME_STEPS = 1
    X_train = pd.DataFrame(X_train_full)
    X_train = create_dataset(X_train, TIME_STEPS)
    y_train = pd.DataFrame(y_train)
    y_train = create_dataset(y_train, TIME_STEPS)
    X_valid = pd.DataFrame(X_valid_full)
    X_valid = create_dataset(X_valid, TIME_STEPS)
    y_valid = pd.DataFrame(y_valid)
    y_valid = create_dataset(y_valid, TIME_STEPS)

    return X_train, y_train, X_valid, y_valid


def evaluate_accuracy(x, y, model):
    output, pre_out = model(x)
    output = torch.reshape(output, [-1, 1])
    correct = (output.ge(0.5) == y).sum().item()
    n = y.shape[0]
    return correct / n


# function to convert to time domain dataset
def create_dataset(X, time_steps):
    Xs = []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
    return np.array(Xs)


def training(epochs, train_dataloader):
    device = torch.device('cuda')
    model = LSTM(input_size, output_size, hidden_size, num_layers).to(device)
    loss_function = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for batch, (batch_x, batch_y) in enumerate(train_dataloader):
            batch_x = batch_x.cuda()

            output, pre_out = model(batch_x)
            output = torch.reshape(output, [-1, 1])

            batch_y = np.array(batch_y)
            batch_y = torch.tensor(np.reshape(batch_y, [-1, 1]))
            batch_y = batch_y.float()
            batch_y = batch_y.cuda()

            loss = loss_function(output, batch_y)
            acc = evaluate_accuracy(batch_x, batch_y, model)
            if (batch + 1) % 100 == 0:
                print('epoch:{} batch:{} loose:{}'.format(epoch + 1, batch + 1, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print("epoch:{} batch:{} loss:{} acc:{}".format(epoch, batch, loss.item(), acc))
    return model

# def training(epochs):
#     device = torch.device('cuda')
#     model = LSTM(input_size, output_size, hidden_size, num_layers).to(device)
#     loss_function = nn.BCELoss().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     time_start = time.time()
#     for epoch in range(epochs):
#         model.train()
#         x = torch.tensor(X_train[:, :, :]).float()
#         x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
#
#         y = y_train[:]
#         y = np.array(y)
#         y = torch.tensor(np.reshape(y, [-1, 1]))
#         y = y.float()
#
#         x = x.cuda()
#         y = y.cuda()
#
#         output, pre_out = model(x)
#         output = torch.reshape(output, [-1, 1])
#         loss = loss_function(output, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if epoch % 5 == 0:
#             print("epoch:{} loss:{}".format(epoch, loss.item()))
#
#
#         time_end = time.time()
#         print('totally cost', time_end - time_start)




def pred_training_Cost(alpha, beta, data_size):
    train_time = alpha * data_size + beta
    return train_time


def create_dataload(x, y):
    data_x = torch.tensor(x[:, :, :]).float()
    data_x = torch.where(torch.isnan(data_x), torch.full_like(x, 0), x)

    data_y = y[:]
    data_y = np.array(data_y)
    # y = torch.tensor(np.reshape(y,[-1,1]))
    data_y = torch.tensor(data_y)
    data_y = data_y.float()
    torch_dataset = Data.TensorDataset(data_x, data_y)
    BATCH_SIZE = 60
    train_dataloader = torch.utils.data.DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    return train_dataloader


def update(epochs, X_train, y_train, model):
    device = torch.device('cuda')
    loss_function = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(len(X_train))
    time_start = time.time()

    for epoch in tqdm(range(epochs)):
        model.train()
        x = torch.tensor(X_train[:, :, :]).float()
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)

        y = y_train[:]
        y = np.array(y)
        y = torch.tensor(np.reshape(y, [-1, 1]))
        y = y.float()

        x = x.cuda()
        y = y.cuda()

        output, pre_out = model(x)
        output = torch.reshape(output, [-1, 1])
        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate_accuracy(x,y,model)
        if epoch % 5 == 0:
            print("epoch:{} loss:{} acc:{}".format(epoch, loss.item(), acc))

    time_end = time.time()
    print('totally cost', time_end - time_start)
    return

def evaluate_model(X_train,y_train, X_valid, y_valid, model):
    model.eval()
    x = torch.tensor(X_train[:, :, :]).float()
    x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
    y = y_train[:]
    y = np.array(y)
    y = torch.tensor(np.reshape(y, [-1, 1]))
    y = y.float()
    x = x.cuda()
    y = y.cuda()
    x_valid = torch.tensor(X_valid[:, :, :]).float()
    x_valid = torch.where(torch.isnan(x_valid), torch.full_like(x_valid, 0), x_valid)
    y_val = y_valid[:]
    y_val = np.array(y_val)
    y_val = torch.tensor(np.reshape(y_val, [-1, 1]))
    y_val = y_val.float()

    x_valid = x_valid.cuda()
    y_val = y_val.cuda()

    print("train acc", evaluate_accuracy(x, y, model))
    print("valid acc:", evaluate_accuracy(x_valid, y_val, model))

# def get_Datasize(begin, end, Data):
#     datasize = 0
#     for x in range(begin, end):
#         datasize += Data[x].shape[0]
#     return datasize

def get_Datasize(begin, end, Data):
    datasize = Data[begin:end].shape[0]

    return datasize

# begin: start time   end: data  end time
def calcLat(begin, end, alpha, beta, weight, Data):
    data_size = get_Datasize(begin, end, Data)
    train_time = pred_training_Cost(alpha, beta, data_size)
    end_time = train_time*1000 + (end - 1)
    latency = 0
    for x in range(begin, end):
        latency += end_time - x
    return weight * train_time + latency /1000


# finding the best time to update model
def calUpdate(a, alpha, beta, Data):
    n = len(a)
    weight = 0.4
    whole_latency = calcLat(1, n, alpha, beta, weight, Data)
    left = 0
    right = n - 1

    while (left < right):
        mid = left + ((right - left) >> 1);
        latency = calcLat(1, mid, alpha, beta, weight, Data) + calcLat(mid + 1, n, alpha, beta, weight, Data)
        print('latency reduction:', whole_latency - latency)
        print('expect reduction:', weight * whole_latency)
        if whole_latency - latency > weight * whole_latency:
            right = mid
        else:
            left = mid + 1;

    update_time = right
    return update_time, whole_latency - latency

def getDataloader(X_train,y_train):
# 将输入和输出封装进Data.TensorDataset()类对象
    x = torch.tensor(X_train[:,:,:]).float()
    x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
    y = y_train[:]
    y = np.array(y)
# y = torch.tensor(np.reshape(y,[-1,1]))
    y = torch.tensor(y)
    y = y.float()
    torch_dataset = Data.TensorDataset(x,y)
    train_dataloader = torch.utils.data.DataLoader(torch_dataset, batch_size=60, shuffle=False, num_workers=0)
    return X_train, y_train, train_dataloader

# Selected wasted dataset
def selectDataset(bacth_df):
    wasted_df = bacth_df[bacth_df[0] >=1]
    wasted_array = wasted_df.index
    return wasted_array

def replaceDataset(wasted_array, used_dataset, new_dataset):
    for batch_index in wasted_array:
        index = batch_index * 60
        used_dataset[index:(index + 60)] = new_dataset[index:(index + 60)]
    return used_dataset

if __name__ == "__main__":
    input_size = 79
    output_size = 1
    hidden_size = 64
    num_layers = 2
    # alpha = 0.00682448
    # beta = 3.0235812
    # alpha = 0.00017225
    # beta = 2.25016395
    alpha = 0.00659588
    beta = 18.31456436
    path = r'../Dataset/hai-master/hai-21.03'
    df_data,dl = get_Data(path)

    # every second receive one data
    a = []
    for x in range(0, len(df_data)):
        a.append(x)
    print(a)
    # model = buildModel(input_size, output_size, hidden_size, num_layers)
    # update_time, saveLatency = decideUpdate(a, alpha, beta, dl)
    update_time, saveLatency = calUpdate(a, alpha, beta, df_data)
    print('update time:', update_time)

    if update_time != None:
       device = torch.device('cuda')
       model = LSTM(input_size, output_size, hidden_size, num_layers).to(device)
       X_train, y_train, x_val, y_val = data_preprocessing(df_data[0:update_time])
       x, y, dataloader = getDataloader(X_train,y_train)
       print('train data size:', len(X_train))
       # update(200, X_train, y_train, model)
       model = training(1, dataloader)
       evaluate_model(X_train, y_train, x_val, y_val, model)

    X_train, y_train, x_val, y_val = data_preprocessing(df_data[update_time:])
    x, y, dataloader = getDataloader(X_train, y_train)
    print('train data size:', len(X_train))
    # update(200, X_train, y_train, model)
    model = training(1, dataloader)
    evaluate_model(X_train, y_train, x_val, y_val, model)

    X_train, y_train, x_val, y_val = data_preprocessing(df_data)
    x, y, dataloader = getDataloader(X_train, y_train)
    print('train data size:', len(X_train))
    # update(200, X_train, y_train, model)
    model = training(1, dataloader)
    evaluate_model(X_train, y_train, x_val, y_val, model)
    # 761 > 655+89
    pass
