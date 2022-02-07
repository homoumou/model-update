import torch
import pandas as pd
import numpy as np
import time
from torch import nn
from torch.nn import functional as F
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.utils.data as Data
import glob, os
import warnings
warnings.simplefilter('ignore')




# create resblock
class resblock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(resblock, self).__init__()
        self.conv_1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn_1 = nn.BatchNorm2d(ch_out)
        self.conv_2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(ch_out)
        self.ch_trans = nn.Sequential()
        if ch_in != ch_out:
            self.ch_trans = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                                          nn.BatchNorm2d(ch_out))
        # ch_trans表示通道数转变。因为要做short_cut,所以x_pro和x_ch的size应该完全一致

    def forward(self, x):
        x_pro = F.relu(self.bn_1(self.conv_1(x)))
        x_pro = self.bn_2(self.conv_2(x_pro))

        # short_cut:
        x_ch = self.ch_trans(x)
        out = x_pro + x_ch
        out = F.relu(out)
        return out

    # create resnet
class Resnet18(nn.Module):
    def __init__(self, num_class):
        super(Resnet18, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16))
        self.block1 = resblock(16, 32, 1)
        self.block2 = resblock(32, 64, 1)
        self.block3 = resblock(64, 128, 2)
        self.block4 = resblock(128, 256, 2)
        self.outlayer = nn.Linear(256 * 3 * 3, num_class)  # 这个256*3*3是根据forward中x经过4个resblock之后来决定的

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.reshape(x.size(0), -1)  # flatten
        result = self.outlayer(x)
        return result

def get_Data(path):
    file = glob.glob(os.path.join(path, "test2.csv"))
    print(file)
    dl = []
    for f in file:
        dl.append(pd.read_csv(f, header=[0], index_col=None))
    df = pd.concat(dl)
    return df, dl

def get_Mnist(path):
    # load data
    trans = transforms.Compose((transforms.Resize((32, 32)), transforms.ToTensor()))
    train_set = datasets.MNIST(path, train=True, transform=trans, download=False)
    print("train_set length: ", len(train_set))

    val_set = list(datasets.MNIST(path, train=False, transform=trans, download=False))[:5000]
    test_set = list(datasets.MNIST(path, train=False, transform=trans, download=False))[5000:]
    return train_set, val_set , test_set

def create_datasets(train_set, val_set , test_set, train_batch_size, val_batch_size, test_batch_size):
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


def seperate_frist_datasets(path,data_range):
    trans = transforms.Compose((transforms.Resize((32, 32)), transforms.ToTensor()))
    train_set = list(datasets.MNIST(path, train=True, transform=trans, download=False))[:data_range]
    print("train_set length: ",len(train_set))
    return train_set


def seperate_second_datasets(path,data_range):
    trans = transforms.Compose((transforms.Resize((32, 32)), transforms.ToTensor()))
    train_set = list(datasets.MNIST(path, train=True, transform=trans, download=False))[data_range:]
    print("train_set length: ",len(train_set))
    return train_set


def evaluate_accuracy(x, y, model):
    output, pre_out = model(x)
    output = torch.reshape(output, [-1, 1])
    correct = (output.ge(0.5) == y).sum().item()
    n = y.shape[0]
    return correct / n

def training(epochs, train_loader):
    device = torch.device('cuda')
    model = Resnet18(10).to(device)  # 10 Classes
    print('total parameters of trianing model: {}'.format(sum(map(lambda p: p.numel(), model.parameters()))))
    loss_fn = nn.CrossEntropyLoss()  # 选择loss_function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    time_start = time.time()
    est_epoch, best_acc = 0, 0
    for epoch in range(epochs):
        for batch_num, (img, label) in enumerate(train_loader):
            # img.size [b,3,224,224]  label.size [b]
            img, label = img.to(device), label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            if (batch_num + 1) % 100 == 0:
                print('epoch:{} batch:{} loose:{}'.format(epoch + 1, batch_num + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:  # 这里设置的是每训练两次epoch就进行一次validation
            val_acc = evaluate(model, val_loader)
            # 如果val_acc比之前的好，那么就把该epoch保存下来，并把此时模型的参数保存到指定txt文件里
            if val_acc > best_acc:
                print('The accuracy on the validation datset: {}'.format(val_acc))
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'mnist_resnet_ckp.txt')

    time_end = time.time()
    print('totally cost', time_end - time_start)

    print('best_acc:{},best_epoch:{}'.format(best_acc, best_epoch))
    model.load_state_dict(torch.load('mnist_resnet_ckp.txt'))
    print('training finished, now start testing test_set')

    test_acc = evaluate(model, test_loader)
    print('The accuracy on the test set: {}'.format(test_acc))


def evaluate(model, loader):
    device = torch.device('cuda')
    correct_num = 0
    total_num = len(loader.dataset)
    for img, label in loader:  # lodaer中包含了很多batch，每个batch有32张图片
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            logits = model(img)
            pre_label = logits.argmax(dim=1)
        correct_num += torch.eq(pre_label, label).sum().float().item()

    return correct_num / total_num




def pred_training_Cost(alpha, beta, data_size):
    train_time = alpha * data_size + beta
    print('pred_training_time:', train_time)
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
    BATCH_SIZE = 500
    train_dataloader = torch.utils.data.DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    return train_dataloader

def get_Datasize(begin, end, batch_Size):
    datasize = 0
    for x in range(begin, end):
        datasize += batch_Size
    return datasize

# begin: start time   end: data  end time
def calcLat(begin, end, alpha, beta, weight, Data_len):
    data_size = get_Datasize(begin, end, Data_len)
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
    print('whole_latency',whole_latency)
    for x in range(2, n - 1):
        print(x)
        # data_size = get_Datasize(2, n - 1, Data)
        # alpha, beta = get_parameter()
        latency = calcLat(1, x, alpha, beta, weight, Data) + calcLat(x + 1, n, alpha, beta, weight, Data)
        print('latency: ',latency)
        print(whole_latency - latency)
        print(weight * whole_latency)
        if whole_latency - latency > weight * whole_latency:  # l - l` > w*l
            update_time = x
            return update_time, whole_latency - latency

    return None



def evaluate_accuracy(x,y,model):
    output,pre_out = model(x)
    output = torch.reshape(output,[-1,1])
    correct = (output.ge(0.5) == y).sum().item()
    n = y.shape[0]
    return correct

def selectDataset(bacth_df):
    wasted_df = bacth_df[bacth_df[0] >=1]
    wasted_array = wasted_df.index
    return wasted_array

def replaceDataset(wasted_array, used_dataset, new_dataset):
    for batch_index in wasted_array:
        index = batch_index * 60
        used_dataset[index:(index + 60)] = new_dataset[index:(index + 60)]

if __name__ == "__main__":
    alpha = 0.00136738
    beta = 2.57748957

    train_batch_size=100
    val_batch_size=50
    test_batch_size=50
    whole_train_set, val_set, test_set = get_Mnist('./num')
    print('whole_train_set', len(whole_train_set))
    whole_train_loader, val_loader, test_loader = create_datasets(whole_train_set, val_set, test_set,train_batch_size, val_batch_size, test_batch_size)


    # every second receive one data
    a = []
    for x in range(0, len(whole_train_loader)):
        a.append(x)

    update_time, saveLatency = calUpdate(a, alpha, beta, train_batch_size)
    print('whole dataset uqdate time: ',update_time)

    #traning
    # seperate datasets
    print('---------------------------------first datasets training----------------------------------------------------')
    data_range = update_time * train_batch_size
    train_set = seperate_frist_datasets('./num',data_range)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    training(10, train_loader)

    print('----------------------------------secondary datasets training-------------------------------------------------')
    data_range = update_time * train_batch_size
    train_set = seperate_second_datasets('./num', data_range)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    training(10, train_loader)


    print('----------------------------------whole datasets training-------------------------------------------------')
    # data_range = update_time * train_batch_size
    # train_set = seperate_second_datasets('./num', data_range)
    print('train_set length: ', len(whole_train_set))
    train_loader = DataLoader(whole_train_set, batch_size=train_batch_size, shuffle=True)
    training(10, train_loader)


    # 761 > 655+89

    #27.76 + 60.27393817901611

    # 42.28117108345032
    # 44.67838668823242
    # 106.47734546661377

    whole_latency = calcLat(1, 2, alpha, beta, 0.4, 150)
    print(whole_latency)
    whole_latency = calcLat(1, 4, alpha, beta, 0.4, 150)
    print(whole_latency)
    whole_latency = calcLat(1, 6, alpha, beta, 0.4, 150)
    print(whole_latency)




    pass
