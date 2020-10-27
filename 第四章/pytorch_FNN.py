import numpy as np
import torch
from torchvision.datasets import mnist
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

# 数据预处理，将28*28的变成784的格式大小
def data_tf(x):
    x = np.array(x, dtype="float32") / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1))
    x = torch.from_numpy(x)
    return x

# 数据预处理，这里没有使用
transform = transforms.ToTensor()

#  载入数据，并使用transform=data_tf对数据惊醒类型和格式转换
train_set = mnist.MNIST("./data", train=True, download=False, transform=data_tf)
test_set = mnist.MNIST("./data", train=True, download=False, transform=data_tf)

# 使用数据迭代器，每一个batch_size=64
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

# 构建4层神经网络
net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

# 损失函数使用交叉熵函数，采用随机梯度下降法，学习率为0.1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)

losses = []
acces = []
eval_losses = []
eval_acces =[]

for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        lable = Variable(label)
        # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # print(out.shape)
        # print(out)
        # print(out.max(1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录误差
        train_loss = train_loss + loss.data

        # torch.max(0)和 torch.max(1)分别是找出tensor里每列/每行中最大的值，并返回索引（即为对应的预测数字）
        _, pred = out.max(1)
        # 预测正确的总数
        num_correct = (pred == label).sum().data
        # 计算正确率
        acc = float(num_correct) / im.shape[0]  # 注意要换成浮点数，否则整形相除，acc一直为0
        # 将每一个batch的正确率相加，便于后面计算每一个epoch的正确率
        train_acc = train_acc + acc
    # 计算这个epoch的损失和正确率
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

    eval_loss = 0
    eval_acc = 0
    net.eval()
    # 测试集上测试
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)

        eval_loss += loss.data
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data
        acc = float(num_correct) / im.shape[0]
        eval_acc += acc

    # 计算这一个epoch在测试集上的损失和正确率
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))

    # 最后打印出这一个epoch上训练集的正确率和损失，以及测试机上的正确率和损失
    print("epoch :{}, Train Loss:{:.6f}, Train ACC:{:.6f}, Eval Loss: {:.6f}, Eval ACC: {:.6f}".
    format(e, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
            eval_acc / len(test_data)))
