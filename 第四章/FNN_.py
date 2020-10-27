from torch import nn

class simpleNet(nn.Module):
    '''
    定义了一个简单的三层全连接神经网络，每一层都是线性的
    '''
    def   __init__ (self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simpleNet,self).__init__()
        self.layer1=nn.Linear(in_dim,n_hidden_1)
        self.layer2=nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3=nn.Linear(n_hidden_2,out_dim)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

class Activation_Net(nn.Module):
    '''
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数

    '''
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Activation_Net,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
        self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
        self.layer3=nn.Sequential(nn.Linear(n_hidden_2,out_dim))
        '''
        这里的sequential()函数的功能是将网络的层组合到一起
        '''

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

class Batch_Net(nn.Module):
    '''
    在上面的activation_net的基础上，增加了一个加快收敛速度的方法————批标准化
    '''
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Batch_Net,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
        self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidde_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True))


    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x


import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

batch_size=64
learning_rate=0.02
num_epoches=20

data_tf=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])]
)

# 数据集的下载器
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# 选择模型
model = simpleNet(28 * 28, 300, 100, 10)
# model = net.Activation_Net(28 * 28, 300, 100, 10)
# model = net.Batch_Net(28 * 28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


#训练模型
epoch=0
for data in train_loader:
    img,label=data
    img=img.view(img.size(0),-1)
    if torch.cuda.is_available():
        img=img.cuda()
        label=label.cuda()
    else :
        img=Variable(img)
        label=Variable(label)
    out=model(img)
    loss=criterion(out,label)
    print_loss=loss.data.item()


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch+=1
    if epoch%50 == 0:
        print('epoch :{} ,loss :{:.4}'.format(epoch,loss.data.item()))


