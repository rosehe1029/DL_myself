import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
'''
## 问题描述：

利用卷积神经网络，实现对MNIST 数据集的分类问题。


'''

BATCH_SIZE=50
EPOCHS=20 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
keep_prob_rate=0.1
'''
DOWNLOAD_MNIST=False
if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST=True
'''
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)


class CNN (nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,65,kernel_size= 5 ),#input_size=28*28*3
            nn.ReLU(),        #activation function
            nn.MaxPool2d(2),  #pooling operation #input_size= 24 *  24 *65

        )
        self.conv2=nn.Sequential(
            nn.Conv2d(65,64,kernel_size=3 ),#input_size=12*12 *64 #output_size=10*10*64
            nn.ReLU(),
            nn.MaxPool2d(2), #output_size=5*5*64

        )
        self.out1=nn.Linear(5*5*64,1024,bias=True)
        self.dropout=nn.Dropout(keep_prob_rate)
        self.out2=nn.Linear(1024,10,bias=True)

    def forward (self,x):
        x=self.conv1(x)
        #print(x.shape)
        x=self.conv2(x)
        #print(x.shape)
        x=x.view( x.size()[0],-1)#拓展，展平
        #print(x.shape)
        out1=self.out1(x)
        #print(x.shape)
        out1=F.relu(out1)
        out1=self.dropout(out1)
        out2=self.out2(out1)
        output=F.softmax(out2)
        return output


model=CNN().to(DEVICE)
optimizer=optim.Adam(model.parameters())

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))




for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)


