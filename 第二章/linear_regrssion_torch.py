import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
import matplotlib.pyplot
from torch.autograd import Variable


def identity_basis(x):
    ret = np.expand_dims(x, axis=1)
    return ret


def multinomial_basis(x, feature_num=10):
    x = np.expand_dims(x, axis=1)  # shape(N, 1)
    feat = [x]
    for i in range(2, feature_num + 1):
        feat.append(x ** i)
    ret = np.concatenate(feat, axis=1)
    return ret


def gaussian_basis(x, feature_num=10):
    centers = np.linspace(0, 25, feature_num)
    width = 1.0 * (centers[1] - centers[0])
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x] * feature_num, axis=1)

    out = (x - centers) / width
    ret = np.exp(-0.5 * out ** 2)
    return ret


def load_data(filename, basis_func=gaussian_basis):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))
        xs, ys = zip(*xys)
        xs, ys = np.asarray(xs), np.asarray(ys)

        o_x, o_y = xs, ys
        phi0 = np.expand_dims(np.ones_like(xs), axis=1)
        phi1 = basis_func(xs)
        xs = np.concatenate([phi0, phi1], axis=1)
        return (np.float32(xs), np.float32(ys)), (o_x, o_y)

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()#继承父类构造函数
        self.linear=torch.nn.Linear(1,1) #线性输出

    def forward(self,x):#前向传播
        out=self.linear(x)
        return out

(x,y),(o_x,o_y)=load_data('Utrain.txt')

model=LinearRegression()  #实例化对象

epoch_n=1000 #迭代次数
learning_rate=1e-2 #学习率

criterion=torch.nn.MSELoss() #损失函数
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)#优化函数

for epoch in range(epoch_n):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    optimizer.zero_grad()      #清空上一步参数值
    loss.backward()#反向传播
    optimizer.step()#更新参数


def evaluate(y,y_pred):
    std=np.sqrt(np.mean(np.abs(y-y_pred)**2))
    return std


y_preds = model.forward(x)
std = evaluate(y, y_preds)
print('训练集预测值与真实值的标准差：{:.1f}'.format(std))

(xs_test, ys_test), (o_x_test, o_y_test) = load_data('test.txt')

y_test_preds = model.forward(x)
std = evaluate(ys_test, y_test_preds)
print('训练集预测值与真实值的标准差：{:.1f}'.format(std))

plt.plot(o_x, o_y, 'ro', markersize=3)
plt.plot(o_x_test, y_test_preds, 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend(['train', 'test', 'pred'])
plt.show()



