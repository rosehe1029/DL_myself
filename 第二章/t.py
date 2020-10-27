import numpy as np


def load_data(filename):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)


class SigmoidActivator(object):
    def forward (self,weighted_input):
        return 1.0/(1.0 + np.exp(-weighted_input))
    def backward(self,output):
        return output*(1-output)



class TanActivator(object):
    def forward(self,weighted_input):
        return 2.0/(1.0 +np.exp(-2 *weighted_input))-1.0

    def backward(self,output):
        return 1-output*output

class FullConnectedLayer():
    def __init__(self,input_size,output_size,activator):
        self.input_size=input_size
        self.output_size=output_size
        self.activator=activator
        self.w=np.random.uniform(-0.1,0.1,(output_size,input_size))
        self.b=np.zeros((output_size,1))

    def forward(self,input_data):
        self.input_size=input_data
        self.output_data=self.activator.forward(np.dot(self.w,self.input_size)+self.b)

    def backward(self,input_delta):
        self.sen=input_delta * self.activator.backward(self.output_data)
        output_delta=np.dot(self.w.T,self.sen)
        self.w_grad=np.dot(self.sen,self.input_data.T)
        self.b_grad=self.sen
        self.w_grad_total+=self.w_grad
        self.b_grad_total+=self.b_grad
        return output_delta

    def update(self,lr,MBGD_mode=0):
        if MBGD_mode==0:
            self.w-=lr*self.w_grad
            self.b-=lr*self.b_grad
        elif MBGD_mode ==1:
            self.w-=lr*self.w_grad_add
            self.b-=lr*self.b_grad_add
            self.w_grad_add=np.zeros((self.output_size,self.input_size))
            self.b_grad_add=np.zeros((self.output_size,1))


class Network():
    def __init__(self, params_array, activator):
        # params_array为层维度信息超参数数组
        # layers为网络的层集合
        self.layers = []
        for i in range(len(params_array) - 1):
            self.layers.append(FullConnectedLayer(params_array[i], params_array[i + 1], activator))

    # 网络前向迭代
    def predict(self, sample):
        # 下面一行的output可以理解为输入层输出
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output_data
        return output

    # 网络反向迭代
    def calc_gradient(self, label):
        delta = (self.layers[-1].output_data - label)
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta

    # 一次训练一个样本 ，然后更新权值
    def train_one_sample(self, sample, label, lr):
        self.predict(sample)
        Loss = self.loss(self.layers[-1].output_data, label)
        self.calc_gradient(label)
        self.update(lr)
        return Loss

    # 一次训练一批样本 ，然后更新权值
    def train_batch_sample(self, sample_set, label_set, lr, batch):
        Loss = 0.0
        for i in range(batch):
            self.predict(sample_set[i])
            Loss += self.loss(self.layers[-1].output, label_set[i])
            self.calc_gradient(label_set[i])
        self.update(lr, 1)
        return Loss

    def update(self, lr, MBGD_mode=0):
        for layer in self.layers:
            layer.update(lr, MBGD_mode)

    def loss(self, pred, label):
        return 0.5 * ((pred - label) * (pred - label)).sum()

    def gradient_check(self, sample, label):
        self.predict(sample)
        self.calc_gradient(label)
        incre = 10e-4
        for layer in self.layers:
            for i in range(layer.w.shape[0]):
                for j in range(layer.w.shape[1]):
                    layer.w[i][j] += incre
                    pred = self.predict(sample)
                    err1 = self.loss(pred, label)
                    layer.w[i][j] -= 2 * incre
                    pred = self.predict(sample)
                    err2 = self.loss(pred, label)
                    layer.w[i][j] += incre
                    pred_grad = (err1 - err2) / (2 * incre)
                    calc_grad = layer.w_grad[i][j]
                    print(  'weights(%d,%d): expected - actural %.4e - %.4e' % (i, j, pred_grad, calc_grad))




##########################################################################################################################

import numpy as np
#激活函数tanh
def tanh(x):
    return np.tanh(x)
#tanh的导函数，为反向传播做准备
def tanh_deriv(x):
    return 1-np.tanh(x)*np.tanh(x)
#激活函数逻辑斯底回归函数
def logistic(x):
    return 1/(1+np.exp(-x))
#激活函数logistic导函数
def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))
#神经网络类
class NeuralNetwork:
    def __init__(self,layers,activation='tanh'):
    #根据激活函数不同，设置不同的激活函数和其导函数
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
       #初始化权重向量，从第一层开始初始化前一层和后一层的权重向量
        self.weights = []
        for i in range(1 , len(layers)-1):
         #权重的shape，是当前层和前一层的节点数目加１组成的元组
            self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
            #权重的shape，是当前层加１和后一层组成的元组
            self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)
    #fit函数对元素进行训练找出合适的权重，X表示输入向量，y表示样本标签，learning_rate表示学习率
    #epochs表示循环训练次数
    def fit(self , X , y , learning_rate=0.2 , epochs=10000):
        X  = np.atleast_2d(X)#保证X是二维矩阵
        temp = np.ones([X.shape[0],X.shape[1]+1])
        temp[:,0:-1] = X
        X = temp #以上三步表示给Ｘ多加一列值为１
        y = np.array(y)#将y转换成np中array的形式
        #进行训练
        for k in range(epochs):
            i = np.random.randint(X.shape[0])#从0-epochs任意挑选一行
            a = [X[i]]#将其转换为list
            #前向传播
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))
            #计算误差
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]
            #反向传播，不包括输出层
            for l in range(len(a)-2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            #更新权重
            for i in range(len(self.weights)):
                layer  = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate*layer.T.dot(delta)

    #进行预测
    def predict(self,x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0,len(self.weights)):
            a = self.activation(np.dot(a,self.weights[l]))
        return a
'''


import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
#from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    #加载数字数据集
    digits = load_digits()
    X = digits.data
    y = digits.target
    #对Ｘ进行最大最小值缩放
    X = MinMaxScaler().fit_transform(X)
    #生成一个64*100*10的神经网络，激活函数是logistic
    nn = NeuralNetwork([64,100,10],'logistic')
    #X_train,X_test,y_train,y_test = train_test_split(X,y)
    #对标签进行标签化
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)
    print 'start fitting'
    nn.fit(X_train,labels_train,epochs=3000)
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i])
        predictions.append(np.argmax(o))//选择概率最大的下标作为预测结果
    #预测结果
    print predictions
    #混淆矩阵
    print confusion_matrix(y_test,predictions)
    #分类报告
    print classification_report(y_test,predictions)


import numpy as np
from tqdm import trange	# 替换range()可实现动态进度条，可忽略


def sigmoid(x): # 激活函数采用Sigmoid
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):	# Sigmoid的导数
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:	# 神经网络
    def __init__(self, layers):	# layers为神经元个数列表
        self.activation = sigmoid	# 激活函数
        self.activation_deriv = sigmoid_derivative	# 激活函数导数
        self.weights = []	# 权重列表
        self.bias = []	# 偏置列表
        for i in range(1, len(layers)):	# 正态分布初始化
            self.weights.append(np.random.randn(layers[i-1], layers[i]))
            self.bias.append(np.random.randn(layers[i]))

    def fit(self, x, y, learning_rate=0.2, epochs=3):	# 反向传播算法
        x = np.atleast_2d(x)
        n = len(y)	# 样本数
        p = max(n, epochs)	# 样本过少时根据epochs减半学习率
        y = np.array(y)

        for k in trange(epochs * n):	# 带进度条的训练过程
            if (k+1) % p == 0:
                learning_rate *= 0.5	# 每训练完一代样本减半学习率
            a = [x[k % n]]	# 保存各层激活值的列表
            # 正向传播开始
            for lay in range(len(self.weights)):
                a.append(self.activation(np.dot(a[lay], self.weights[lay]) + self.bias[lay]))
            # 反向传播开始
            label = np.zeros(a[-1].shape)
            label[y[k % n]] = 1	# 根据类号生成标签
            error = label - a[-1]	# 误差值
            deltas = [error * self.activation_deriv(a[-1])]	# 保存各层误差值的列表

            layer_num = len(a) - 2	# 导数第二层开始
            for j in range(layer_num, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))	# 误差的反向传播
            deltas.reverse()
            for i in range(len(self.weights)):	# 正向更新权值
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
                self.bias[i] += learning_rate * deltas[i]

    def predict(self, x):	# 预测
        a = np.array(x, dtype=np.float)
        for lay in range(0, len(self.weights)):	# 正向传播
            a = self.activation(np.dot(a, self.weights[lay]) + self.bias[lay])
        a = list(100 * a/sum(a))	# 改为百分比显示
        i = a.index(max(a))	# 预测值
        per = []	# 各类的置信程度
        for num in a:
            per.append(str(round(num, 2))+'%')
        return i, per

from NeuralNetwork import NeuralNetwork
import numpy as np
import pickle
import csv


def train():
    file_name = 'data/train.csv'	# 数据集为42000张带标签的28x28手写数字图像
    y = []
    x = []
    y_t = []
    x_t = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        header_row = next(reader)
        print(header_row)
        for row in reader:
            if np.random.random() < 0.8:	# 大约80%的数据用于训练
                y.append(int(row[0]))
                x.append(list(map(int, row[1:])))
            else:
                y_t.append(int(row[0]))
                x_t.append(list(map(int, row[1:])))
    len_train = len(y)
    len_test = len(y_t)
    print('训练集大小%d，测试集大小%d' % (len_train, len_test))
    x = np.array(x)
    y = np.array(y)
    nn = NeuralNetwork([784, 784, 10])	# 神经网络各层神经元个数
    nn.fit(x, y)
    file = open('NN.txt', 'wb')
    pickle.dump(nn, file)
    count = 0
    for i in range(len_test):
        p, _ = nn.predict(x_t[i])
        if p == y_t[i]:
            count += 1
    print('模型识别正确率：', count/len_test)


def mini_test():	# 小型测试，验证神经网络能正常运行
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 2, 3]
    nn = NeuralNetwork([2, 4, 16, 4])
    nn.fit(x, y, epochs=10000)
    for i in x:
        print(nn.predict(i))


# mini_test()
train()

import csv
import pickle
import numpy as np
from matplotlib import pyplot as plt


def diplay_test():	# 读取测试集，预测，画图
    file_name = 'data/test.csv'
    file = open('NN.txt', 'rb')
    nn = pickle.load(file)
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        header_row = next(reader)
        print(header_row)
        i = 0
        for row in reader:
            i += 1
            img = np.array(row, dtype=np.uint8)
            img = img.reshape(28, 28)
            plt.imshow(img, cmap='gray')
            pre, lst = nn.predict(row)
            plt.title(str(pre), fontsize=24)
            plt.axis('off')
            plt.savefig('img/img' + str(i) + '.png')


diplay_test()

'''