import collections
import math
import os
import random
import sys
import tarfile
import time
import zipfile
from tqdm import tqdm

from IPython import display
from matplotlib import pyplot as plt
import torch
import  torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchtext
import torchtext.vocab as Vocab
import numpy as np
import torch.nn as nn


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_jay_lyrics():
    '''加载周杰伦歌词数据集'''
    with open('jay.txt',encoding='utf-8') as f:
        corpus_chars=f.read()#.decode('utf-8')
    corpus_chars=corpus_chars.replace('\n',' ').replace('r',' ')
    corpus_chars=corpus_chars[0:10000]
    idx_to_char=list(set(corpus_chars))
    char_to_idx=dict([(char,i) for i ,char in enumerate(idx_to_char)])
    vocab_size=len(char_to_idx)
    corpus_indices=[char_to_idx[char ] for char in corpus_chars]
    return corpus_indices,char_to_idx,idx_to_char,vocab_size

corpus_indices, char_to_idx, idx_to_char, vocab_size=load_data_jay_lyrics()

def data_iter_random(corpus_indices,batch_size,num_steps,devices=None):
    num_examples=(len(corpus_indices)-1)//num_steps
    epoch_size=num_examples//batch_size
    example_indices=list(range(num_examples))
    random.shuffle(example_indices)

    #返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos:pos+num_steps]
    if devices is None:
        devices=torch.device('cuda' if torch.cuda().is_available() else 'cpu')

    for i in range(epoch_size):
        #每次读取batch_size个随机样本
        i=i*batch_size
        batch_indices=example_indices[i:i+batch_size]
        X=[_data(j*num_steps)for j in batch_indices]
        Y=[_data(j*num_steps+1)for j in batch_indices]
        yield  torch.tensor(X,dtype=torch.float32,device=device),torch.tensor(Y,dtype=torch.float32,device=device)


def data_iter_consecutive(corpus_indices,batch_size,num_steps,device=None):
    if device is None:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices=torch.tensor(corpus_indices,dtype=torch.float32,device=device)
    data_len=len(corpus_indices)
    batch_len=data_len//batch_size
    indices=corpus_indices[0:batch_size*batch_len].view(batch_size,batch_len)
    epoch_size=(batch_len-1)//num_steps
    for i in range(epoch_size):
        i=i*num_steps
        X=indices[:,i:i+num_steps]
        Y=indices[:,i+1:i+num_steps+1]
        yield X,Y

def one_hot(x,n_class,dtype=torch.float32):
    x=x.long()
    res=torch.zeros(x.shape[0],n_class,dtype=dtype,device=device)
    res.scatter_(1,x.view(-1,1),1)
    return res

def to_onehot(X,n_class):
    return [one_hot(X[:,i],n_class) for i in range(X.shape[1])]

def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

def predict_rnn(prefix,num_chars,rnn,params,init_rnn_state,num_hiddens,vocab_size,device,idx_to_char,char_to_idx):
    state=init_rnn_state(1,num_hiddens,device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

class RNNModel(nn.Module):
    def __init__(self,rnn_layer,  vocab_size):
        super(RNNModel, self).__init__()
        self.rnn=rnn_layer
        self.hidden_size = rnn_layer.hidden_size*(2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):  # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = to_onehot(inputs, self.vocab_size)  # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                        char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)  # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            (output, state) = model(X, state)  # output: 形状为(num_steps * batch_size, vocab_size)

            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % 2 == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))


num_hiddens=8
rnn_layer=nn.RNN(input_size=vocab_size,hidden_size=num_hiddens)
rnn=RNNModel( rnn_layer,vocab_size)

num_epochs=10
num_steps=1
lr=0.0005
clipping_theta=0.0000001
batch_size=2
pred_len=7
prefixes='爱'
t=train_and_predict_rnn_pytorch(rnn, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size,  pred_len, prefixes)
print(t)