#https://github.com/920232796/pytorch-rnn_poem/blob/master/d2lzh_pytorch/utils%E7%9A%84%E5%89%AF%E6%9C%AC.py

#处理文本
import numpy as np
datas=np.load("tang.npz", allow_pickle=True)
data=datas['data']
ix2word=datas['ix2word'].item()
word2ix=datas['word2ix'].item()

#print(data)
#print('ix2word',ix2word)
#print('word2ix',word2ix)

####数据再继续处理一下，全部弄成五言律诗，没有标点，也没有额外的标识符
poem_list=[]
for each_row in  data:
    each_list=[]
    for i in each_row:
        word=ix2word[i]    #得到一个具体的字了
        if (word=="</s>"  or word =="<START>" or word =="。"or word=="<EOP>"):
            continue
        else :
            each_list.append(word)
    poem_list.append(each_list)

#得到的诗全是逗号分割
#print(len(poem_list))
#print(poem_list[:100])

poem=[]
for each_poem in poem_list:
    str_poem="".join(each_poem)
    sentence_poem=str_poem.split(",") #这样就把每句诗都弄到一个列表里面 每句诗都是字符串
    for each in sentence_poem:
        poem.append(each)

#print(poem[:10])

#提取五言律诗
five_poem=[]
for each_poem in poem:
    if (len(each_poem)==5):
        five_poem.append(each_poem)

#print(len(five_poem))
#print(five_poem[:10])

#将五言律诗弄成indice的形式，以后方便输入
five_poem_indice=[]
for each_poem in five_poem:
    five_poem_indice.append([word2ix[word] for word in  each_poem])

print(len(five_poem_indice))
print(five_poem_indice[:10])


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)



import torch
#构造输入输出连续采样，一句一句诗来取
def data_iter_consecutive(poem_indice,batch_size,num_step=4,device=None):
    if device==None:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_size=len(poem_indice)//batch_size
    poem_indice=torch.tensor(poem_indice,dtype=torch.float32,device=device)
    indice=poem_indice[0:epoch_size*batch_size]

    for epoch in range(epoch_size):
        i=epoch*batch_size
        X=indice[i:i +batch_size]#现在得到的X是五个字，但是输入只需要前四个字就ok
        #print('x',X)
        X=X[:,0:4]
        Y=indice[i:i+batch_size]
        Y=Y[:,1:5]
        yield X,Y
        #print('X',X)
        #print('Y',Y)

data_iter=data_iter_consecutive(five_poem_indice,2)
i=0
for X,Y in data_iter:
    if (i==2):
        break
    i+=1
    #print(X)
    #print(Y)

def one_hot(x,n_class,dtype=torch.float32):
    x=x.long()
    res=torch.zeros(x.shape[0],n_class,dtype=dtype,device=x.device)
    res.scatter_(1,x.view(-1,1),1)
    return res

def to_onehot(X,n_class):
    return [one_hot(X[:,i],n_class) for i in range(X.shape[1])]

import time
import math
import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_hiddens=256
vocab_size=len(ix2word)

rnn_layer=nn.RNN(input_size=vocab_size,hidden_size=num_hiddens)
num_step=35 #时代步数
batch_size=2
state=None #隐藏状态

class RNNModel(nn.Module):

    def __init__(self,rnn_layer,vocab_size):
        super(RNNModel,self).__init__()
        self.rnn=rnn_layer
        self.hidden_size=rnn_layer.hidden_size
        self.vocab_size=vocab_size

        self.dense=nn.Linear(self.hidden_size,vocab_size) ##全连接层，最后输出用
        self.state=None

    def forward(self,inputs,state):
        X=to_onehot(inputs,self.vocab_size)
        Y,self.state=self.rnn(torch.stack(X),state)

        output=self.dense(Y.view(-1,Y.shape[-1]))
        return output,self.state


#写一个预测函数
def predict_rnn(prefix,num_chars,model,vocab_size,device,idx_to_char,char_to_idx):
    state=None
    output=[char_to_idx[prefix[0]]] #先把第一个字符加入到结果里面
    print('prefix',prefix)
    for t in range(num_chars+len(prefix)-1):
        X=torch.tensor([output[-1]],device=device).view(1,1) #构造出来x
        #print('X',X)
        if state is not None:
            if isinstance(state,tuple):
                state=(state[0].to(device),state[1].to(device))
            else :
                state=state.to(device)
        (y,state)=model(X,state)
        if (t<len(prefix)-1):
            output.append(char_to_idx[prefix[t+1]])

        else :
            output.append(int(y.argmax(dim=1).item()))
    #print('output',output)
    return " ".join([idx_to_char[i] for i in output])

#随机权值预测一次
model=RNNModel(rnn_layer,vocab_size).to(device)
p0=predict_rnn('花',4,model,vocab_size,device,ix2word,word2ix)
p1=predict_rnn('前',4,model,vocab_size,device,ix2word,word2ix)
p2=predict_rnn('月',4,model,vocab_size,device,ix2word,word2ix)
p3=predict_rnn('下',4,model,vocab_size,device,ix2word,word2ix)
print(p0)
print(p1)
print(p2)
print(p3)
'''

def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx, num_epochs,  num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
  loss = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr)

  model.to(device)
  state = None

  for epoch in range(num_epochs):
    l_sum, n, start = 0.0, 0, time.time()
    data_iter = data_iter_consecutive(corpus_indices, batch_size, device)

    for X, Y in data_iter:
      if state is not None:
        if isinstance (state, tuple): # LSTM, state:(h, c)
          state = (state[0].detach(), state[1].detach())
        else:
          state = state.detach()

      (output, state) = model(X, state)

      ## y 应该先转置 然后再view 成 跟output行数一样！
      y = torch.transpose(Y, 0, 1).contiguous().view(-1)
      l = loss(output, y.long())
      optimizer.zero_grad()##梯度清零
      l.backward()

      grad_clipping(model.parameters(), clipping_theta, device)
      optimizer.step()
      l_sum += l.item() * y.shape[0] ## shape[0] 是batch_size * num_step 因为求loss的时候 除以N了

    if (epoch + 1) % pred_period == 0:
        print('epoch %d, time %.2f sec, loss %.8f' % (
                epoch + 1, time.time() - start, l_sum))
        for prefix in prefixes:
          print(' -', predict_rnn( prefix, pred_len, model, vocab_size, device, idx_to_char, char_to_idx))



num_epochs, batch_size, lr, clipping_theta = 10, 2, 1e-3, 1e-2 # 注意这里的学习率设置
pred_period, pred_len, prefixes = 4, 4, ['佩', '奇', '小', '兔']
p=train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,five_poem_indice, ix2word, word2ix,  num_epochs, 4, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)

torch.save(model, "./model.pkl")
print(p)
'''
