import numpy as np
import matplotlib.pyplot as plt
import os

save_dir='lg_w_b'
def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)

#定义sigmoid函数
def sigmoid(z):
    res=1/(1.0 + np.exp(-z))
    return np.clip(res,1e-8,1-(1e-8))

#验证模型的正确性
def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)


def train(X_train, Y_train):
    X_train = np.mat(X_train)
    print(X_train.shape)
    Y_train = np.mat(Y_train)
    Y_train=Y_train.T
    print(Y_train.shape)
    # 创建原始参数，设定学习速率，训练次数，每次训练用多少数据
    w = np.zeros((2, ))
    b = np.zeros((1,))
    print('w', w)
    print('b', b)
    l_rate = 0.1
    batch_size = 10
    train_data_size = len(X_train)
    step_num = int(float(train_data_size / batch_size))
    epoch_num = 1000
    save_param_iter = 50
    total_loss = 0.0
    # 开始训练
    for epoch in range(1, epoch_num):
        # 模型验证与保存参数
        if (epoch) % save_param_iter == 0:
            print('=====Saving Param at epoch %d=====' % epoch)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'w'), w)
            np.savetxt(os.path.join(save_dir, 'b'), [b,])
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
        # 每batch_size个数据为一组数据
        for idx in range(step_num):
            X = X_train[idx * batch_size:(idx + 1) * batch_size]
            #print('X.shape',X.shape)
            Y = Y_train[idx * batch_size:(idx + 1) * batch_size]
            #print('Y.shape',Y.shape)
            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z).reshape(batch_size,1)
            #print('y.shape',y.shape)

            #print(np.dot(np.squeeze(Y), np.log(y))
            #print(np.squeeze(Y).shape)
            #print(np.log(y).shape)
            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y))+ np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            #print(cross_entropy)
            total_loss += cross_entropy

            #print('np.squeeze(Y)-y',(np.squeeze(Y) - y).shape)
            w_grad = np.sum((-1 *(np.squeeze(Y) - y)* X), axis=0)
            b_grad = np.sum(-1 * (np.squeeze(Y) - y))

            w = w - l_rate * w_grad
            b = b - l_rate * b_grad

def predict(X_test):
    # 加载所得结果参数w和b
    print('=====Loading Param from %s=====' % save_dir)
    w = np.loadtxt(os.path.join(save_dir, 'w'))
    b = np.loadtxt(os.path.join(save_dir, 'b'))
    print('w',w)
    print('b',b)
    test_data_size=len(X_test)
    z=(np.dot(X_test,np.transpose(w))+b)
    y=sigmoid(z)
    y_=np.around(y)

    return y_



if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练
      # 训练模型


    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2] # 真实标签
    train(x_train,t_train.T)
    t_train_pred = predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))


