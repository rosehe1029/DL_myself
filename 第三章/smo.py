import numpy as np
from numpy import *
import copy

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


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)

def select_j_rand(i,m):
    #选取alpha
    j=i
    while j==i:
        j=int(random.uniform(0,m))
    return j

def clip_alpha(aj,H,L):
    #修剪alpha
    if aj>H:
        aj=H
    if L>aj:
        aj=L

    return aj

def smo(data_mat_In,class_label,C,toler,max_iter):
    #转化为numpy的mat存储
    data_matrix=np.mat(data_mat_In)
    label_mat=np.mat(class_label).transpose()
    #初始化b,统计data_matrix的维度
    b=0
    m,n=np.shape(data_matrix)
    #初始化alpha,设为0
    alphas=np.mat(np.zeros((m,1)))
    #初始化迭代次数
    iter_num=0
    #最多迭代max_iter次
    while iter_num <max_iter:
        alphas_pairs_changed=0
        for i in range(m):
            #计算误差Ei
            fxi=float(np.multiply(alphas,label_mat).T*(data_matrix*data_matrix[i,:].T))+b
            Ei=fxi-float(label_mat[i])
            #优化alpha,松弛向量
            if (label_mat[i]*Ei< -toler and alphas[i] <C) or (label_mat[i]*Ei >toler and alphas[i]>0):
                #随机选取另一个与alpha_j成优化的alpha_j
                j=select_j_rand(i,m)
                #1.计算误差Ej
                fxj=float(np.multiply(alphas,label_mat).T*(data_matrix*data_matrix[j,:].T))+b
                Ej=fxj-float(label_mat[j])
                #保存更新前的alpha，deepcopy
                alpha_i_old=copy.deepcopy(alphas[i])
                alpha_j_old=copy.deepcopy(alphas[j])
                #2.计算上下界L和H
                if label_mat[i]!=label_mat[j]:
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else :
                    L=max(0,alphas[j]+alphas[i]+C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:
                    print("L==H")
                    continue
                #3.计算eta
                eta=2.0*data_matrix[i,:]*data_matrix[j,:].T-data_matrix[i,:]*data_matrix[i,:].T-data_matrix[j,:]*data_matrix[j,:].T
                if eta>=0:
                    print("eta >= 0")
                    continue
                #4.更新alpha_j
                alphas[j]-=label_mat[j]*(Ei-Ej)/eta
                #5.修剪alpha_j
                alphas[j]=clip_alpha(alphas[j],H,L)
                if abs(alphas[j]-alphas[i]) <0.001:
                    print("alphas_j变化太小")
                    continue
                #6.更新alpha_i
                alphas[i]+=label_mat[j]*label_mat[i]*(alpha_i_old-alphas[j])
                #更新b_1和b_2
                b_1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b_2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                #根据b_1和b_2更新b
                if 0<alphas[i] and C >alphas[i]:
                    b=b_1
                elif 0<alphas[j] and C > alphas[j]:
                    b=b_2
                else:
                    b=(b_1+b_2)/2
                #统计优化次数
                alphas_pairs_changed+=1
                #打印统计信息
                print("第%d次迭代 样本 ：%d ，alpha 优化次数 :%d"%(iter_num,i,alphas_pairs_changed))
        #更新迭代次数
        if alphas_pairs_changed == 0:
            iter_num += 1
        else :
            iter_num=0
        print("迭代次数：%d"%iter_num)

    return b,alphas

def caluelate_w(data_mat,label_mat,alphas):
    #计算w
    alphas=np.array(alphas)
    data_mat=np.array(data_mat)
    label_mat=np.array(label_mat)
    w=np.dot((np.tile(label_mat.reshape(1,-1).T ,(1,2))*data_mat).T,alphas)
    return w.tolist()

def prediction(test,w,b):
    test=np.mat(test)
    result=[]

    for i in test:
        if i*w+b >0:
            result.append(1)
        else :
            result.appens(-1)

    print(result)

    return result


class SVM():
    """
    SVM模型。
    """

    def __init__(self):
        pass


    def train(self, data_train):
        """
        训练模型。
        """
        x_train = data_train[:, :2]  # feature [x1, x2]
        t_train = data_train[:, 2]  # 真实标签
        b,alphas=smo(x_train,list(t_train),0.6,0.001,40)
        print(b)
        print(alphas)
        w=caluelate_w(x_train,t_train,alphas)
        print(w)



    def predict(self, x):
        """
        预测标签。
        """
        result=prediction(x,w,b)
        return result





if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
