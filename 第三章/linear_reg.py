import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    '''载入数据'''
    xys=[]
    with open(filename ,'r')  as f:
        for line in f:
            xys.append(map(float,line.strip().split()))
        xs,ys=zip(*xys)
        return np.asarray(xs),np.asarray(ys)



def evluate(ys,ys_pred):
    std=np.sqrt(np.mean(np.abs(ys-ys_pred)**2))
    return std


def main(x_train,y_train):
    phi0=np.expand_dims(np.ones_like(x_train),axis=1)
    print('phi0',phi0)
    phi1=np.expand_dims(x_train,axis=1)
    print('phi1',phi1)
    phi=np.concatenate([phi0,phi1],axis=1)
    print('phi',phi)
    w=np.dot(np.linalg.pinv(phi),y_train).T
    print(w)

    def f(x):
        phi0=np.expand_dims(np.ones_like(x),axis=1)
        phi1=np.expand_dims(x,axis=1)
        phi=np.concatenate([phi0,phi1],axis=1)
        y=np.dot(phi,w)
        return y
        pass

    return f


if __name__=='__main__':
    train_file= 'data/Utrain.txt'
    test_file='result.txt'

    x_train,y_train=load_data(train_file)
    x_test,y_test=load_data(test_file)
    print(x_train.shape)
    print(x_test.shape)

    f=main(x_train,y_train)

    y_test_pred=f(x_test)
    print('y_test_pred',y_test_pred)
    print('y_test_pred.shape',y_test_pred.shape)
    std=evluate(y_test,y_test_pred)
    print('预测值与真实值的标准差：{:.1f}'.format(std))

    plt.plot(x_train,y_train,'ro',markersize=3)
    plt.plot(x_test,y_test,'k')
    plt.plot(x_test,y_test_pred)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend(['train','test','pred'])
    plt.show()



