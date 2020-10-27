import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)


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


def main(x_train, y_train):
    """
    训练模型，并返回从x到y的映射。

    """
    basis_func = identity_basis
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)
    phi1 = basis_func(x_train)
    phi = np.concatenate([phi0, phi1], axis=1)

    # ==========
    # todo '''计算出一个优化后的w，请分别使用最小二乘法以及梯度下降两种办法优化w'''
    # ==========#梯度下降法优化w
    # 两种终止条件
    loop_max = 10000  # 最大迭代次数(防止死循环)
    epsilon = 1e-3
    # 初始化权值
    np.random.seed(0)
    w = np.random.randn(2)
     # w = np.zeros(2)
    alpha = 0.001  # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
    diff = 0.
    error = np.zeros(2)
    count = 0  # 循环次数
    finish = 0  # 终止标志
    # -------------------------------------------随机梯度下降算法---------------------------------------------------------
    while count < loop_max:
        count += 1

        # 遍历训练数据集，不断更新权值
        for i in range(m):
            diff = np.dot(w, x_train[i]) - y_train[i]  # 训练集代入,计算误差值

            # 采用随机梯度下降算法,更新一次权值只使用一组训练数据
            w = w - alpha * diff * x_train[i]

            # ------------------------------终止条件判断-----------------------------------------
            # 若没终止，则继续读取样本进行处理，如果所有样本都读取完毕了,则循环重新从头开始读取样本进行处理。

        # ----------------------------------终止条件判断-----------------------------------------
        # 注意：有多种迭代终止条件，和判断语句的位置。终止判断可以放在权值向量更新一次后,也可以放在更新m次后。
        if np.linalg.norm(w - error) < epsilon:     # 终止条件：前后两次计算出的权向量的绝对误差充分小
            finish = 1
            break
        else:
            error = w
    print ('loop count = %d' % count,  '\tw:[%f, %f]' % (w[0], w[1]))

    def f(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        phi = np.concatenate([phi0, phi1], axis=1)
        y = np.dot(phi, w)
        return y

    return f


def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std


# 程序主入口（建议不要改动以下函数的接口）
if __name__ == '__main__':
    train_file = 'Utrain.txt'
    test_file = 'test.txt'
    # 载入数据
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(x_train.shape)
    print(x_test.shape)

    # 使用线性回归训练模型，返回一个函数f()使得y = f(x)
    f = main(x_train, y_train)

    y_train_pred = f(x_train)
    std = evaluate(y_train, y_train_pred)
    print('训练集预测值与真实值的标准差：{:.1f}'.format(std))

    # 计算预测的输出值
    y_test_pred = f(x_test)
    # 使用测试集评估模型
    std = evaluate(y_test, y_test_pred)
    print('预测值与真实值的标准差：{:.1f}'.format(std))

    # 显示结果
    plt.plot(x_train, y_train, 'ro', markersize=3)
    #     plt.plot(x_test, y_test, 'k')
    plt.plot(x_test, y_test_pred, 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend(['train', 'test', 'pred'])
    plt.show()