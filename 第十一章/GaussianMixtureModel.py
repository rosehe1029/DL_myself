import numpy as np
from sklearn import datasets


class GMM:
    def __init__(self, k=3):
        self.k = k  # 定义聚类个数,默认值为3
        # 声明变量
        self.alpha = np.ones(1)
        self.mu = 0
        self.sigma2 = np.ones(1)

    def rand_theta(self, dataset):
        """
        初始化模型参数
        :param dataset: 数据集,m*1， np.array
        :return: alpha（k,), mu(k,), sigma2(k,)
        """
        alpha = np.ones(self.k) / self.k  # 初始化每个alpha为1/k
        mu = dataset[np.random.choice(dataset.shape[0], self.k, replace=False), 0]  # 随机选择k个样本作为均值
        sigma2 = np.ones(self.k) / 10.0  # 初始化每个sigma2为0.1,sigma2意为sigma的平方
        return alpha, mu, sigma2

    def cal_gamma(self, x, alpha, mu, sigma2):
        """
        计算数据x在每个高斯分布下的gamma（根据推导，此lambda就是后验概率）
        :param x: 一元数据（即单个实数，X属于R）
        :param alpha: 模型参数，相当于各高斯分布的权重
        :param mu: 模型参数，高斯分布的参数mu
        :param sigma2: 模型参数，高斯分布的参数sigma2
        :return: gamma（k,）
        """
        gamma = np.zeros(self.k)  # 定义好gamma，此处针对一个样本数据
        temp = np.sum(alpha / (np.sqrt(2 * np.pi * sigma2)) * np.exp(-(x - mu) ** 2 / (2 * sigma2)))
        for i in range(self.k):
            gamma[i] = alpha[i] / (np.sqrt(2 * np.pi * sigma2[i])) * \
                       np.exp(-(x - mu[i]) ** 2 / (2 * sigma2[i])) / temp  # 根据公式计算
        return gamma

    def training(self, dataset):
        """
        一元高斯混合分布算法
        :param dataset: 数据集，m*1， np.array
        :return: alpha（k,), miu(k,), delta2(k,), 样本的信息矩阵（m*2）,存储所属高斯分布的索引和后验概率。
        """
        m = dataset.shape[0]
        self.alpha, self.mu, self.sigma2 = self.rand_theta(dataset)  # 初始化参数
        gamma = np.zeros((m, self.k))  # 定义好gamma，此处针对所有样本数据
        cluster = np.ones((m, 2)) * -1  # 初始化最终需要输出的样本信息：所属高斯分布的索引和后验概率,*-1是为了区分真正的索引和概率
        cluster_changed = True  # 循环控制开关

        while cluster_changed:
            cluster_changed = False  # 关闭开关
            for i in range(m):
                gamma[i] = self.cal_gamma(dataset[i], self.alpha, self.mu, self.sigma2)
                if cluster[i, 0] != np.argmax(gamma[i]):  # 如果样本所属分布有变动,那么打开开关,继续循环
                    cluster_changed = True
                    cluster[i, :] = np.argmax(gamma[i]), gamma[i].max()  # 更新样本信息
            self.alpha = np.sum(gamma, axis=0) / m  # 根据公式更新alpha， 利用array数组，计算非常方便
            self.mu = np.sum((gamma * dataset), axis=0) / np.sum(gamma, axis=0)  # 同上
            self.sigma2 = np.sum((gamma * np.power(dataset - self.mu, 2)), axis=0) / np.sum(gamma, axis=0)  # 同上
        return cluster


def test():
    # 使用datasets生成高斯分布的一元数据,可以观察到不错的聚类效果
    dataset = datasets.make_blobs(n_features=1)[0]
    gmm = GMM(k=3)
    cluster = gmm.training(dataset)
    print(cluster)


test()
