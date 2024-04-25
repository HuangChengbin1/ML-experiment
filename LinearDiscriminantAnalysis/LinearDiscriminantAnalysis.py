import numpy as np

class LinearDiscriminantAnalysis(object):
    def __init__(self):
        self.Xi_meansVector = None  #每个类别的均值向量
        self.meansVector = None     #整体的均值向量
        self.Xi_covMatrix = None    #每个类别的协方差矩阵
        self.covMatrix = None       #整体的协方差矩阵
        self.X = None               #训练数据
        self.y = None               #训练数据的分类标签
        self.classes = None         #具体类别
        self.priors = None          #每个类别的先验概率
        self.n_samples = None       #训练数据的样本数
        self.n_features = None      #训练数据的特征数
        self.n_components = None    #特征数
        self.w = None               #特征向量

    def params_init(self, X, y):
        #赋值X和y
        self.X, self.y = X, y
        #计算样本数量和特征数量
        self.n_samples, self.n_features = X.shape
        #计算类别值、每个类别的先验概率
        self.classes, yidx = np.unique(y, return_inverse=True)
        self.priors = np.bincount(y) / self.n_samples
        #计算每类的均值
        means = np.zeros((len(self.classes), self.n_features))
        np.add.at(means, yidx, X)
        self.Xi_meansVector = means / np.expand_dims(np.bincount(y), 1)
        #计算每类的协方差矩阵、整体的协方差矩阵
        self.Xi_covMatrix = [np.cov(X[y == group].T) \
                          for idx, group in enumerate(self.classes)]
        self.covMatrix = np.cov(self.X.T)
        #计算总体均值向量
        self.meansVector = np.dot(np.expand_dims(self.priors, axis=0), self.Xi_meansVector)
        return

    def train(self, X, y, n_components = None):
        self.params_init(X, y)
        #类内平均散度
        Sw = self.covMatrix
        #类间平均散度
        Sb = np.cov(self.Xi_meansVector.T) * len(self.Xi_meansVector)

        #广义特征值分解求解投影方向
        la, vectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        la = np.real(la)
        vectors = np.real(vectors)
        #特征值下标从大到小排序
        laIdx = np.argsort(-la)
        #默认选取N-1个特征值的下标
        if n_components == None:
            n_components = len(self.classes) - 1
        #选取特征值和向量
        lambda_index = laIdx[:n_components]
        w = vectors[:, lambda_index]
        self.w = w
        self.n_components = n_components
        return

    #求出投影后的矩阵
    def transform(self, X):
        return np.dot(X, self.w)

    def predict_prob(self, X):
        #求整体协方差的逆
        Sigma = self.covMatrix
        U, S, V = np.linalg.svd(Sigma)
        Sn = np.linalg.inv(np.diag(S))
        Sigman = np.dot(np.dot(V.T, Sn), U.T)
        #线性判别函数
        value = np.dot(X, np.linalg.inv(Sigma)).dot(self.Xi_meansVector.T) - 0.5 * np.dot(self.Xi_meansVector, np.linalg.inv(Sigma)).sum(axis=1)
        return value / np.expand_dims(value.sum(axis=1), 1)

    def predict(self, X):
        pValue = self.predict_prob(X)
        return np.argmax(pValue, axis=1)