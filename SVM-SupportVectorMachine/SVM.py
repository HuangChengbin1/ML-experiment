import random
import matplotlib.pyplot as plt
import numpy as np

class SVM:

    def __init__(self):
        self

    def clip_alpha(self, alpha, L, H):
        """
        clip alpha between L and H
        """
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha

    def select_j(self, i, m):
        """
        Randomly select num remaining in m except for i

        :param i:   alpha i index
        :param m:   data matrix m dimension
        :return:    j
        """
        l = list(range(m))
        seq = l[: i] + l[i + 1:]
        return random.choice(seq)

    def simple_SMO(self, data, labels, C, tolerance, max_iter):
        """
        Simplified SMO algorithm implementation

        :param data:        data features
        :param labels:      data labels
        :param C:           soft interval constant
        :param tolerance:   tolerance
        :param max_iter:    maximum number of iterations
        :return:
        """
        # matrixing
        data = np.mat(data)
        labels = np.mat(labels).transpose()

        # sample nums --m
        # dimension --n
        m, n = np.shape(data)

        # initialize bias b and Lagrange parameter alpha
        b = 0
        alphas = np.mat(np.zeros((m, 1)))
        print(alphas.shape)

        iter = 0
        while iter < max_iter:
            alpha_pair_changed = 0
            for i in range(m):
                # calc the pred and err of alphas[i]
                # 将复杂的问题转化为二阶问题，抽取alphas[i], alphas[j]进行优化，将大问题转为小问题
                pred_Xi = float(np.multiply(alphas, labels).T * (data * data[i, :].T)) + b
                Ei = pred_Xi - float(labels[i])

                # Optimize if the KKT condition is not met
                if ((labels[i] * Ei < -tolerance) and (alphas[i] < C)) or \
                    ((labels[i] * Ei > tolerance) and (alphas[i] > 0)):

                    # Randomly select non-i alphas[j]
                    j = self.select_j(i, m)
                    print(f'i: {i}, j: {j}')

                    # calc the pred and err of alphas[j]
                    pred_Xj = float(np.multiply(alphas, labels).T * (data * data[j, :].T)) + b
                    print("pred_Xj: ", pred_Xj)
                    Ej = pred_Xj - float(labels[j])

                    # Value before optimization
                    alphas_i_old = alphas[i].copy()
                    alphas_j_old = alphas[j].copy()

                    # 二变量优化问题
                    if labels[j] != labels[i]:
                        # 如果是异侧，相减ai - aj = k，那么定义域未[k, C + k]
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        # 如果是同侧，相加ai + aj = k，那么定义域未[k - C, k]
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])

                    if L == H:
                        print("定义域确定，没有优化空间")
                        continue

                    # Optimize alphas[j]
                    # calc it's eta = 2 * a * b - a^2 - b^2, if eta >= 0 then exit loop
                    eta = 2.0 * data[i, :] * data[j, :].T - data[i, :] * data[i, :].T - data[j, :] * data[j, :].T
                    if eta >= 0:
                        print("eta >= 0")
                        continue
                    print("eta: ", eta)
                    print("Ei: ", Ei)
                    print("Ej: ", Ej)

                    print(labels[j] * (Ei - Ej) / eta)
                    print(alphas[j])
                    # aj -= yj * (Ei - Ej) / eta
                    alphas[j] -= labels[j] * (Ei - Ej) / eta
                    # Adjust domain
                    alphas[j] = self.clip_alpha(alphas[j], L, H)
                    if abs(alphas[j] - alphas_j_old) < 0.00001:
                        print("J not moving enough")
                        continue

                    # optimize alphas[i], ai += yj * yi * (a_j_old - aj)
                    alphas[i] += labels[j] * labels[i] * (alphas_j_old - alphas[j])

                    # bi = b - Ei - yi * (ai - a_i_old) * xi * xi.T - yj * (aj - a_j_old) * xi * xj.T
                    b1 = b - Ei - labels[i] * (alphas[i] - alphas_i_old) * data[i, :] * data[i, :].T - labels[j] * (alphas[j] - alphas_j_old) * data[i, :] * data[j, :].T
                    # bj = b - Ej - yi * (ai - a_i_old) * xi * xj.T - yj * (aj - a_j_old) * xj * xj.T
                    b2 = b - Ej - labels[i] * (alphas[i] - alphas_i_old) * data[i, :] * data[j, :].T - labels[j] * (alphas[j] - alphas_j_old) * data[j, :] * data[j, :].T

                    # 判断哪个模型常量值符合定义域规则， 不满足就暂时赋予 b = (bi + bj) / 2.0
                    if 0 < alphas[i] < C:
                        b = b1
                    elif 0 < alphas[j] < C:
                        b = b2
                    else :
                        b = (b1 + b2) / 2.0
                    alpha_pair_changed += 1
                    print(f"iter: {iter}, i: {i}, pairs changed: {alpha_pair_changed}")
            # Check whether alpha has been updated
            if alpha_pair_changed == 0:
                iter += 1
            else :
                iter = 0
            print(f"iter: {iter}")
        return b, alphas

    def get_w(self, alphas, data, labels):
        """
        Calculate w based on alpha

        :param alphas:  Lagrange multiplier
        :param data:    feature
        :param labels:  label
        :return:        w   regression
        """
        X = np.mat(data)
        labels = np.mat(labels).transpose()
        m, n = np.shape(X)
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alphas[i] * labels[i], X[i, :].T)
        print("w: ", w)
        return w

    def plot_SVM(self, x, y, w, b, alphas):
        """
        Draw SVM

        :param x:       features
        :param y:       labels
        :param w:       regression coef
        :param b:       bias
        :param alphas:  Lagrange parameter
        :return:
        """
        x = np.mat(x)
        y = np.mat(y)

        # convert b to array    --[][0]
        b = np.array(b)[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(x[:, 0].flatten().A[0], x[:, 1].flatten().A[0])

        x_ = np.arange(-1.0, 10.0, 0.1)
        # x * w + b = 0 ==> w0 * x1 + w1 * x2 + b = 0 ==> y = (-b - w0 * x1) / w1
        y_ = (-b - w[0, 0] * x_) / w[1, 0]
        ax.plot(x_, y_)

        for i in range(np.shape(y[0, :])[1]):
            if y[0, i] > 0:
                ax.plot(x[i, 0], x[i, 1], 'bo')
            else:
                ax.plot(x[i, 0], x[i, 1], 'ro')

        # SV
        for i in range(100):
            if alphas[i] > 0.0:
                ax.scatter(x[i, 0], x[i, 1], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='g')

        plt.show()
