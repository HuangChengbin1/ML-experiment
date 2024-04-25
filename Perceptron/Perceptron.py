import numpy as np

class Perceptron:

    def __init__(self):
        self.W = None       # weight
        self.theta = None   # threshold
        self.lr = None      # learning rate
        self.errors = None  # errors num

    def __init__(self, input_size, lr=0.1):
        self.W = np.zeros(input_size)
        self.theta = 0
        self.lr = lr
        self.errors = []

    def predict(self, X):
        """
        Predict X by function sign

        :param X:   feature mat
        :return:    1 or -1
        """
        return np.sign(np.dot(X, self.W) - self.theta)                  # y = f(WX - theta), the f here is sign

    def train(self, X, y, epochs):
        """
        Train model

        :param X:       feature mat
        :param y:       labels
        :param epochs:  max number of iteration
        :return:
        """
        y = np.where(y == 0, -1, y)     # converts label 0 to 1
        for epoch in range(epochs):
            errors = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                if prediction != y[i]:
                    self.W += self.lr * (y[i] - prediction) * X[i]      # w = w + (eta * (y - y_hat) * x)
                    self.theta -= self.lr * (y[i] - prediction)         # theta = theta - eta * (y - y_hat), if y_hat > y then theshold increment, else theshold reduction
                    errors += 1
                print(f'Epoch: {epoch + 1}, Update: {i + 1}, W: {self.W}, theta: {self.theta}')
            self.errors.append(errors)