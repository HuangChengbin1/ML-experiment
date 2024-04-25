import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron

file_path = "D:\\机器学习\\实验\\实验4\\ch05研讨+实验\\感知机数据集\\perceptron_data.txt"
sep = '\t'

data = pd.read_csv(file_path, sep=sep, header=None)
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

perceptron = Perceptron(input_size=X.shape[1])
perceptron.train(X, Y, epochs=15)
print("The number of errors per iteration: \n", perceptron.errors)

plt.scatter(X[:, 0], X[:, 1], c=Y)

# W^T * X - theta = 0
x = np.linspace(-5, 5, 100)
y = (-perceptron.W[0] * x + perceptron.theta) / perceptron.W[1]

plt.plot(x, y, '-r', label='Linear Hyperplane')
plt.legend()
plt.show()