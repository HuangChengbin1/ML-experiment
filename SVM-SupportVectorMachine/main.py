import numpy as np
import pandas as pd
from SVM import SVM

filepath = "dataset\\dataset.txt"
sep = '\t'
data = pd.read_csv(filepath, sep=sep, header=None)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
svm = SVM()
b, alphas = svm.simple_SMO(x, y, 0.6, 0.001, 40)
w = svm.get_w(alphas, x, y)
svm.plot_SVM(x, y, w, b, alphas)
