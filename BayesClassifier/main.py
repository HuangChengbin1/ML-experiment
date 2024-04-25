import pandas as pd
import numpy as np

filepath = "dataset\\data.txt"
sep = '\t'

data = pd.read_csv(filepath, sep=sep, header=None)
size_mapping = {'S': 1, 'M': 2, 'L': 3}
data.iloc[:, 1] = data.iloc[:, 1].map(size_mapping)
print("train_data after mapping: \n", data)

test_x = pd.DataFrame({0: 2, 1: 'S'}, index=[0])
test_x.iloc[:, 1] = test_x.iloc[:, 1].map(size_mapping)
print("test_data after mapping: \n", test_x)

from BayesClassifier import BayesClassifier
bc = BayesClassifier()
test = bc.naive_bayes_classifier(data, test_x)
print("test_data with predict result: \n", test)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(data.iloc[:, :-1], data.iloc[:, -1])
pred_x = clf.predict(test_x.iloc[:, :-1])
print("predict result by sklearn.naive_bayes GaussionNB: \n", pred_x)
