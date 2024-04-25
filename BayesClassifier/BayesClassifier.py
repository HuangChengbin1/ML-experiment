import numpy as np
import pandas as pd

class BayesClassifier:

    def __init__(self):
        self.X = None
        self.y = None
        self.classes = None
        self.parameters = None

    def naive_bayes_classifier(self, train_data, test_data):
        """
        naive bayes classifier
        
        :param train_data:  train data
        :param test_data:   test data
        :return:    test data with predict result
        """
        
        labels = train_data.iloc[:, -1].value_counts().index    # type of labels
        mean = []   # mean for per class
        var = []    # variance for per class
        res = []    # prediction for test data

        for i in labels:
            item = train_data.loc[train_data.iloc[:, -1] == i, :]   # extract each class
            m = item.iloc[:, :-1].mean()    # mean for cur class
            v = np.sum((item.iloc[:, :-1] - m)**2) / (item.shape[0])    # variance for cur class
            mean.append(m)
            var.append(v)

        means = pd.DataFrame(mean, index=labels)
        vars = pd.DataFrame(var, index=labels)

        for j in range(test_data.shape[0]):
            iset = test_data.iloc[j, :].tolist()  # cur sample
            iprob = np.exp(-1 * (iset - means)**2 / (vars*2)) / (np.sqrt(2 * np.pi) * vars)     # normal distribution
            prob = 1
            for k in range(test_data.shape[1] - 1):
                prob *= iprob[k]
                cla = prob.index[np.argmax(prob.values)]    # class with greatest prob
            res.append(cla)
        test_data['predict'] = res
        return test_data
