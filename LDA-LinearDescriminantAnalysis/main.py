import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis as LDA

#读取数据
data = pd.read_csv("D:\\机器学习\\实验\\实验2\\输血服务中心数据集\\blood_data.txt", sep=',', header=None)
data.columns = ['Recency', 'Frequency', 'Monetary', 'Time', 'BD']
print(data)

#划分训练集和测试集
data_shuffled = data.sample(frac=1, random_state=42)
train_data = data_shuffled.head(600)
test_data = data_shuffled.tail(148)

train_data_X = train_data[['Recency', 'Frequency', 'Monetary', 'Time']]
train_data_Y = train_data['BD']
test_data_X = test_data[['Recency', 'Frequency', 'Monetary', 'Time']]
test_data_Y = test_data['BD']

#实例化LDA
lda = LDA()
lda.train(train_data_X, train_data_Y)

#对数据集进行线性判别分析
#分类情况
predicted_y = lda.predict(test_data_X)
print(predicted_y)
print(test_data_Y)
print("LDA's accuracy is ", np.mean(predicted_y == test_data_Y.values))

#投影方向
w = lda.w
print("投影方向: \n", w)

#降维后的数据集画图
X_self = lda.transform(test_data_X)

#LDA降维后的数据
plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 1)
plt.title("Self-defined LDA")
plt.scatter(X_self[predicted_y == 0], X_self[predicted_y == 0], c='r', marker='+', label='class-0')
plt.scatter(X_self[predicted_y == 1], X_self[predicted_y == 1], c='b', marker='o', label='class-1')
plt.legend()

#使用sklearn对比
print("\nsklearn:\n")

from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn import metrics

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data[['Recency', 'Frequency', 'Monetary', 'Time']], data['BD'], test_size=0.2, random_state=0)
lda_model = skLDA(solver='eigen', shrinkage=None).fit(data[['Recency', 'Frequency', 'Monetary', 'Time']], data['BD'])
y_pred = lda_model.predict(X_test)
print("混淆矩阵:\n", metrics.confusion_matrix(Y_test, y_pred))
print("分类报告:\n", metrics.classification_report(Y_test, y_pred))
print("投影方向:\n", lda_model.coef_)
print("精确度:\n", lda_model.score(X_test, Y_test))
X_s = lda_model.transform(X_test)
plt.subplot(1, 2, 2)
plt.title('sklearn LDA')
plt.scatter(X_s[y_pred == 0], X_s[y_pred == 0], c='r', marker='+', label='class-0')
plt.scatter(X_s[y_pred == 1], X_s[y_pred == 1], c='b', marker='o', label='class-1')
plt.legend()
plt.show()