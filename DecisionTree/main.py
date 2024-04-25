from DrawTree import createPlot
from DecisionTree import DecisionTree
from mapping import mapping

t = DecisionTree()  # 创建决策树模型实例
t.load_data("dataset\\lenses_data.txt")    # 加载数据集
data = t.data
data = data.reset_index(drop=True)  # 重置索引列
print(data)

test_row = data.sample(n=1, random_state=42)    # 从数据集中随机抽取一行作为测试数据
temp = test_row     # 备份数据，用于sklearn的DecisionTreeClassifier模型使用，因为后续将test_row转化为了字典，所以先备份
print(test_row)

train_data = data.drop(test_row.index)  # 从数据集中删除测试数据，构成训练集
tree = t.create_tree(train_data)        # 构建决策树模型

test_row = test_row.reset_index(drop=True).to_dict(orient='records')[0]     # 将test_row转化为字典
# print(test_row)
predicted = t.predict(tree, test_row)     # 预测
print("预测结果: ", predicted, "实际标签: ", test_row['label'])
print(tree)

mapped_tree = mapping(tree)     # 将构建完的树中对应的数字类别映射为名称
print(mapped_tree)

createPlot(mapped_tree)         # 将构建的决策树可视化

'''
调用sklearn中的DecisionTreeClassifier对比构建出的树
'''
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree
import matplotlib.pyplot as plt

dtc = DecisionTreeClassifier(criterion="entropy")   # 创建实例

# 训练集和测试集
x_train = train_data.iloc[:, :-1]
x_test = train_data.iloc[:, -1]
y_train = temp.iloc[:, :-1]
y_test = temp.iloc[:, -1]

dtc.fit(x_train, x_test)        # 训练模型
pred = dtc.predict(y_train)     # 预测结果

print("实际标签: ", y_test.iloc[-1])
print("预测结果: ", pred)

# 获取构建的决策树模型
ttt = export_text(dtc, feature_names=['age', 'prescription', 'astigmatic', 'tearRate'])
print("决策树: \n", ttt)

fig = plt.figure(figsize=(25, 25))
_ = plot_tree(
    dtc,
    feature_names = ['age', 'prescription', 'astigmatic', 'tearRate'],
    class_names = ['hard', 'soft', 'no lenses'],
    filled = True
)
fig.savefig("sklearn_DT.png")
