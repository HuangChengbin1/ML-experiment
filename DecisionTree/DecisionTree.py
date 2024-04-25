import numpy as np
import pandas as pd

class DecisionTree:

    def __init__(self):
        self.data = None
        self.path = None
        self.column_count = None
        self.data_label = None
        self.info_gain_rate = {}
        self.info_gain = {}

    def load_data(self, path):
        self.path = path
        data = pd.read_csv(self.path, sep='\t', header=None)
        data = data[0].str.split(expand=True)
        data.set_index(0, inplace=True)
        data.columns = ['age', 'prescription', 'astigmatic', 'tearRate', 'label']
        self.data = data
        column_count = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])
        self.column_count = column_count
        self.depth = 0
        # print(data)

    # 计算信息熵
    def cal_information_entropy(self, data):
        data_label = data.iloc[:, -1]
        label_class = data_label.value_counts()
        # print(label_class)
        Ent = 0
        for k in label_class.keys():
            p_k = label_class[k] / len(data_label)
            Ent += -p_k * np.log2(p_k)
        return Ent

    # 计算给定数据属性a的信息增益
    def cal_information_gain(self, data, a):
        Ent = self.cal_information_entropy(data)
        feature_class = data[a].value_counts()  # 特征有多少种可能
        gain = 0
        for v in feature_class.keys():
            weight = feature_class[v] / data.shape[0]
            Ent_v = self.cal_information_entropy(data.loc[data[a] == v])
            gain += weight * Ent_v
        self.info_gain[a] = Ent - gain
        # print("gain——", a, ": ", Ent - gain)
        return Ent - gain

    # 计算信息增益率
    def cal_gain_ratio(self, data, a):
        # 先计算固有值intrinsic_value
        IV_a = 0
        feature_class = data[a].value_counts()
        for v in feature_class.keys():
            weight = feature_class[v] / data.shape[0]
            IV_a += -weight * np.log2(weight)
        gain_ratio = self.cal_information_gain(data, a) / IV_a
        self.info_gain_rate[a] = gain_ratio
        # print("gain rate——", a, ": ", gain_ratio)
        return gain_ratio

    # 获取标签最多的一类
    def get_most_label(self, data):
        # data_label = self.data_label
        data_label = data.iloc[:, -1]
        label_sort = data_label.value_counts(sort=True)
        return label_sort.keys()[0]

    # 挑选最优特征，即信息增益大于平均水平的特征中选取增益率最高的特征
    def get_best_feature(self, data):
        features = data.columns[:-1]
        res = {}
        for a in features:
            temp = self.cal_information_gain(data, a)
            gain_ration = self.cal_gain_ratio(data, a)
            res[a] = (temp, gain_ration)
        res = sorted(res.items(), key=lambda x:x[1][0], reverse=True)   # 按信息增益排名
        res_avg = sum([x[1][0] for x in res]) / len(res)    # 信息增益平均水平
        good_res = [x for x in res if x[1][0] >= res_avg]   # 选取信息增益高于平均水平的特征
        result = sorted(good_res, key=lambda x:x[1][1], reverse=True)   # 将信息增益高的特征按照信息增益率排名
        return result[0][0]     # 返回高信息增益中增益率最大的特征

    # 将数据转化为（属性值: 数据）的元组形式返回，并删除之前的特征列
    def drop_exist_feature(self, data, best_feature):
        attr = pd.unique(data[best_feature])
        new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
        new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
        return new_data

    def create_tree(self, data, depth=0):
        data_label = data.iloc[:, -1]
        # 只有一类
        if len(data_label.value_counts()) == 1:
            return data_label.values[0]
        # 所有数据的特征值一样，选样本最多的类作为分类结果
        if all(len(data[i].value_counts()) == 1 for i in data.iloc[:, :-1].columns):
            return self.get_most_label(data)
        best_feature = self.get_best_feature(data)  # 根据信息增益得到的最优划分特征
        best_feature_gain_rate = self.info_gain_rate[best_feature]
        if depth == 0:
            print("各节点对应特征的信息增益率: ")
        print("d{} {:<15}——gain rate——   {:<}".format(depth, best_feature, best_feature_gain_rate))
        Tree = {best_feature: {}}   # 用字典存储决策树
        exist_vals = pd.unique(data[best_feature])  # 当前数据下最佳特征的取值
        # 若特征的取值比原来的少
        if len(exist_vals) != len(self.column_count[best_feature]):
            no_exist_attr = set(self.column_count[best_feature]) - set(exist_vals)  # 少的特征
            for no_feature in no_exist_attr:
                Tree[best_feature][no_feature] = self.get_most_label(data)  # 缺失的特征分类为当前类别最多的
        # 根据特征值的不同递归创建决策树
        for item in self.drop_exist_feature(data, best_feature):
            Tree[best_feature][item[0]] = self.create_tree(item[1], depth + 1)
        return Tree

    def predict(self, Tree, test_data):
        first_feature = list(Tree.keys())[0]
        second_dict = Tree[first_feature]
        input_first = test_data.get(first_feature)
        input_value = second_dict[input_first]
        # 判断分支还是不是字典
        if isinstance(input_value, dict):
            class_label = self.predict(input_value, test_data)
        else:
            class_label = input_value
        return class_label