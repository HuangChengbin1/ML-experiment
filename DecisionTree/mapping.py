# 定义特征名称的映射字典
feature_mapping = {
    'age': {'1': 'young', '2': 'pre-presbyopic', '3': 'presbyopic'},
    'prescription': {'1': 'myope', '2': 'hypermetrope'},
    'astigmatic': {'1': 'no', '2': 'yes'},
    'tearRate': {'1': 'reduced', '2': 'normal'}
}
label_mapping = {'1': 'hard', '2': 'soft', '3': 'no lenses'}

def map_tree(tree):
    if isinstance(tree, dict):
        for key, value in tree.items():
            if key in feature_mapping:
                tree[key] = {feature_mapping[key][k]: v for k, v in value.items()}
            map_tree(value)
    return tree

def map_leaf(tree):
    if isinstance(tree, dict):
        for key, value in tree.items():
            tree[key] = map_leaf(value)
    elif tree in label_mapping:
        tree = label_mapping[tree]
    return tree

def mapping(tree):
    mapped_tree = map_tree(tree)
    mapped_tree = map_leaf(mapped_tree)
    return mapped_tree