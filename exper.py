from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from TreeInTree import TnT, Node
from sklearn.metrics import accuracy_score
from pydotplus import graph_from_dot_data
import numpy as np


def compare_tnt_cart(X, Y, ccp_alpha, feature_names, class_names):
    print('-'*50)
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(\
        X, Y, test_size=0.1, random_state=10)
    tnt = TnT(N1=2, N2=2, ccp_alpha=ccp_alpha, random_state=0)
    tnt.fit(trX=train_inputs, trY=train_labels)
    prediction_train = tnt.predict(teX=train_inputs)
    accuracy_train = accuracy_score(train_labels, prediction_train)
    print("TnT accuracy (train set):", accuracy_train)

    prediction_test = tnt.predict(teX=val_inputs)
    accuracy_test = accuracy_score(val_labels, prediction_test)
    print("TnT accuracy (test set):", accuracy_test)
    internal_cnt, leaf_cnt = tnt.check_complexity()
    print("TnT model complexity: ", internal_cnt, " internal nodes, ", leaf_cnt, " leaf nodes")



    cart = DecisionTreeClassifier(max_leaf_nodes=internal_cnt+1, random_state=0)
    cart.fit(X=train_inputs, y=train_labels)
    prediction_train = cart.predict(train_inputs)
    accuracy_train = accuracy_score(train_labels, prediction_train)
    print("CART accuracy (train set):", accuracy_train)
    prediction_test = cart.predict(val_inputs)
    accuracy_test = accuracy_score(val_labels, prediction_test)
    print("CART accuracy (test set):", accuracy_test)
    print("CART model complexity: ", internal_cnt, " internal nodes, ", internal_cnt+1, " leaf nodes")
    print('-'*50)

    dot_data = extract_dotdata_from_tnt(tnt=tnt, 
                                        feature_names=feature_names,  
                                        class_names=class_names,
                                        X=train_inputs)
    # print(dot_data)
    graph = graph_from_dot_data(dot_data)
    graph.write_png('tnt.png')

    dot_data = export_graphviz(cart, out_file=None, 
                                feature_names=feature_names,  
                                class_names=class_names,
                                filled=True,
                                impurity=False)
    dot_data = dot_data.replace('black', 'white', 1)
    graph = graph_from_dot_data(dot_data)
    graph.write_png('cart.png')
    

def extract_dotdata_from_tnt(tnt:TnT, feature_names, class_names, X):
    colors = ['#e58139', '#efe6fc', '#51e890', '#843ee6', '#67e539', '#d65bdc'\
            '#ee6688', '#ee6677', '#eedd33', '#eedd22', '#eedd11', '#2d7e95', \
            '#0dd2a3', '#813f5a', '#c0c623', '#7e6ed4', '#04ee51', '#b08f6e', \
            '#421c1a']
    node_all, _ = tnt.graph_traverse(teX=X)
    node_map = {}
    for i, node in enumerate(node_all):
        node_map[node] = i
    if type(class_names) is not list:
        class_names = class_names.tolist()
    info = ''
    for i, node in enumerate(node_all):
        if node.left is not None and node.right is not None: # not leaf
            if type(node.threshold) in [float, np.float64]:
                label_info = f'{feature_names[node.feature_index]} <= {node.threshold:.3f}'
            else:
                label_info = f'{feature_names[node.feature_index]} <= {node.threshold:.3f}'
            color = colors[node.feature_index]
        else:
            label_info = f'label = {class_names[node.label]}'
            color = colors[node.label]
        tmp = f'{i} [label=\"{label_info}\", fillcolor=\"{color}\"] ;'
        info += f'{tmp}\n'
    for i, node in enumerate(node_all):
        if node.left is not None and node.right is not None: # not leaf
            li, ri = node_map[node.left], node_map[node.right]
            info += f'{i} -> {li} ;\n{i} -> {ri} ;\n'
    return 'digraph Tree {\n' + \
        'node [shape=box, style="filled", color="white", fontname="helvetica"] ;edge [fontname="helvetica"] ;\n' + \
            info + \
                '}'



def iris():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    compare_tnt_cart(X=X, Y=Y, ccp_alpha=0.01, 
                        feature_names=iris.feature_names,
                        class_names=iris.target_names)

def f():
    X = np.array([[0, 0], [0, 1], [0, 2],
                    [1, 0], [1, 1], [1, 2],
                    [2, 0], [2, 1], [2, 2]])
    Y = np.array([0, 1, 0,
                    1, 1, 1,
                    0, 1, 0])
    compare_tnt_cart(X=X, Y=Y, ccp_alpha=0.01, 
                        feature_names=np.array(['t1', 't1']),
                        class_names=np.array(['triangle', 'cube']))

iris()