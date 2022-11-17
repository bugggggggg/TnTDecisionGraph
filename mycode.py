from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from TreeInTree import TnT, Node
from sklearn.metrics import accuracy_score
from pydotplus import graph_from_dot_data
import numpy as np
import time
import pandas as pd


def compare_tnt_cart(X, Y, ccp_alpha, feature_names, class_names, dataset_name, draw=True):
    print('-'*25+'tnt'+'-'*25)
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(\
        X, Y, test_size=0.1, random_state=10)
    tnt = TnT(N1=2, N2=2, ccp_alpha=ccp_alpha, random_state=0)
    
    start_time = time.time()
    tnt.fit(trX=train_inputs, trY=train_labels)
    end_time = time.time()
    prediction_train = tnt.predict(teX=train_inputs)
    accuracy_train = accuracy_score(train_labels, prediction_train)
    print("TnT accuracy (train set):", accuracy_train)

    prediction_test = tnt.predict(teX=val_inputs)
    accuracy_test = accuracy_score(val_labels, prediction_test)
    print("TnT accuracy (test set):", accuracy_test)
    internal_cnt, leaf_cnt = tnt.check_complexity()
    print(f"train time: {end_time-start_time:.2f}")
    print("TnT model complexity: ", internal_cnt, " internal nodes, ", leaf_cnt, " leaf nodes")


    print('-'*25+'cart'+'-'*25)
    cart = DecisionTreeClassifier(max_leaf_nodes=internal_cnt+1, random_state=0)
    start_time = time.time()
    cart.fit(X=train_inputs, y=train_labels)
    end_time = time.time()
    prediction_train = cart.predict(train_inputs)
    accuracy_train = accuracy_score(train_labels, prediction_train)
    print("CART accuracy (train set):", accuracy_train)
    prediction_test = cart.predict(val_inputs)
    accuracy_test = accuracy_score(val_labels, prediction_test)

    print("CART accuracy (test set):", accuracy_test)
    print(f"train time: {end_time-start_time:.2f}")
    print("CART model complexity: ", internal_cnt, " internal nodes, ", internal_cnt+1, " leaf nodes")
    print('-'*50)
    # tnt.trX = train_inputs
    # tnt.prune()
    if draw:
        dot_data = extract_dotdata_from_tnt(tnt=tnt, 
                                            feature_names=feature_names,  
                                            class_names=class_names,
                                            X=train_inputs)
        # print(dot_data)
        graph = graph_from_dot_data(dot_data)
        graph.write_png(f'tnt_{dataset_name}.png')

        dot_data = export_graphviz(cart, out_file=None, 
                                    feature_names=feature_names,  
                                    class_names=class_names,
                                    filled=True,
                                    impurity=False,
                                    label='none')
        dot_data = dot_data.replace('black', 'white', 1)
        graph = graph_from_dot_data(dot_data)
        graph.write_png(f'cart_{dataset_name}.png')
    

def extract_dotdata_from_tnt(tnt:TnT, feature_names, class_names, X):
    colors = ['#e58139', '#efe6fc', '#51e890', '#843ee6', '#67e539', '#d65bdc'\
            '#ee6688', '#ee6677', '#eedd33', '#eedd22', '#eedd11', '#2d7e95', \
            '#0dd2a3', '#813f5a', '#c0c623', '#7e6ed4', '#04ee51', '#b08f6e', \
            '#421c1a', '#000e4b', '#0b3c7f', '#757779', '#cc1177', '#776fdb', \
            '#f5a4aa', '#515f94', '#6677b9', '#f5a4aa', '#16537e', '#4e767e', \
            '#65744c', '#d4af37', '#99c3a6', '#ff7373', '#662e7b', '#9a46b7', \
            '#9f55a5', '#87006e', '#813f5a', '#c0c623', '#DFFF00', '#FFBF00', \
            '#FF7F50', '#DE3163', '#9FE2BF', '#40E0D0', '#6495ED', '#CCCCFF']
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
            if len(feature_names) < 20:
                color = colors[node.feature_index + len(class_names)]
            else:
                color = colors[1 + len(class_names)]
            # print(node.feature_index)
        else:
            label_info = f'label = {class_names[node.label]}'
            color = colors[node.label]
        tmp = f'{i} [label=\"{label_info}\", fillcolor=\"{color}\"] ;'
        info += f'{tmp}\n'
    for i, node in enumerate(node_all):
        if node.left is not None and node.right is not None: # not leaf
            li, ri = node_map[node.left], node_map[node.right]
            info += f'{i} -> {li} [label = "True"];\n{i} -> {ri} [label = "False"];\n'
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
                        class_names=iris.target_names,
                        dataset_name='iris')

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

def pendigits():
    df = pd.read_csv('data/pendigits_csv.csv')
    X, Y = [], []
    for i in range(len(df)):
        tmp = []
        for index in df.columns:
            if index != 'class':
                tmp.append(int(df[index][i]))
            else:
                Y.append(int(df[index][i]))
        X.append(tmp)
    feature_names, target_names = [], []
    for index in df.columns:
        if index != 'class':
            feature_names.append(index)
    for i in range(10):
        target_names.append(str(i))
    compare_tnt_cart(X=np.array(X), Y=np.array(Y), ccp_alpha=0.0005, 
                        feature_names=np.array(feature_names),
                        class_names=np.array(target_names),
                        dataset_name='pendigits',
                        draw=True)

def letter_recognition():
    df = pd.read_csv('data/letter-recognition.csv')
    X, Y = [], []
    for i in range(len(df)):
        tmp = []
        for index in df.columns:
            if index != 'letter':
                tmp.append(int(df[index][i]))
            else:
                Y.append(ord(df[index][i])-ord('A'))
        X.append(tmp)
    feature_names, target_names = [], []
    for index in df.columns:
        if index != 'letter':
            feature_names.append(index)
    for i in range(65,91):
        target_names.append(chr(i))
    compare_tnt_cart(X=np.array(X), Y=np.array(Y), ccp_alpha=0.001, 
                        feature_names=np.array(feature_names),
                        class_names=np.array(target_names),
                        dataset_name='letter_recognition',
                        draw=True)

letter_recognition()
