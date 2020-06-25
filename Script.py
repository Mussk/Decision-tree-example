import random
import numpy as np
from sklearn.datasets import load_iris
import CART as cart
from Node import Node as node_class
from ete3 import Tree



tree_arr = list()


def splitTrainTest (data, target, testPercent):
    trainData_x = []
    trainData_y = []
    testData_x  = []
    testData_y = []
    for d, t in zip(data, target):
        if random.random() < testPercent:
            testData_x.append(d)
            testData_y.append(t)
        else:
            trainData_x.append(d)
            trainData_y.append(t)
    return trainData_x, trainData_y, testData_x, testData_y




def create_tree(node, feature_names, target_names, granulation, min_gini):

        node_left = node_class(None,None,None,None)
        node_right = node_class(None,None,None,None)

        node.child_nodes = np.array([node_left,node_right])

        node_left.data, node_left.target, node_right.data,\
        node_right.target, score, feature, threshold = cart.split_node_node(node.data,
                                                         node.target,
                                                         granulation,
                                                         cart.get_Gini)

        node.score = score
        node.feature = feature
        node.threshold = threshold

        tree_arr.append(node_left)
        tree_arr.append(node_right)

        if cart.get_Gini(node_left.target) > min_gini:
            create_tree(node_left, feature_names, target_names, granulation, min_gini)
        else:
            node.child_nodes[0].is_leaf = True
        if cart.get_Gini(node_right.target) > min_gini:
            create_tree(node_right, feature_names, target_names, granulation, min_gini)
        else:
            node.child_nodes[1].is_leaf = True


def test_nodes(node, min_gini):

    for d, t in zip(node.data,node.target):
        if d[node.feature] < node.threshold:
            node.child_nodes[0].data.append(d)
            node.child_nodes[0].target.append(t)
        else:
            node.child_nodes[1].data.append(d)
            node.child_nodes[1].target.append(t)

    for i in range(2):
        if (cart.get_Gini(node.child_nodes[i].target) > min_gini) \
                & (node.child_nodes[i].is_leaf is False):
            test_nodes(node.child_nodes[i],min_gini)


def run_test(tree,root, testData_x, testData_y, min_gini):

    root.data = np.asarray(testData_x, dtype='float64')
    root.target = np.asarray(testData_y, dtype='int')

    for node in tree:
        node.data = list()
        node.target = list()

    test_nodes(root,min_gini)


def create_visual_node(t ,node, target_names):

    if len(node.target) > 0:

        t = t.add_child(name=str(str(target_names[cart.get_class_for_node(node.target)]) + ', elements: '
                                 + str(len(node.data))) + ', gini: ' + str(cart.get_Gini(node.target)))

        if node.child_nodes is not None:
            for i in range(2):
                create_visual_node(t, node.child_nodes[i], target_names)


def show_tree(root,target_names, testData_x):

    print('Size of test dataset: ' + str(len(testData_x)))

    t = Tree()
    t.name = target_names[cart.get_class_for_node(root.target)]
    create_visual_node(t,root,target_names)

    print(t)

'''
IRIS DATASET
features:
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica
'''
iris = load_iris()
x = iris.data
y = iris.target
x_names = iris.feature_names
y_names = iris.target_names

granulation = 10
min_gini = 0.1
testPercent = 0.2

trainData_x, trainData_y, testData_x, testData_y = splitTrainTest(x,y,testPercent)

node = node_class(np.asarray(trainData_x,dtype='float64'), np.asarray(trainData_y,dtype='int'), x_names, y_names)

create_tree(node, iris.feature_names, iris.target_names, granulation, min_gini)

run_test(tree_arr,node,testData_x,testData_y,min_gini)


show_tree(node,y_names,testData_x)