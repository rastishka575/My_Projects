from sklearn import datasets
import numpy as np
import math
import random
import time
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import pickle


def predict(data, forest2):
    pred = np.empty((len(forest2), len(data)))
    for i in range(len(forest2)):
        pred[i] = forest2[i].getPrediction(data)

    predic = np.empty(len(data))
    for i in range(len(data)):
        dic = {x: 0 for x in range(10)}
        for dig in pred[:, i]:
            dic[dig] += 1
        k = 0
        max = 0
        for i in range(10):
            if (dic[i] > max):
                k = i
                max = dic[i]
        predic[i] = k
    return predic


def accur(data, label, forest2):
    acc = 0
    label_pred = predict(data, forest2)
    for i in range(len(data)):
        if label_pred[i] == label[i]:
            acc += 1
    acc /= len(data)
    return acc


def confusion_matrix(data, label, forest2):
    cm = np.zeros((10, 10))
    pred_l = predict(data, forest2)
    for i in range(len(label)):
            cm[int(label[i]),int(pred_l[i])] += 1
    dic = {x: 0 for x in range(10)}
    for dig in label:
        dic[dig] += 1
    for i in range(10):
        cm[i] /= dic[i]
    return cm


def pickle_it(data, path):
    """
    Сохранить данные data в файл path
    :param data: данные, класс, массив объектов
    :param path: путь до итогового файла
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle_it(path):
    """
    Достать данные из pickle файла
    :param path: путь до файла с данными
    :return:
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

class TreeNode:
    divide_value = None
    split_characteristic = None

    left_child = None
    right_child = None

    classes_count = None

    def __init__(self, divide_value, split_characteristic, left_child, right_child, classes_count=None):
        self.left_child = left_child
        self.right_child = right_child
        self.divide_value = divide_value
        self.split_characteristic = split_characteristic


class DeсisionTree:
    MAX_HEIGHT_TREE = None
    MIN_ENTROPY = None
    STEPS_COUNT = None
    FEATURES_SIZE = None
    FEATURES_TO_EVALUATE = None
    MIN_NODE_ELEMENTS = None
    CLASSES_COUNT=None

    data = None
    labels = None
    root = None

    def __init__(self, max_height_tree, min_entropy,steps_count, min_node_elements):
        self.MAX_HEIGHT_TREE = max_height_tree
        self.MIN_ENTROPY = min_entropy
        self.STEPS_COUNT = steps_count
        self.MIN_NODE_ELEMENTS = min_node_elements

    def train(self, data, labels):
        self.data = data
        self.labels = labels
        self.VECTOR_SIZE = data.shape[0]
        self.CLASSES_COUNT = 10

        self.root = self.slowBuildTree(self.data, self.labels, 0)


    def getEntropy(self, labels):
        dic = self.get_classes(labels)
        h = 0
        for i in range(10):
            ratio = dic[i]/len(labels)
            if ratio == 0:
                continue
            h -=ratio*np.log(ratio)
        return h

    def get_classes(self, labels):
        #гистограмма классов содержащихся в labels
        dic = {x: 0 for x in range(10)}
        for dig in labels:
            dic[dig] += 1
        return dic

    def getRandomSplitCharacteristics(self):
        # возвращает индексы features, по которым будет происходить поиск лучшего split
        return np.random.random_integers(0,63,1)


    def slowBuildTree(self, data, labels, height_tree):

        #print("\tHeight tree %s, length data %s" % (height_tree, len(data)))
        allDataEntropy = self.getEntropy(labels)
        """
        Условия на создание терминального узла
        Вывод информации о каждом случае в консоль
        """
        if height_tree > self.MAX_HEIGHT_TREE or allDataEntropy < self.MIN_ENTROPY or data.shape[0] <= self.MIN_NODE_ELEMENTS:
            dic=self.get_classes(labels)
            k=0
            max=0
            for i in range(10):
                if(dic[i]>max):
                    k=i
                    max=dic[i]
            t=TreeNode(None, None, None, None)
            t.classes_count = k
            return t
        """
        Подсчёт лучшего разделения и поиск лучшего information gain
        """
        best_feature = 0
        best_features = 0
        best_information_gain = 0
        best_data_right = None
        best_data_left = None
        best_label_right = None
        best_label_left = None
        for tay in range(500):
            feature = int(self.getRandomSplitCharacteristics())
            min=data[:,feature].min()
            max=data[:,feature].max()
            t=np.random.random_integers(min, max, 1)
            trues=data[:,feature] <= t
            data_right =data[trues]
            data_left = data[~trues]
            label_right = labels[trues]
            label_left = labels[~trues]
            information_gain=0
            if len(label_right)==0:
                continue
                #information_gain = allDataEntropy - (self.getEntropy(label_left) * len(label_left)) / len(labels)
            if len(label_left)==0:
                continue
                #information_gain = allDataEntropy - (self.getEntropy(label_right) * len(label_right)) / len(labels)
            if  len(label_left)!=0 and len(label_right)!=0:
                information_gain = allDataEntropy-(self.getEntropy(label_right)*len(label_right)+self.getEntropy(label_left)*len(label_left))/len(labels)
            if best_information_gain < information_gain:
                best_information_gain = information_gain
                best_feature = feature
                best_features = t
                best_data_left=data_left
                best_data_right=data_right
                best_label_left=label_left
                best_label_right=label_right
        #print(best_information_gain)
        """
        Деление выборки и перенаправление в новые узлы, рекурсивный вызов этой функции
        """
        if best_information_gain==0:
            print(data.shape[0])
            print(allDataEntropy)
            np.set_printoptions(threshold=np.nan)
            dic = self.get_classes(labels)
            k = 0
            max = 0
            for i in range(10):
                if (dic[i] > max):
                    k = i
                    max = dic[i]
            t = TreeNode(None, None, None, None)
            t.classes_count = k
            return t

        slowBuildTree_right=self.slowBuildTree(best_data_right,best_label_right,height_tree+1)
        slowBuildTree_left=self.slowBuildTree(best_data_left,best_label_left,height_tree+1)
        return TreeNode(best_features,best_feature,slowBuildTree_left,slowBuildTree_right)


    def getPrediction(self, data):
        #Возвращает предсказание для новых данных на основе корня дерева
        pred = np.empty(len(data))
        for i in range(len(data)):
            tree = self.root
            while(tree.classes_count == None):
                if data[i][tree.split_characteristic]<=tree.divide_value:
                    tree = tree.right_child
                else:
                    tree = tree.left_child
            pred[i]=tree.classes_count

        return pred


if __name__ == "__main__":

    """
    Загрузка датасета digits
    """
    data = datasets.load_digits()

    """
    Формирование выборки
    """
    low_bord = 0.8
    high_bord = 0.9

    trn = data.data
    label = data.target

    ind = len(trn)
    dts = np.arange(ind)
    indx = np.random.permutation(dts)
    train_ind = indx[:np.int32(0.8 * ind)]
    valid_ind = indx[np.int32(0.8*ind):np.int32(0.9*ind)]
    test_ind = indx[np.int32(0.9*ind):]
    trn_x=trn[train_ind]
    trn_l=label[train_ind]
    vld_x=trn[valid_ind]
    vld_l=label[valid_ind]
    tst_x=trn[test_ind]
    tst_l=label[test_ind]
    '''
    # Набор обучающих данных
    trn_x = data.data[:int(low_bord * data.data.shape[0])]
    trn_l = label.data[:int(low_bord * data.data.shape[0])]
    # Набор валидационных данных
    vld_x = data.data[int(low_bord * data.data.shape[0]):int(high_bord * data.data.shape[0])]
    vld_l = label.data[int(low_bord * data.data.shape[0]):int(high_bord * data.data.shape[0])]
    # Набор тестовых данных
    tst_x = data.data[int(high_bord * data.data.shape[0]):]
    tst_l = label.data[int(high_bord * data.data.shape[0]):]
    '''
    """
    Валидация по количеству случайных семплирований 5, 50, 250, 500, 1000, +500 в зависимости от мощности компьютера
    """

    best_forest =[]
    best_acc = 0
    # случайный лес
    for i in range(5):
        forest = []

        for j in range(10):
            tree = DeсisionTree(1000, 0.1, 5, 5)
            tree.train(trn_x, trn_l)
            forest.append(tree)
            #pickle_it(forest, "tree_small_data_set1.pickle")
            #forest=unpickle_it("tree_small_data_set.pickle")
        acc_v = accur(vld_x,vld_l,forest)
        print(acc_v)
        if acc_v > best_acc:
            best_forest=forest
            best_acc=acc_v
            
    pickle_it(best_forest, "tree_small_data_set1.pickle") #сохранение модели 
    #best_forest=unpickle_it("tree_small_data_set1.pickle") #загрузка готовой модели
    print(accur(tst_x,tst_l,best_forest))
    print(confusion_matrix(tst_x,tst_l,best_forest))
    #pickle_it(best_forest, "tree_small_data_set.pickle")