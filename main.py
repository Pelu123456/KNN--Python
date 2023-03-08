from csv import reader
from math import sqrt

import csv
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def load_csv(filename, a):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=a)
        for row in csv_reader:
            if not row:
                continue
            dataset.append([float(i) for i in row])
    return dataset

def check(filename):
    file = open(filename)
    search_word = ","
    if(search_word in file.read()):
        return 1
    else:
        return 2

def euclid_distance(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

def getMostFrequentElement(list):
    temp = list(set(list))
    res1, res2 = -1, 0
    for i in temp:
        if list.count(i) > res1:
            res1 = list.count(i)
            res2 = i
    return res2

def Predict(Cordinate_train, Class_train, Cordinate_input, k):
    result_labels = []
    for item in Cordinate_input:
        point_dist = []
        for j in range(len(Cordinate_train)):
            distances = euclid_distance(Cordinate_train[j,:], item )
            point_dist.append(distances)
        dist = sorted(point_dist)[:k]
        labels = Class_train[dist]
        labels = list(labels)
        lab = getMostFrequentElement(labels)
        result_labels.append(lab)
    return result_labels

def kNN(trainfile, testfile, k):
    a = check(trainfile)
    if a == 1:
        train = load_csv(trainfile, ',')
        test = load_csv(testfile, ',')
    else:
        train = load_csv(trainfile,',')
        test = load_csv(testfile, ',')
    Cordinate_train = np.array([x[:-1] for x in train])
    Class_train = np.array([int(x[-1]) for x in train])
    Cordinate_test = np.array([x[:-1] for x in test])
    Class_test = np.array([int(x[-1]) for x in test])

    y_pred = Predict(Cordinate_train, Class_train, Cordinate_test, k)
    accuracyScore = accuracy_score(Class_test, y_pred)
    print(">>>>>>>>>>>>>>>>>>>KNN Classifer <<<<<<<<<<<<<<<<<<<<<<<<")
    print("The result of KNN classifer of" , trainfile , "with", k ,"nearest neibours")
    print("Accuracy points: ", (accuracyScore * 100), "%")
    print("Confusion Matrix: \n", metrics.confusion_matrix(Class_test,y_pred))
    print("Classification Report: \n", metrics.classification_report((Class_test,y_pred)))


