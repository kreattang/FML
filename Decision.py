#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/18 18:45
# @Author  : blvin.Don
# @File    : Decision.py

from sklearn import tree
import csv
X = []
Y = []
csv_reader = csv.reader(open('temp2.csv', encoding='utf-8'))
for row in csv_reader:
    temp = []
    for i in range(24):
        temp.append(float(row[i]))
    X.append(temp)
    Y.append(row[-1])
# print(X[0])
from collections import Counter
values_counts = Counter(Y)
print("正负样本类别统计：",values_counts)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
y_predict = clf.predict(X_test)
y_train_predict = clf.predict(X_train)
from collections import Counter
values_counts = Counter(y_test)
print("正负样本类别统计：",values_counts)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_predict)
print(X_test[:3])
print(y_predict[:3])
print(confusion_matrix)
print(accuracy_score(y_train,y_train_predict))
print(accuracy_score(y_predict,y_test))

with open("jueceshu.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f)

