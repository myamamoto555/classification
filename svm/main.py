#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cross_validation 

labels = np.array([0, 1, 1, 0])
alldata = np.array([[1.2, 3.4, 8.2], [2.3, 1.1, 2.7], [3.5, 2.1, 2.2], [1.1, 3.5, 7.2]])

k_fold = cross_validation.KFold(4, 2, shuffle = True) # 4事例を2分割交差検定する

score = 0
for train_index, test_index in k_fold:
    label_train = labels[train_index]
    label_test = labels[test_index]
    data_train = alldata[train_index]
    data_test = alldata[test_index]

    # SVM実行
    from sklearn.svm import SVC # SVM用
    model = SVC()               # インスタンス生成
    model.fit(data_train, label_train) # SVM実行

    # 予測実行
    from sklearn import metrics       # 精度検証用
    predicted = model.predict(data_test) # テストデーテへの予測実行
    s = metrics.accuracy_score(label_test, predicted)
    score += s
    print s

print "-----------------"
print float(score)/ 2
