#! /usr/bin/env python
# -*-coding:utf-8 -*-
# Author:zhouweiwei

'''使用knn进行图像识别'''
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
def knn_t():

    '''加载数据'''
    digits = load_digits()
    data = digits.data
    #数据探索
    print(data.shape)
    #查看第一幅图
    print(digits.images[0])
    #查看第一幅图的分类
    print(digits.target[0])
    #数据可视化
    plt.gray()
    plt.imshow(digits.images[0])
    plt.show()

    #数据特征提取，数据展示的图像，没有特征
    #数据提取
    train_data,test_data,train_target,test_target = train_test_split(data,digits.target,test_size=0.25,random_state=33)

    #将数据标准化
    ss = StandardScaler()
    train_ss_data = ss.fit_transform(train_data)
    #ss经过了拟合，这里直接进行转换就可以了
    test_ss_data = ss.transform(test_data)

    #构建模型
    knn_model = KNeighborsClassifier(n_neighbors=200,weights='uniform',algorithm='auto',leaf_size=30)
    knn_model.fit(train_ss_data,train_target)
    prediction_result = knn_model.predict(test_ss_data)
    print('knn的准确度：',accuracy_score(test_target,prediction_result))


    '''构建svm，decisionTree,bayes'''
    #在进行构造bayes的时候，使用的是多项式贝叶斯，参数不能有负值
    max_min_scan = MinMaxScaler()
    train_max_min_data = max_min_scan.fit_transform(train_data)
    test_max_min_data = max_min_scan.transform(test_data)
    bay_model = MultinomialNB()
    bay_model.fit(train_max_min_data,train_target)
    predict_bay_result = bay_model.predict(test_max_min_data)
    print('bay accuracy:',accuracy_score(test_target,predict_bay_result))

    svm_model = SVC()
    svm_model.fit(train_ss_data,train_target)
    predict_svc_result = svm_model.predict(test_ss_data)
    print('svc accuracy:',accuracy_score(test_target,predict_svc_result))

    tree_model = DecisionTreeClassifier()
    tree_model.fit(train_ss_data,train_target)
    predict_tree_result = tree_model.predict(test_ss_data)
    print('tree accuracy:',accuracy_score(test_target,predict_tree_result))













if __name__=='__main__':
    knn_t()