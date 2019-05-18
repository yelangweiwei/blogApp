#! /usr/bin/env python
# -*-coding:utf-8 -*-
# Author:zhouweiwei
'''
使用svm进行乳腺癌的检测
'''
import os
import pandas as pd
import seaborn as sns
import numpy as py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import StandardScaler

def linearSCV_t():
    #加载数据
    data_path = os.path.dirname(os.path.realpath(__file__))+'/data/breast_cancer_data-master/data.csv'
    data = pd.read_csv(data_path)

    '''探索数据'''
    #展示列表信息
    pd.set_option('display.max_columns', None)
    print(data.columns)
    # print(len(data.columns))
    print(data.head(5))

    '''数据清理，使用所有的特征字段'''
    data.drop('id',axis=1,inplace=True)
    #其中有均值，标准差，最大值
    features = data.columns[2:32]
    #将诊断结果标记为数字
    data['diagnosis'] = data['diagnosis'].map({'B':0,'M':1})

    '''将诊断结果可视化'''
    sns.countplot(data['diagnosis'])
    plt.show()
    #使用热力图查看所有变量之间的相关性
    corr =data[features].corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr,annot=True)
    plt.show()

    '''特征选择 这个使用全部的特征数据'''
    # train,test = train_test_split(data,test_size=0.3)
    # train_data = train[features]
    # train_label = train['diagnosis']
    # test_data  = test[features]
    # test_label = test['diagnosis']
    # '''将数据规范化'''
    # ss = StandardScaler()
    # train_data = ss.fit_transform(train_data)
    # test_data = ss.fit_transform(test_data)
    # #构建模型
    # model = LinearSVC()
    # model.fit(train_data,train_label)
    # predict_data = model.predict(test_data)
    # print('model accuracy:',accuracy_score(test_label,predict_data))

    #使用k折交叉检验
    train_data = data[features]
    ss = StandardScaler()
    train_data = ss.fit_transform(train_data)
    labels = data['diagnosis']

    model = LinearSVC()
    svc = model.fit(train_data,labels)
    print('交叉检验结果:',py.mean(cross_val_score(svc,train_data,labels,cv=10)))

#使用SCV，顺便在添加PCA主程序分析法来进行分析
def SCV_t():
    #加载数据
    data_path = os.path.dirname(os.path.realpath(__file__))+'/data/breast_cancer_data-master/data.csv'
    data = pd.read_csv(data_path)

    '''探索数据'''
    #展示列表信息
    pd.set_option('display.max_columns', None)
    print(data.columns)
    # print(len(data.columns))
    print(data.head(5))

    '''数据清理，使用所有的特征字段'''
    data.drop('id',axis=1,inplace=True)
    #其中有均值，标准差，最大值
    features = data.columns[2:32]
    #将诊断结果标记为数字
    data['diagnosis'] = data['diagnosis'].map({'B':0,'M':1})

    '''将诊断结果可视化'''
    sns.countplot(data['diagnosis'])
    plt.show()
    #使用热力图查看所有变量之间的相关性
    corr =data[features].corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr,annot=True)
    plt.show()

    '''1特征选择 这个使用全部的特征数据'''
    # train,test = train_test_split(data,test_size=0.3)
    # train_data = train[features]
    # train_label = train['diagnosis']
    # test_data  = test[features]
    # test_label = test['diagnosis']
    # '''将数据规范化'''
    # ss = StandardScaler()
    # train_data = ss.fit_transform(train_data)
    # test_data = ss.fit_transform(test_data)
    # #构建模型
    # model = LinearSVC()
    # model.fit(train_data,train_label)
    # predict_data = model.predict(test_data)
    # print('model accuracy:',accuracy_score(test_label,predict_data))

    #2使用k折交叉检验
    # train_data = data[features]
    # ss = StandardScaler()
    # train_data = ss.fit_transform(train_data)
    # labels = data['diagnosis']
    #
    # model = LinearSVC()
    # svc = model.fit(train_data,labels)
    # print('交叉检验结果:',py.mean(cross_val_score(svc,train_data,labels,cv=10)))










if __name__=='__main__':
    linearSCV_t()



