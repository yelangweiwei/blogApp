#! /usr/bin/env python
# -*-coding:utf-8 -*-
# Author:zhouweiwei
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import  AdaBoostClassifier,AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.metrics import zero_one_loss

def adaBootRegressor_t():
    #加载数据
    data = load_boston()
    #分割数据
    train_x,test_x,train_y,test_y = train_test_split(data.data,data.target,test_size=0.25,random_state=33)
    #使用adaboost回归模型
    regression = AdaBoostRegressor()
    regression.fit(train_x,train_y)
    pred_y  = regression.predict(test_x)
    mse = mean_squared_error(test_y,pred_y)
    print('房价预测结果：',pred_y)
    print('均方差=',round(mse,2))

#每次运行的结果不一样，这个和每次获得测试数据和训练数据的部分不一样造成的
def deciAndKnn_t():
    data = load_boston()
    train_x,test_x,train_y,test_y = train_test_split(data.data,data.target,test_size=0.25,random_state=33)
    dec_regression = DecisionTreeRegressor()
    dec_regression.fit(train_x,train_y)
    pred = dec_regression.predict(test_x)
    mse  = mean_squared_error(test_y,pred)
    print('dtr均方差是:',round(mse,2))

    knn_regression  = KNeighborsRegressor()
    knn_regression.fit(train_x,train_y)
    pred_y = knn_regression.predict(test_x)
    mse = mean_squared_error(test_y,pred_y)
    print('knn均方差是:',round(mse,2))


def compare_adabost_and_dec():
    #设置adaboost的迭代次数
    n_estimators = 200
    #产生二分类的数据
    x,y = datasets.make_hastie_10_2(n_samples=12000,random_state=1)
    #从 12000个数据中提取前2000用来测试
    test_x,test_y = x[:2000],y[:2000]
    train_x,train_y = x[2000:],y[2000:]
    #弱分类器
    dt_sample = DecisionTreeClassifier(max_depth=1,min_samples_leaf=1)
    dt_sample.fit(train_x,train_y)
    dt_sample_error = 1-dt_sample.score(test_x,test_y)
    print('dt_sample_error:',dt_sample_error)
    #决策树分类器
    dt = DecisionTreeClassifier()
    dt.fit(train_x,train_y)
    dt_error = 1-dt.score(test_x,test_y)
    print('de_error:',dt_error)
    # adaboost
    ada = AdaBoostClassifier(base_estimator=dt_sample,n_estimators=n_estimators)
    ada.fit(train_x,train_y)
    #计算每种的错误率
    ada_error = 1-ada.score(test_x,test_y)
    print('ada_error:',ada_error)

    #将三个分类器进行可视化
    fig = plt.figure()
    #设置显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax = fig.add_subplot(111)
    ax.plot([1,n_estimators],[dt_sample_error]*2,'k-',label=u'决策树弱分类器 错误率')
    ax.plot([1,n_estimators],[dt_error]*2,'k--',label=u'决策树模型 错误率')
    ada_error = np.zeros((n_estimators,))
    #遍历每次迭代的结果 i为迭代的次数，pred_y 为预测的结果
    for i,pred_y in enumerate(ada.staged_predict(test_x)):
        #统计错误率
        ada_error[i] = zero_one_loss(pred_y,test_y)
    #绘制每次迭代的adaboost的错误率
    ax.plot(np.arange(n_estimators)+1,ada_error,label='adaboost test 错误率',color='orange')
    ax.set_xlabel('迭代的次数')
    ax.set_ylabel('错误率')
    leg = ax.legend(loc='upper right',fancybox=True)
    plt.show()











if __name__=='__main__':
    # adaBootRegressor_t()
    # deciAndKnn_t()
    compare_adabost_and_dec()

