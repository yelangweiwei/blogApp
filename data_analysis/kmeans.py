#! /usr/bin/env python
# -*-coding:utf-8 -*-
# Author:zhouweiwei

'''
kmeans：聚类算法：确定k类的中心，找到了.0这些k类的中心，也就完成了聚类
'''
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
def k_means_t():
    data_path = os.path.dirname(os.path.realpath(__file__))+'/data/kmeans-master/data.csv'
    data = pd.read_csv(data_path,encoding='gbk')

    #进行数据探索
    print(data.columns)
    print(data.head(5))
    print(data.dtypes)
    #经过数据探索，发现其中的值变化比较大，进行数据标准化
    features = ['2019年国际排名','2018世界杯','2015亚洲杯']
    train_data = data[features]
    ss = MinMaxScaler()
    train_data = ss.fit_transform(train_data)
    #训练模型
    k_model  = KMeans(n_clusters=3)
    k_model.fit(train_data)
    predict_y = k_model.predict(train_data)
    #将聚类的结果合并到原始数据中
    result  = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
    result.rename({0:u'聚类'},axis=1,inplace=True)
    print(result)

if __name__=='__main__':
    k_means_t()




