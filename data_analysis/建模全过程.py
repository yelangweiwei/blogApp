#! /usr/bin/env python
# -*-coding:utf-8 -*-
# Author:zhouweiwei

'''
先进性探索性数据分析，发现各个特征之间的关系，然后使用预测性数据分析
预测发展趋势
'''
# 探索性数据分析：
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def exploratory_data_analysis():
    #导入数据
    data_path = 'F:\\blog\\blogApp\\data_analysis\\data\\housing\\housing.csv'
    housing_df = pd.read_csv(data_path)
    head_data = housing_df.head(5)
    # print(head_data)

    #查看数据的属性
    print(housing_df.columns)

    #绘制房屋价格分布图
    sns.distplot(housing_df['SalePrice'])
    plt.show()



if __name__=='__main__':
    #探索性数据分析
    exploratory_data_analysis()