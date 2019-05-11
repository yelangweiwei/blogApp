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
#探索性数据分析
def exploratory_data_analysis():
    #导入数据
    data_path = 'F:\\blog\\blogApp\\data_analysis\\data\\housing\\housing.csv'
    housing_df = pd.read_csv(data_path)
    head_data = housing_df.head(5)
    # print(head_data)

    #查看数据的属性
    print(housing_df.columns)

    #绘制房屋价格分布图
    # sns.distplot(housing_df['SalePrice'])
    #绘制房屋面积分布图
    # sns.distplot(housing_df['GrLivArea'])
    #地下室面积分布图
    # sns.distplot(housing_df['TotalBsmtSF'])
    #探索连续性变量之间的相关性
    # sns.regplot(x='GrLivArea',y='SalePrice',data=housing_df)
    # sns.regplot(x='TotalBsmtSF',y='SalePrice',data=housing_df)

    #探索离散型变量和连续型变量之间的关系
    #总面积和售价之间的 关系x
    # sns.regplot(x='OverallQual',y='SalePrice',data=housing_df)
    # sns.boxplot(x='OverallQual',y='SalePrice',data=housing_df)
    # sns.boxplot(x='CentralAir', y='SalePrice', data=housing_df)
    # sns.boxplot(x='Neighborhood', y='SalePrice', data=housing_df)


    #计算这些属性之间的 相关系数矩阵，并使用heatmap方法绘制出相关系数的矩阵热力图，可以快速的并且直观的探索出与saleprice
    # 之间的关系
    info = ['SalePrice','OverallQual','GarageArea','GrLivArea','TotalBsmtSF','YearBuilt']
    #设置字体
    sns.set(font_scale=0.7)
    #使用.corr计算属性之间的相关系数矩阵，在使用heatmap计算热力图 annot = true:的作用是把相关的系数在图中注释出来
    sns.heatmap(housing_df[info].corr(),annot=True,vmin=0,vmax=1)
    #从图中可以看出售价和各个因素之间的关系。
    plt.show()

'''
预测性数据分析
    注意事项：
    1）使用探索性数据分析，探索出跟预测目标相关的 因素
    2）对相关的数据进行处理使其符合建模的要求
    3）特征挑选应该同时注重模型的准确率和可解释性。
    
'''
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
def predict_t():
    data_path = 'F:\\blog\\blogApp\\data_analysis\\data\\housing\\housing.csv'
    housing_df = pd.read_csv(data_path)
    features = ['OverallQual','GarageArea','GrLivArea','TotalBsmtSF','YearBuilt']
    target = 'SalePrice'
    lr = LinearRegression()
    rf = RandomForestRegressor(100) #设置子树的个数
    models= [lr,rf]
    for model in models:
        scores = cross_val_score(model,housing_df[features],housing_df[target],cv=5,scoring='neg_mean_absolute_error')
        print(type(model).__name__,np.mean(scores))

def predict_new_t():
    data_path = 'F:\\blog\\blogApp\\data_analysis\\data\\housing\\housing.csv'
    housing_df = pd.read_csv(data_path)

    #创建新的特征
    housing_df['HasBsmt'] = 0
    housing_df.at[housing_df['TotalBsmtSF']>0,'HasBsmt'] =1
    housing_df['LogArea'] = np.log(housing_df['GrLivArea'])
    new_features = ['OverallQual','GarageArea','LogArea','HasBsmt','YearBuilt']
    target = 'SalePrice'
    lr = LinearRegression()
    rf = RandomForestRegressor(100) #设置子树的个数
    models= [lr,rf]
    for model in models:
        scores = cross_val_score(model,housing_df[new_features],housing_df[target],cv=5,scoring='neg_mean_absolute_error')
        print(type(model).__name__,np.mean(scores))



if __name__=='__main__':
    #探索性数据分析
    # exploratory_data_analysis()
    predict_t()
    predict_new_t()