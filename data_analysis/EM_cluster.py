'''
EM聚类，也叫最大期望算法
三个主要的步骤，初始化参数，观察预期，重新估计；前两个是期望步骤，最后一个是最大化步骤。
'''
import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns
#高斯混合模型
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

def get_data(data_path):
    data_content = pd.read_csv(data_path,encoding='GBK')
    #将获得数据
    print(type(data_content))
    print(data_content.head(5))
    #打印dataframe的列
    # print(data_content.columns)
    #通过查看前5条数据，获得数据的features
    features = list(data_content.columns)
    # print(features)

    #



def hero_em_t():
    #读取数据
    data_path = os.path.dirname(os.path.realpath(__file__))+'/data/EM/EM_data/heros.csv'
    get_data(data_path)




if __name__=='__main__':
    hero_em_t()