import pandas as pd
import os
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats,integrate


def read_iris():
    iris_data_path = os.path.dirname(os.path.realpath(__file__))+'/data/iris_data.txt'
    iris_data = pd.read_csv(iris_data_path)
    iris_data.columns= ['sepal_length','sepal_width','petal_length','petal_width','species']
    return iris_data

def huitu():
    iris_data = read_iris()
    seaborn.set_style('darkgrid')
    # 绘图
    # seaborn.countplot(x='species',data=iris_data)  #按照分类进行分组计数
    # seaborn.barplot(x='species',y='petal_length',data=iris_data)
    # seaborn.boxplot(x='species',y='petal_length',data=iris_data)
    # seaborn.distplot(iris_data['petal_width'])

    # 分类绘图
    # iris_vir = iris_data[iris_data.species=='Iris-virginica']
    # iris_s = iris_data[iris_data.species=='Iris-setosa']
    # iris_ver = iris_data[iris_data.species=='Iris-versicolor']
    # seaborn.distplot(iris_vir['petal_width'],label='vir').set(ylim=(1,15))
    # seaborn.distplot(iris_s['petal_width'],label='s')
    # seaborn.distplot(iris_ver['petal_width'],label='ver').legend()


    # facetgrid 从数据集不同的侧面进行画图，hue指定分类的字段，使得代码会更加简洁
    # 尝试修改row/col 参数，替代hue,r
    # g = seaborn.FacetGrid(iris_data,col='species',hue='species',aspect=1,palette='colorblind',legend_out=True)
    # g.map(seaborn.distplot,'petal_length')
    # g.add_legend()

    # 化线性回归曲线
    # seaborn.regplot(x='petal_width',y='petal_length',data=iris_data)

    # 分类画线性回归
    # g = seaborn.FacetGrid(iris_data,hue='species')
    # g.set(xlim=(0,2.5))
    # g.map(seaborn.regplot,'petal_width','petal_length')
    # g.add_legend()

    # 不显示拟合曲线
    # g = seaborn.FacetGrid(iris_data,hue='species')
    # g.set(xlim=(0,2.5))
    # g.map(plt.scatter,'petal_width','petal_length')
    # g.add_legend()
    # plt.show()

def sinplot(flip=1):
    # seaborn.set_style('darkgrid')
    # seaborn.axes_style()
    np.random.seed(sum(map(ord, 'aesthetics')))
    x = np.linspace(0,14,100)
    for i in range(1,7):
        plt.plot(x,np.sin(x+i*0.5)*flip*(7-i))
    seaborn.despine()

def set_style():
    # seaborn.set_style('darkgrid')
    # data = np.random.normal(size=(20,6))+np.arange(6)/2
    # seaborn.boxplot(data=data)
    # seaborn.violinplot(data=data)
    # seaborn.despine()
    # seaborn.despine(offset=10,trim=True)
    # with seaborn.axes_style('darkgrid'):  #临时设置图标样式
    #     plt.subplot(211)
    #     sinplot()
    # plt.subplot(212)
    # sinplot(-1)

    # seaborn.set_context('paper')
    # plt.figure(figsize=(8,6))
    # sinplot()

    seaborn.set_context('poster')
    plt.figure(figsize=(8, 6))
    sinplot()

def distribution_data_visiual():
    seaborn.set(color_codes=True)
    np.random.seed(sum(map(ord,'distributions')))

if __name__ == '__main__':
    # set_style()
    iris = seaborn.load_dataset('iris')
    seaborn.pairplot(iris)
    plt.show()

