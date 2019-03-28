import numpy as py
import pandas as pd
from scipy.stats.stats import ttest_1samp,ttest_ind
from statsmodels.stats.weightstats import  ztest

def check_z():
    #假设检验：
    #构造平均值是175，标准差是5，样本量是100，
    x = py.random.normal(175,5,100).round(1)
    # print(x)
    #使用Z检验pval
    z,pavel =ztest(x,value=175)
    #直接返回pavel,用pvael判断零假设是都成立
    print(pavel)

def check_t():
    x = py.random.normal(175,5,100).round(1)
    t,pval =ttest_1samp(x,popmean=175)
    print(pval)


def check_tt():
    x1 = py.random.normal(175,5,100).round(1)
    x2 = py.random.normal(175,5,100).round(1)
    t,pval =ttest_ind(x1,x2)
    print(pval)

def check_datasets_by_iris():
    iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header =None)
    iris.columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
    sample_iris_data = iris.sample(10)
    print(sample_iris_data)

    # data_mean = py.mean(iris['SepalLengthCm'])
    data_mean = py.mean(iris['SepalLengthCm'])
    print('data_mean:',data_mean)

    # # z检验
    # z, pval = ztest(sample_iris_data['SepalLengthCm'],value=7)
    # print(pval)
    #
    # #t
    # t,pval = ttest_1samp(sample_iris_data['SepalLengthCm'],popmean=7)
    # print(pval)

    #tt检验
    iris1 = iris[iris.Species =='Iris-setosa']
    iris2 = iris[iris.Species=='Iris-virginica']
    t,pval = ttest_ind(iris1['SepalLengthCm'],iris2['SepalLengthCm'])
    iris1_mean = py.mean(iris1['SepalLengthCm'])
    print(iris1_mean)
    iris2_mean = py.mean(iris2['SepalLengthCm'])
    print(iris2_mean)
    print('pval',pval)









if __name__ == '__main__':
    check_datasets_by_iris()