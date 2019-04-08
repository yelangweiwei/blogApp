'''
特征工程介绍：
    数据和特征决定了机器学习的上限，模型和算法只是逼近这个上限而已；特征工程：目的是从原始数据中提取特征以供算法和模型使用。
特征工程包括：特征使用方案，特征获取方案，特征处理，特征监控

特征使用方案：
    1）要实现我们目标需要哪些数据
    2）可用性评估（获取难度，覆盖率，准确率）
特征获取方案：
    1）如何获取这些特征，
    2）如何存储
特征处理：
    特征处理是特征工程的核心，sklearn提供了较为完整的特征处理方案，包括数据预处理，特征选择，降维等，
    1）数据预处理：
        (1)不属于同一个量纲，数据的规格不一样，不能够放在一起比较；
        (2)信息冗余，对于某些定量特征，其包含的有效信息为区间划分，例如只关心“及格”和“不及格”，那么需要将定量的考分，转化为0和1表示不及格和及格，二值化解决这一问题
        (3)定性特征不能直接使用，某些机器学习算法和模型只能接收定性特征的输入，那么需要将定性特征转换为定量特征，最简单的方式，是为每一个特性值指定一个定量值，但是这种方式过于灵活，增加了调参的工作；通常使用哑编码的方式将定性特征转换为定量特征：假设有N中定性值，则将这一个特征扩展为N种特征，当原始特征值为第i种定性值时，第i个扩展为特征赋值为1，其他特征赋值为0，哑编码的方式相比直接指定的方式，不用增加调参的工作，对于线性模型来说，使用哑编码后的特征可达到非线性的效果
        (4) 存在缺失值：缺失值需要补充
        (5) 信息利用率低：不同的机器学习算法和模型对数据中信息的利用是不同的，之前提到的对定性特征哑编码可以达到非线性的效果，类似的，对定量变量多项式话化，或者进行其他的转变
            ，都能达到非线性的效果
    
'''

'''
无量纲：
    1）无量纲化使不同规格的数据转换为同一规格，常见的无量纲化方法有标准化和区间缩放法。标准化的前提是特征值服从正态分布，标准化后，其转换成标准的正态分布。区间缩放法利用了边界值信息，将特征的取值区间缩放到某个特点的范围；
    标准化：需要计算的特征的均值和标准差：公式为：x‘ = (x-x_mean)/S

'''
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
def standscaler():
    from sklearn.preprocessing import StandardScaler
    #标准化，返回值为标准化后的数据
    # 读取数据
    iris = load_iris()
    iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    result = StandardScaler().fit_transform(X=iris.data)
    print(result)
    print(result[:,0])
    print(sum(result[:,0]))
    seaborn.distplot(result[:,1])
    plt.show()


'''
区间缩放法
区间缩放发的形式有多样，比较常见的是：使用最值进行缩放；公式是 x = (x-min)/(max-min)
'''
from sklearn.preprocessing import MinMaxScaler
def minMaxScaler():
    iris = load_iris()
    scaler = MinMaxScaler()
    result = scaler.fit_transform(iris.data)
    print(result[:,0])
    seaborn.distplot(result[:,1])
    plt.show()

'''
标准化和归一化的区别：
1）标准化是依照矩阵的列处理数据，其通过z-score的方法，将样本的特征值转换到同一量纲下
2）归一化是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似时，拥有统一的标准，也就是说都转换为"单位向量"。公式是：x_result = x/sqrt(sum(x[j]^2))
'''
from sklearn.preprocessing import Normalizer
def normalize():
    iris = load_iris()
    #归一化，返回值为归一化后的数据
    normallizeH = Normalizer()
    result = normallizeH.fit_transform(iris.data)
    print(result[:,1])
    seaborn.distplot(result[:,0])
    plt.show()

'''
对定量特征二值化：
    1）定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
        公式：if x>threshold x =1;if x<=threshold x = 0;
'''
from sklearn.preprocessing import Binarizer
def binarizer():
    iris = load_iris()
    #二值化，阈值设置为3，返回值为二值化后的数据
    bin = Binarizer(threshold=1)
    result = bin.fit_transform(iris.data)
    print(result)

'''
对定性特征哑编码
    1）由于iris数据集的特征皆为定量特征，故使用其目标值进行哑编码、
    2）定性特征：
'''



if __name__=='__main__':
    # minMaxScaler()
    # normalize()
    binarizer()

