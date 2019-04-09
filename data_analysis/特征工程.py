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
'''
from sklearn.preprocessing import OneHotEncoder
def oneHotEncoder():
    iris = load_iris()
    oneHot = OneHotEncoder()
    result = oneHot.fit_transform(iris.target.reshape(-1,1))
    print(result)

'''
缺失值计算：
    1)在iris数据集中添加缺失值，使用preprocessing库的imputer 类对数据进行缺失值的计算
    2)缺失值计算：返回值为计算缺失值后的数据
    3）参数missing_value为缺失值的表示形式，默认为NaN
    4)参数strategy为缺失值的填充方式，默认为mean
'''
from numpy import vstack,array,nan
from sklearn.preprocessing import Imputer

def imputer():
     iris = load_iris()
     imputerH = Imputer()
     result = imputerH.fit_transform(vstack((array([nan,nan,nan,nan]),iris.data)))
     #把补充的值打印出来
     print(result[0,:])  #[5.84333333 3.054      3.75866667 1.19866667]  这是均值的结果

    #使用中值
     imputerH = Imputer(strategy='median')
     result = imputerH.fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))
     print(result[0, :]) #[5.8  3.   4.35 1.3 ]

     #使用频率
     imputerH = Imputer(strategy='most_frequent')
     result = imputerH.fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))
     print(result[0, :])  #[5.  3.  1.5 0.2]

'''
数据变换：
    1）常见的数据变换基于多项式，基于指数函数的，基于对数函数的。
'''
from sklearn.preprocessing import PolynomialFeatures
def polynomialFeature():
    iris = load_iris()

    #多现实转换，参数degree为度，默认值是2
    poly = PolynomialFeatures()
    result = poly.fit_transform(iris.data)
    print(result)

    #基于单变元函数的数据变换可以使用一个统一的方式完成，使用preprocessing库的FunctionTransformer对数据进行对数函数转换的代码：
    
    from numpy import log1p
    from sklearn.preprocessing import FunctionTransformer
    
    #自定义转换函数对数函数的数据转换
    #第一个参数是单变元函数
    fun = FunctionTransformer(log1p)
    result = fun.fit_transform(iris.data)
    print(result)


#特征选择
'''
数据预处理后，需要将有意义的特征输入机器学习的算法进行算法和模型进行训练
1）方面1：特征是否发散；如果一个特征不发散，例如方差接近0，就是在这个特征上基本没有差异，这个特征对于样本的区分并没有用
2）方面2：特征与目标的相关性：与目标的相关性高的特征，应当优先选择，除了方差外，还有相关性
3）根据特征选择的形式，可以将特征选择方法分为3种：
    （1）Filter:过滤法，按照发散性或者相关性，对各个特征进行评分，设定阈值或者选择阈值的个数，选择特征
    （2）wrapper:包装法，根据目标函数（通常涉及预测效果评分），每次选择若干特征，或者排除若干特征；
    （3）Embedded:嵌入法，先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。类似于过滤方法，这个就是通过训练来确定特征的优劣。
'''

'''
过滤方法：
1）方差选择
    使用方差选择方法，先计算各个特征的方差，根据阈值，选择方差大于阈值的特征。
'''
from sklearn.feature_selection import VarianceThreshold
def varianceThreshold():
    iris = load_iris()
    #方差选择法，返回值为特征选择后的数据
    #参数threshold为方差的阈值
    vaH = VarianceThreshold(threshold=3)
    print(iris.data)
    result = vaH.fit_transform(iris.data)
    #将大于阈值的那一列的特征数值打印出来
    print(result)

'''
相关系数法：
    使用相关系数法，先要计算各个特征对目标值的相关系数以及相关系数的P值。
'''
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
def selectkBest():
    iris = load_iris()
    #选择k个最好特征，返回选择特征后的数据
    #第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（平分，p值）的数组，数组第i项为评分和p值，在此定义为计算相关系数
    skb = SelectKBest(lambda X,Y:array(list(map(lambda  x:pearsonr(x,Y),X.T))).T,k=2)
    result = skb.fit_transform(iris.data,iris.target)
    print(result)






from pandas import Series
if __name__=='__main__':
    # minMaxScaler()
    # normalize()
    # binarizer()
    # oneHotEncoder()
    # imputer()
    # polynomialFeature()
    # varianceThreshold()
    selectkBest()



