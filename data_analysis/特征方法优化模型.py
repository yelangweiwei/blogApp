'''
数据和特征决定可机器学习的上限，模型和算法只是逼近这个上限而已

特征选择目的：
1）主要介绍从训练集合中挑选最合适的子集作为训练模型是所用的特征，使最终的模型的准确率最高
2）特征选择的定义：
a)特征选择，又名特征子集选择，或属性选择；从已有的M个特征中选择N个特征使得系统的特定指标最优化，是从原始特征中选择出的一些最有效特征以降低数据集维度的过程
3）特征选择的方法：
 (1)数据驱动，分析手上已有的训练数据，得出哪些x里面的特征对预测y最重要：
    a)相关性：考察已有的数据里面的特征x和预测值y的相关度
    b)迭代删除（增加）：确定要使用的那个算法后，选择最合适的训练子集，从而使模型效果最好
    c)基于模型，通过随机森林等可以直接得出每个训练特征的重要性的模型；或者在进行预测时加入的一些正则化调整，引起的对特征的筛选，从而挑选出最重要的特征
（2）领域专家：
    通过相关领域的专家知识，经验来挑选特征
    

'''


'''
相关性系数：皮尔逊系数
    1）用于度量两个变量x和y之间的相关性，其值介于-1和1之间；在自然科学领域，该系数广泛用于度量两个变量之间的相关程度。
    2）r = sum((x-x_mean)*(y-y_mean))/sqrt((x-x_mean)^2*(y-y_mean)^2)
'''
from scipy.stats.stats import pearsonr
def pearsonr_pratice(x,y):
    #判断x,y的迭代特征
    pearsonr(x,y)


'''
使用迭代特征进行选择：
0）暴力解决：将所有数据子集都测试一遍，用交叉验证来看哪个特征子集预测效果最好
1）增加：添加任何特征，模型性能都不会提升
2）递减：去除任何特征，模型性能都会下降
'''
import pandas as pd
import numpy as py
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
def iterator_selection():

    #读取数据
    iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

    #将分类数据化
    le = LabelEncoder()
    le.fit(iris['Species'])
    y = le.fit_transform(iris['Species'])

    #模型
    lm = linear_model.LogisticRegression(solver='liblinear',multi_class='ovr')
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

    selected_features = []
    rest_features= features[:]
    beat_acc = 0

    while len(rest_features)>0:
        temp_best_i = ''
        temp_best_acc = 0
        for feature_i in rest_features:
            temp_features = selected_features+[feature_i,]
            x = iris[temp_features]
            scores = cross_val_score(lm,x,y,cv = 5,scoring='accuracy')
            acc = py.mean(scores)
            if acc >temp_best_acc:
                temp_best_acc = acc
                temp_best_i = feature_i
        print('select',temp_best_i,'acc:',temp_best_acc)

        if temp_best_acc>beat_acc:
            beat_acc = temp_best_acc
            selected_features+=[temp_best_i,]
            rest_features.remove(temp_best_i)
        else:
            break
    print('beat features set:',selected_features,'acc:',beat_acc)



if __name__=='__main__':
    iterator_selection()


