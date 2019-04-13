'''
随机森林的特征选择法：——Gini Importance
原理：
    使用Gini指数表示节点的纯度，Gini指数越大，纯度越低。然后计算每个节点的Gini指数--子节点的Gini指数之和，记做Gini decrease.最后将所有树
    上相同那个特征节点的Gini decrease加权的和记为Gini importance.该数值会在0-1之间，该数值越大，即代表该节点（特征）重要性越大。
参数计算：
    Gini index:衡量决策树每一棵树上的节点上面所存在的数据的纯净度的一个指标，这个值越小，纯净度越高。
    公式：Gini(p) = sum(pi*(1-pi)) = 1-sum(pi^2);pi是节点内各个特征所占的概率
    Gini decrease:每个节点的Gini index -子节点的Gini index之和（这里的和是加权和）
    Gini importance:将所有树上相同特征节点的Gini decrease加权的和。

注意：
    1）这种方法存在偏向，对具有更多取值的特征会更有利
    2）对于存在关联的多个特征，其中任意一个都可以作为指示器（优秀的特征），并且一旦某个特征被选择之后，其他特征的重要度就会急剧下降，因为不纯度已经被
    选中的那个特征降下来了，其他的特征就很难在降低那么多的不纯度了，这样一来，只有先被选中的那个特征重要度很高，其他的关联特征重要度往往较低，在理解
    数据时，这就会造成误解，导致错误的认为先被选中的恩正很重要，其余的特征不重要。但实际上这些特征对响应变量的作用确实非常接近。
'''

import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder
def gini_importance_t():
    iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

    le = LabelEncoder()
    le.fit(iris['Species'])
    rf = ensemble.RandomForestClassifier()

    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    y = np.array(le.transform(iris['Species']))
    x = np.array(iris[features])

    #Gini importance
    rf.fit(x,y)
    print(rf.feature_importances_)


'''
随机森林特征选择法--Mean Decrease Accuracy
原理：主要思路是大论每个特征法的特征值顺序，并且度量顺序变动对模型的精确率的影响。很明显，对于不重要的变量来说，打乱顺序对模型
    的精确率影响不会太大，但是对于重要的变量来说，打乱顺序就会降低模型的精确率。
    
实现步骤：
    1）训练一个随机森林模型，在测试集检验得到的accuracy0;
    2)随机重排测试集的某个特征xi,检验得到的accuracy1;
    3)(accuracy0-accuracy1)/accuracy0，即为特征xi的重要性。
'''
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
def decrease_accuracy_te():
    #读取数据
    iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

    #获得数据
    features= ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    x = np.array(iris[features])
    #种类二值化
    le = LabelEncoder()
    le.fit(iris['Species'])
    y = np.array(le.transform(iris['Species']))

    #模型实例化
    rf= ensemble.RandomForestClassifier()

    #将原始的数据分成10分，其中测试的数占10%，训练的数据占剩余的数据。，random_state：控制每次随机产生的数目都是一样的，可以不用
    rs = ShuffleSplit(n_splits=10,test_size=0.1,random_state=0)
    scores = np.zeros((10,4))
    count = 0

    for train_idx,test_idx in rs.split(x):
        x_train ,x_test = x[train_idx],x[test_idx]
        y_train,y_test = y[train_idx],y[test_idx]

        rf.fit(x_train,y_train)
        #获得测试数据和模型预测数据的精度
        acc = accuracy_score(y_test,rf.predict(x_test))
        for i in range(len(features)):
            #将测试的数据进行复制，不干扰原数据
            x_t = x_test.copy()
            #将拷贝的数据，某一列数据进行混排
            np.random.shuffle(x_t[:,i])
            #计算测试数据和模型预测的数据的精度
            shuff_acc = accuracy_score(y_test,rf.predict(x_t))
            #从这里的值看出某个特征的重要性。
            scores[count,i] = ((acc-shuff_acc)/acc)
        count+=1
    print(np.mean(scores,axis=0))



'''
线性回归特征选择---L1正则化Lasso
    1) 什么是正则化：监督问题就是在规则化参数的同时最小化误差；最小化误差是为了让我们的模型拟合我们训练的数据，而规则化参数是防止我们的模型
        过分拟合我们的训练数据。
    2）正则化的作用：
        (1):约束参数，降低模型的复杂度
        (2):规则项的使用还可以约束我们的模型的特性。这样就可以将人对这个模型的先验知识融入到模型的学习当中，强行地让学习到的模型具有人想要的特性，例如
            稀疏，低秩，平滑等。
    3）L1范数：
        （1）定义：向量中各个元素绝对值之和，也有个美称叫做“稀疏规则算子”
        （2）作用：由于L1范数的天然性质，对l1优化的解是一个稀疏解，因此l1范数也被叫做稀疏规则算子。通过l1可以实现特征的稀疏，去掉一些没有信息的特征，例如对用户
            的电影做分类的时候，用户有100个特征，可能只有十几个特征是对分类是有用的，大部分特征入身高体重是无用的。利用l1范数可以过滤掉。
        
    4）L1正则化Lasso：估计系数的线性模型。
        作用：是估计系数系数的线性模型，他倾向于使用具有较少参数值的情况，有效的减少给定解决方案所依赖变量的数量。因此，lasso及其变体是压缩感知领域的基础。在一定的条件下
        ，他可以恢复一组非零权重的精确集。
        (1) 普通线性回归：y = α0+α1*x1+...+αn*xn
        (2) 普通线性回归目标函数：min(sum((y-y_mean)^2))
        (3)lasso目标函数：min(sum(y-y_mean)^2)+θ*∑κ(κ=1....n)|αk|
            其中θ*∑κ(κ=1....n)|αk| 是l1的正则项，θ越大，对于系数α的惩罚就会越严重，所以会有更多的系数倾向于0（因为要是目标函数尽可能小，所以系数α会尽量变小）；θ越小
            ，对于系数α的惩罚就越轻，回归得到的系数会与接近与普通的线性回归。        
'''

# import numpy as np
# import pandas as pb
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
def lasso_t():
    iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    x = np.array(iris[features])

    le = LabelEncoder()
    le.fit(iris['Species'])
    y = np.array(le.transform(iris['Species']))

    #模型
    lm = linear_model.Lasso(0.02)
    lm.fit(x,y)
    print(lm.coef_)





if __name__=='__main__':
    # gini_importance_t()
    # decrease_accuracy_te()
    lasso_t()




