#q欠拟合：
'''

'''
#过拟合
'''
在训练数据上拟合的很好，但是在训练集外的数据集上却不能很好的拟合，这个假设为过拟合的现象。主要的原因是：训练数据集中存在噪音或者训练数据少
'''
#查找临界点
'''
通过重采样和验证集方法找到完美的临界点
最流行的重采样技术是k折交叉验证。指的是在训练数据的子集上训练和测试模型k次，同时建立对于机器学习模型在未知数据上表现的评估。
'''

import pandas as pd
import seaborn
import numpy as py
from sklearn import linear_model
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

def linearRegression_pratice():
    #读取数据,展示数据
    iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    # seaborn.regplot(x ='PetalLengthCm',y ='PetalWidthCm',data =iris)
    # plt.show()

    #模型训练
    lm = linear_model.LinearRegression()
    #使用fit进行回归
    features = ['PetalLengthCm','SepalLengthCm']
    # features = ['PetalLengthCm']
    X = iris[features]
    Y= iris['PetalWidthCm']
    print(X.shape,Y.shape)
    model = lm.fit(X,Y)
    print(model.intercept_,model.coef_)

    #预测数据,将模型训练好后，使用一个随机的二维数据进行测试
    print(model.predict([[1,2]]))

    #预测性能的评估，交叉检验
    #neg_mean_absolute_error：得到每个回归模型的平均绝对值误差
    # scores = -cross_val_score(lm,X,Y,cv = 5,scoring='neg_mean_absolute_error')
    scores = -cross_val_score(lm,X,Y,cv = 5,scoring='neg_mean_squared_error')
    print(scores)

    #求平均值，作为误差结果
    print(py.mean(scores))
    #尝试修改scoring,得到neg_mean_squared_error

#回归对连续型数据进行预测；分类对离散型数据进行预测   获得的结果：回归的的y是连续型变量，分类的y是离散型变量
'''
1,正则化选择参数：penalty
LogisticRegression和LogisticRegressionCV默认自带正则化选项，可选择是：l1的正则化和l2的正则化
调参主要用于解决过拟合，一般选择L2正则化，如果还是过拟合，就选择L1;如果特征比较多，将一些不重要的特征系数归零，让模型系数化，也使用L1
penalty:影响损失函数的选择，基solver的选择，如果L2:newton-cg lbfgs libnear sag  如果是L1:只能liblinear,L1正则化的损失函数不是连续可导的；其它的三种优化算法需要损失函数的一阶
或者二阶连续导数
2.优化算法的选择：solver:
a)liblinear：使用了坐标轴下降算法来迭代优化损失函数
b)lbfgs:拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数
c)newtog-cg:利用二阶导数矩阵即海森矩阵来迭代优化损失函数
d)sag：随机平均梯度下降，是梯度下降的变种，和普通梯度下降的区别是：每次迭代仅仅用一部分样本来计算梯度适用于样本数据比较多的时候，SAG是一种
线性收敛算法
 newton-cg lbfgs sag在优化的时候需要损失函数的一阶或者二阶连续导数，，不能用于没有连续导数的L1正则化，只能用于L2正则化。liblinear可以用L1,L2正则化

liblinear 支持L1,L2,支持one-vs-rest（OvR多分类）   其他的三种只支持L2,支持OvR和MvM（many-vs-many）多分类;many-vs-many比OvR分类的准确性要高一些

3，分类方式的选择：multi_class
multi_class 决定了分类方式的选择，有ovr和multinomial（MvM）两个值可以选择，默认是ovr；

OvR的思想：无论是多少元逻辑回归，都可以看做是：二元逻辑回归；具体的做法是：对于第K类的分类决策，我们把所有的第K类的样本作为正例，除了第K类样本以外的所有样本都作为
负例，然后在二元逻辑回归，得到第k类的分类模型。
MvM:这里以one-vs-one（OvO）作为讲解：如果模型有T类，我们模型有T类，每次在所有的T类样本里选择两类出来，不妨记为T1类和T2类，把所有的输出为T1和T2的样本放在一起，把T1
作为正例，把T2作为负例，进行二元逻辑回归，得到模型参数。
比较：OvRa相对比较简单，但是分类效果相对略差，而MvM分类相对精确，但是分类速度没有OvR快

4：类型权重参数：class_weight:
    1)用于标识分类模型中各种类型的权重，可以不输入，默认是所有类型权重一致，如果选择输入，使用balanced让类库自己计算类型权重，或者自己输入权重，可以定义
    class_wieight= {1:0.9,0:0.1}
    2)使用balanced，类库会根据训练样本量来计算权重，某种类型样本量越多，则权重越低，样本量越少，则权重越高
    
5，样本权重参数：sample_weight:
    1)样本不平衡，导致样本不是总体样本的无偏估计，从而可能导致我们模型预测能力下降，可以通过调节样本权重来解决这个问题，调节样本权重两种方法：1）使用class_weight=balanced
      第二种：在调用fit函数的时候，通过sample_weight来调节每个样本的权重
    2）如果两种都用到了，真正的权重是：class_weight*sample*weight

模型评估：
AUC，
ROC：receiver operating characteristic


'''

iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']


def logistic_algorithm():

    #获取数据
    X = iris[features]

    #将species中的字符转换为数值类型
    le = LabelEncoder()
    le.fit(iris['Species'])
    #将离散值转换为标签
    y = le.transform(iris['Species'])
    print('将离散值转换为标签',list(y))

    # 构建模型
    lm = linear_model.LogisticRegression(solver='liblinear',multi_class='ovr')

    #通过交叉检验，得到分类的准确率
    scores = cross_val_score(lm,X,y,cv=5,scoring='accuracy')
    print('交叉检验准确率的平均值:',py.mean(scores))

#线性回归
def linear_regression():
    X = iris[features]
    le = LabelEncoder()
    le.fit(iris['Species'])
    y = le.transform(iris['Species'])
    lm = linear_model.LinearRegression()
    score = -cross_val_score(lm,X,y,cv=5,scoring='neg_mean_squared_error')
    print('linear_regression:',py.mean(score))




#其他的回归和分类算法
'''
1,原理：引用距离的分类
'''
from sklearn import neighbors
def kNN_pratice():

    #数值化species
    le = LabelEncoder()
    le.fit(iris['Species'])
    y = le.transform(iris['Species'])
    X = iris[features]

    #k近邻分类
    n_neighbors = 5  #这里选择5个近邻
    knn = neighbors.KNeighborsClassifier(n_neighbors,weights='uniform')  #uniform:即近邻点的权重都相同，或者distance,权重是距离的倒数
    # score = cross_val_score(knn,X,y,cv=5,scoring='accuracy')
    # print('kNN_pratice:', py.mean(score))

    knn_model = knn.fit(X,y)
    print(knn_model.predict_proba(X))

    #k近邻回归
    # knn = neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')
    # score = -cross_val_score(knn,X,y,cv = 5,scoring='neg_mean_squared_error')
    # print('kNN_pratice:',py.mean(score))

    # knn_model_instance = neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')
    # knn_model = knn_model_instance.fit(X,y)

    # print(knn_model.get_params(deep=True))
    # print(knn_model.kneighbors([[1,1,1,1]]))
    # print(knn_model.kneighbors(X,5,return_distance=False))
    # print(knn_model.kneighbors_graph(X))



from sklearn import tree
def decision_tree():
    # 读取数据
    X = iris[features]
    #将标签数字换
    le = LabelEncoder()
    le.fit(iris['Species'])
    y  = le.transform(iris['Species'])

    #设定模型
    #决策树分类
    # dt = tree.DecisionTreeClassifier()
    # score = cross_val_score(dt,X,y,cv = 5,scoring='accuracy')
    # print('decision_tree:', py.mean(score))

    #回归
    # dt = tree.DecisionTreeRegressor()
    # score = -cross_val_score(dt,X,y,cv = 5,scoring='neg_mean_squared_error')
    # print('decision_tree:',py.mean(score))

    dt = tree.DecisionTreeRegressor(random_state=0)
    dtmodel = dt.fit(X,y)
    print(X)


#随机森林;在决策树的基础上，引入多棵决策树，并综合所有决策树，根据少数服从多说或者求平均值的原则，一般，分类和回归要比knn和decistion要好
from sklearn import ensemble
def rand_forest():
    X = iris[features]
    le = LabelEncoder()
    le.fit(iris['Species'])
    y = le.transform(iris['Species'])
    #设定随机分类模型
    # rf = ensemble.RandomForestClassifier(5)
    # score = cross_val_score(rf,X,y,cv=5,scoring='accuracy')
    # print('rand_forest:',py.mean(score))

    #设定回归
    rf = ensemble.RandomForestRegressor(5)
    score = -cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
    print('rand_forest:', py.mean(score))

#kd k近邻理论和实践
'''
1)使用交叉验证的方法，选择k的最优值；k较小时，近似误差会减小，但是整体的复杂度升高，容易过拟合；k较大时，减小估计误差，但是近似误差会变大，整体变的简单，容易忽略细节的问题  
2）计算距离，使用曼哈顿或者欧式距离算法
3）决策一般就是：少数服从多数

'''
from sklearn.model_selection import train_test_split
def kd_tree():
    iris = datasets.load_iris()
    print(iris.data.shape)
    #对数据集进行分割，要做到随机采样，避免相同特征的数据聚集在一起
    #random_state:种子不同，获得的随机种子数不同，种子相同，实例不同，获得的随机种子数也相同，范围是0-2^32
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)
    print('x_train:',x_train.shape)
    print('x_test:',x_test.shape)
    print('y_train:',y_train.shape)
    print('y_test:',y_test.shape)


    #分类模型
    #导入数据标准化模块
    from sklearn.preprocessing import StandardScaler

    #标准换数据训练集
    '''
    1)统一资料中自变量或特征范围的方法。
    2）特征标准化，是各特征依照比例影响距离；第二个理由是：能加速梯度下降法的收敛
    3）(x-min)/(max-min)
    4）资料标准化后，使每个特征好着呢个的数值平均变为0（将每个特征值减去原来资料中的该特征的平均），标准差变为1
    '''
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)

    #标准化测试集
    x_test = ss.fit_transform(x_test)

    #knn分类器赋给变量
    knn = KNeighborsClassifier(algorithm='kd_tree')

    model = knn.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    print(y_test)
    print(y_predict)
    #检测模型的准确性
    accuracy_result = [y_test[i]  for i in range(len(y_predict)) if y_test[i]==y_predict[i]]
    print(list(accuracy_result))
    print('accuracy:',len(accuracy_result)/len(y_test))


#knn的理论实践
from numpy import *
import operator #运算符模块，执行排序操作时用到

def createDataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return (group,labels)

#简单分类
def classify0(inX,dataSet,labels,k):
    #shape[0]得到矩阵的行数，shape[1] 得到列数
    dataSetSize = dataSet.shape[0]
    #tile()得到和dataset相同的维数，进行相减
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    print(diffMat)

    #各向量相减后平方
    sqDiffMat = diffMat**2

    #axis = 1按行求和，得到平方和
    sqDistances = sqDiffMat.sum(axis=1)

    #开根号，求得输入向量和训练集各向量的欧式距离
    distances = sqDistances**0.5

    #得到各距离索引值，是升序，即最小距离到最大距离
    sortedDistIndicies = distances.argsort()

    classCount = {}

    for i in range(k):
        #前k个最小的距离的标签
        voteIlabel = labels[sortedDistIndicies[i]]
        #累计投票数
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    print('classCount:',classCount)

    #把分类后的结果进行排序，然后返回得票多的分类的结果
    #其中的iteitems()把字典分解为元组列表，itemgetter(1)按照第二个元素的次序对元组排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    print(sortedClassCount)
    #输出分类标签
    return sortedClassCount[0][0]

#随机森林
'''
1)集成学习和决策树于一身，优点：在随机森林算法中每棵树都尽可能最大成都的 生长，并没有剪枝过程
2）两个随机性：随机选择样本，随机的选择特征进行训练。这样使得随机森林不容易陷入过拟合，并且具有很好的抗噪能力
'''

#决策树和迭代决策树
'''
1)性能良好，与训练数据矛盾较小，2）泛化能力好；对训练数有很好的分类效果，对测试集有较低的误差率

三步骤：1）特征选择，2）决策树的生成 3）决策树的剪枝

ID3:核心：在决策树各个节点上应用信息增益准则选择特征，每一次都选择使得信息增益最大的特征进行分裂，递归的构建决策树
使用信息增益划分数据：缺点：选择取值比较多的特征，会具有较大的信息增益，ID3偏向于取值较多的特征

C4.5:根据信息增益比来选择特征
CART:指分类回归树。使用平方误差最小化作为选择特征的准则，用作分类树时采用基尼指数最小化，进行特征选择，递归的生成二叉树

剪枝：
1）决策树在生成过程中使用贪婪的方法来选择特征，从而达到对训练数据进行更好的拟合；剪枝是为了简化模型的复杂度，防止决策树过拟合问题


GBDT（迭代决策树）：是机器学习中常用的一种机器学习算法，
1）使用这个的目的是为了防止过拟合，过拟合让训练精度更高；这个不容易陷入过拟合，而且能达到更高的精度

'''



if __name__ == '__main__':
    # linear_regression()
    # kNN_pratice()
    # decision_tree()
    # rand_forest()
    # kd_tree()


    group,labels = createDataset()
    print('training data set:',group)
    print('labels of training data set:',labels)
    #简单分类
    tt = classify0([0,0],group,labels,3)
    print('classification results:',tt)