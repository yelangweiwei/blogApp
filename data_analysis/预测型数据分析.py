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

决策树：量化纯度的方法：熵，gini,错误率，一般使用熵公式
纯度差：也称为信息增益

过度拟合的原因：1）噪音数据，在训练的数据中噪声的数据很多  2）缺少代表性的数据，导致某一类的数据不能很好的匹配，这一点可以通过混淆矩阵分析得出
3）多重比较：

优化的方案：
1）修剪枝叶：
a)前置裁剪：在构建决策树之前就停止，切分节点的条件很苛刻，导致决策树很短小，结果就是决策树无法达到最优，
b)后置裁剪：决策树构建好之后，才开始裁剪，1）用单一的叶子节点代替整个子树，叶子节点的分类采用子类中最主要的分类；2）将一个子树代替另一棵树；有些浪费
优化方案2：
k折交叉验证：首先计算出整体的决策树，子节点个数记做N，设i属于[1,N],对每个i使用交叉验证，并裁剪到i个节点，计算错误率，最后求出平均错误率，这样可以用具有最小错误率的对应的i
作为最终决策树的大小，对原始决策树进行裁剪，得到最优的决策树

优化方案3:随机森林
random forest 用训练数据随机的计算出许多决策树，形成一个森林，然后使用森林对未知的数据进行预测，选取投票最多的分类；这个得到的错误率经过了进一步的降低。


准确率的估计：使用统计学中的置信区间

'''


#随机森林
'''
1）随机森林的分类效果  a)森林中任意两棵树的相关性越强，错误率越大  b)森林中的树的分类能力越强，整个森林的错误率越低
2）特征的选择m，m减小，树的相关性和分类能力会下降，变大，两者也会增大


'''


#聚类算法：（k均值，DBSCAN）
'''
1)聚类是无监督学些，本质是抖索数据结构的关系，常用于对客户细分，对文章聚类。
分类：已知有哪些标签进行分类，已知存在有哪些类别。

kmeans:
这种算法适用于：数据集呈现数圆形和球形分布的数据，如果数据没有呈现出这种规律，很可能聚类的效果很差
'''
from sklearn.cluster import KMeans
def kmean_pratice():
    #读取数据
    iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
    iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

    g = seaborn.FacetGrid(iris, hue='Species')
    g.set(xlim=(0, 3), ylim=(0, 9))
    g.map(plt.scatter, 'PetalWidthCm', 'PetalLengthCm')
    g.add_legend()


    #使用模型训练
    k= 3
    km = KMeans(k)   #k为聚簇的数目
    #使用的数据
    x = iris[['PetalWidthCm','PetalLengthCm']]
    km.fit(x)
    iris['cluster_k3'] = km.predict(x)
    print(iris['cluster_k3'])

    # 探索数据分析
    g = seaborn.FacetGrid(iris, hue='cluster_k3')
    g.set(xlim=(0, 3), ylim=(0, 9))
    g.map(plt.scatter, 'PetalWidthCm', 'PetalLengthCm')
    g.add_legend()
    plt.show()

#DBSCAN  基于密度的聚类方法
'''
1)目的：在于过滤低密度区域，发现稠密度样本点，跟传统的基于层次的聚类和划分聚类的凸形聚类簇不同，改算法可以发现任意形状的聚类簇。
2）特点：
a)基于密度的特点是不依赖距离，依赖于密度，从而客服基于距离的算法只能发现球形聚簇的缺点
3）核心思想是：从某个核心触发，不断向密度可达的区域扩张，从而得到一个包含核心点的最大化区域，区域中的热议两点密度相连

优点：
1）克服基于距离的算法，只能发现类圆形的聚类的缺点
2）可发现任意形状的聚类，且对噪声不敏感
3）不需要指定类的数目
4）算法中只有两个参数，扫描半径和最小包含的点数
缺点：
1）计算复杂，不进行任何优化时，事件的复杂度是O(N^2),通常可利用R-tree,K-d tree ball tree 索引来加速计算，并将算法的复杂度降低为O(NLog(N))
2）受eps影响大，数据分布密度不均匀时，密度小的cluster会被划分到多个性质相似的cluster,eps较大；会使距离较近，且密度较大的cluster被合并成一个cluster，在高维数据时，因为维数灾难问题，eps的选取比较困难
3）会依赖距离公式的选择，距离 标量标准不重要
4）不适合数据集集中密度差异很大的，因为eps和metric选取困难

'''



from pandas import DataFrame
from sklearn.cluster import DBSCAN
def descan_pratice():
    #生成圆形的随机样本，n_samples长程样本点个数，factor：是内圆和外圆比例有因子，noise:高斯噪声标准差
    noisy_circles = datasets.make_circles(n_samples=1000,factor=0.5,noise=0.05)
    print(noisy_circles)

    df = DataFrame()
    df['x1'] = noisy_circles[0][:,0]
    df['x2'] = noisy_circles[0][:,1]
    df['label'] = noisy_circles[1]
    temp = df.sample(10)  #采样点，取10个点

    #探索性数据分析
    # g = seaborn.FacetGrid(temp,hue='label')
    # g.map(plt.scatter,'x1','x2')
    # g.add_legend()
    # plt.show()

    #使用dbscan
    dbscan = DBSCAN(eps=0.2,min_samples=10)
    x = df[['x1','x2']]
    dbscan.fit(x)
    df['dbscan_label'] = dbscan.labels_
    g = seaborn.FacetGrid(df,hue='dbscan_label')
    g.map(plt.scatter,'x1','x2')
    g.add_legend()

    #使用kmean
    km = KMeans(2)
    x = df[['x1','x2']]
    km.fit(x)
    df['kmeans_label'] = km.predict(x)
    g = seaborn.FacetGrid(df,hue='kmeans_label')
    g.map(plt.scatter,'x1','x2')
    g.add_legend()

    plt.show()


from sklearn.metrics.pairwise import euclidean_distances
def DBSCN_Python():

    #获得数据
    noisy_circles = datasets.make_circles(n_samples=1000, factor=0.5, noise=0.05)
    print(noisy_circles)
    df = DataFrame()
    df['x1'] = noisy_circles[0][:, 0]
    df['x2'] = noisy_circles[0][:, 1]
    df['label'] = noisy_circles[1]

    #
    eps = 0.2
    MinPts = 5

    ptses = []
    dist = euclidean_distances(df)

    for row in dist:
        #密度
        density = py.sum(row<eps)
        pts =0
        if density>MinPts:
            pts=1
        elif density>1:
            pts = 2
        else:
            pts = 0
        ptses.append(pts)

    #把噪声点过滤掉，因为噪声点无法聚类，独自一类
    corePoints = df[pd.Series(ptses)!=0]
    coreDist = euclidean_distances(corePoints)

    #首先每个点的邻域都作为一类
    #邻域
    #空间中任意一点的邻域是以该点为圆心，以eps为半径的原区域内包含的点的集合
    cluster = dict()
    i = 0























if __name__ == '__main__':
    # linear_regression()
    # kNN_pratice()
    # decision_tree()
    # rand_forest()
    # kd_tree()


    # group,labels = createDataset()
    # print('training data set:',group)
    # print('labels of training data set:',labels)
    # #简单分类
    # tt = classify0([0,0],group,labels,3)
    # print('classification results:',tt)
    descan_pratice()
