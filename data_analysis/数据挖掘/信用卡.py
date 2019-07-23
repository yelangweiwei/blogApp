#自动参数搜索模块，进行参数调优
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
#使用iris进行分类
def gridSearchCV_te():
    #使用randomForest对iris数据集进行分类
    '''
    n_estimators:随机森林中决策树的个数，默认是0
    criterion:决策树的标准，默认是基尼指数。
    max_depth:决策树的最大深度，默认是None,也就是不限制决策树的深度，也可以设置一个整数，限制决策书的深度
    n_jobs:拟合和预测的时候cpu的核数，默认是1，也可以是整数，如果是-1，代表cpu的核数
    :return: 
    '''
    rf = RandomForestClassifier()
    paramters = {'n_estimators':range(1,11)}
    iris = load_iris()
    #使用gridSearchCV进行调优
    clf = GridSearchCV(estimator=rf,param_grid=paramters)
    #对iris进行分类
    clf.fit(iris.data,iris.target)
    print('最优分数:%.4lf'%clf.best_score_)
    print('最优参数:',clf.best_params_)

#使用pipline管道机制进行流水线作业
def pipLine_tes():
    '''
    采用的方式是：（'名称',方法）
    :return: 
    '''
    rf = RandomForestClassifier()
    pipline = Pipeline([
        ('scaler',StandardScaler()),#数据标准化,均值为0，方差为1
        ('pca',PCA()),   #对数据进行降维
        ('randomforestclassifier',rf)
    ])

    iris = load_iris()
    parameters = {"randomforestclassifier__n_estimators":range(1,11)}   #这里注意：那个模型的参数要标明：classifier__参数
    clf = GridSearchCV(estimator=pipline,param_grid=parameters)
    clf.fit(iris.data,iris.target)
    print('最优分数:%4lf'%clf.best_score_)
    print('最有参数:',clf.best_params_)




if __name__=='__main__':
    #使用GridSearchCV进行调优参数
    # gridSearchCV_te()
    #使用pipline进行流水线工作
    pipLine_tes()
