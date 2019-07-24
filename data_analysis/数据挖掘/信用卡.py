#自动参数搜索模块，进行参数调优
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import  SVC
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import  seaborn as sns
import os
import pandas as pd
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

'''
使用多个模型分析信用卡的使用情况
'''

#对具体的分类器进行GridSearchCV参数调优
def GridSearchCV_work(pipeline,train_x,tran_y,test_x,test_y,param_grid,score='accuracy'):
    response = {}
    gridsearch = GridSearchCV(pipeline,param_grid=param_grid,scoring=score)

    #寻找最优的参数和最优的准确率方案
    search = gridsearch.fit(train_x,tran_y)
    print('最优的准确率 %0.4lf'%search.best_score_)
    print('最优的参数:',search.best_params_)

    predict_y = search.predict(test_x)
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y,predict_y)
    return response


def credit_card_default_rate():
    #数据加载
    data_path = os.path.dirname(os.path.realpath(__file__))+'/data/UCI_Credit_Card.csv'
    data = pd.read_csv(data_path)

    #数据探索
    # print(data.describe())
    # print(data.shape)
    # print(data.head(5))

    #查看下一个月的违约情况,按值计数
    # next_month = data['default.payment.next.month'].value_counts()
    #将违约情况可视化
    # df = pd.DataFrame({'default.payment.next.month':next_month.index,'values':next_month.values})
    # # 用来正常显示中文标签
    # plt.rcParams['font.sans-serif'] = ['simHei']
    # plt.figure(figsize=(6,6))
    # plt.title('信用卡违约客户\n(违约:1,守约:0)')
    # sns.set_color_codes('pastel')
    # sns.barplot(x = 'default.payment.next.month',y='values',data = df)
    # locs,labels = plt.xticks()
    # plt.show()

    #获得要分析数据的有效数据和结果
    #删除Id列
    data.drop(['ID'],inplace=True,axis=1)
    # print(data.head(5))
    #获得结果
    target = data['default.payment.next.month'].values
    # print('target:',target)
    #获得要分析的数据
    columns = data.columns.to_list()
    columns.remove('default.payment.next.month')
    # print('columns:',list(columns))
    features = data[columns]
    # print('features:',features.head(5))

    #数据划分为训练数据和测试数据;使用30%的数据作为测试数据，其余的作为训练数据
    train_x,test_x,train_y,test_y = train_test_split(features,target,test_size=0.3,stratify=target,random_state=1)

    #构造各种分类器
    classifier = [
        SVC(random_state=1,kernel='rbf'),
        RandomForestClassifier(random_state=1,criterion='gini'),
        DecisionTreeClassifier(random_state=1,criterion='gini'),
        KNeighborsClassifier(metric='minkowski'),

    ]
    #分类器的名字
    classifier_names = [
        'svc',
        'randomforestclassifier',
        'decisiontreeclassifier',
        'kneighborsclassifier',
    ]

    #分类器参数
    classifier_param_grid = [
        {'svc__C':[1],'svc__gamma':[0.01]},
        {'randomforestclassifier__n_estimators':[3,5,6]},
        {'decisiontreeclassifier__max_depth':[6,9,11]},
        {'kneighborsclassifier__n_neighbors':[4,6,8]},
    ]
    results_list = []
    for model,model_name,model_param_grid in zip(classifier,classifier_names,classifier_param_grid):
        pipline = Pipeline([
            ('scaler',StandardScaler()),
            (model_name,model)
        ])
        result = GridSearchCV_work(pipline,train_x,train_y,test_x,test_y,model_param_grid,score='accuracy')
        results_list.append([model_name,result])
    #显示最后的结果
    for model_name,result in results_list:
        print(model_name,':',result['predict_y'],':',result['accuracy_score'])




























if __name__=='__main__':
    #使用GridSearchCV进行调优参数
    # gridSearchCV_te()
    #使用pipline进行流水线工作
    # pipLine_tes()


    #信用卡分析
    credit_card_default_rate()