from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from six import StringIO
from sklearn import  tree
import graphviz,os
import pydotplus


def cart_t():
    iris = load_iris()
    features = iris.data
    lables = iris.target
    train_features,test_features,train_labels,test_labels = train_test_split(features,lables,test_size=0.33,random_state=0)
    #构造cart分类树
    clf = DecisionTreeClassifier(criterion='gini')
    #拟合
    clf = clf.fit(train_features,train_labels)
    #预测
    test_predict = clf.predict(test_features)
    #预测结果和测试集进行对比
    score = accuracy_score(test_labels,test_predict)
    print('acrt 分类树的准确率:%.4lf'%score)

    #将生成的数据放在指定的文件中，通过gvedit.exe进行展示就可以
    # dot_data = tree.export_graphviz(clf,out_file="G:\\20190426\\zhouweiwei\\mygit\\blogApp\\data_analysis\\graphviz\\cart.dot",feature_names=iris.feature_names)
    # graph = graphviz.Source(dot_data)
    # graph

    #直接生成pdf
    dot_data= StringIO()
    tree.export_graphviz(clf,out_file=dot_data,feature_names=iris.feature_names,class_names=iris.target_names,filled=True,rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('G:\\20190426\\zhouweiwei\\mygit\\blogApp\\data_analysis\\graphviz\\cart.pdf')
    print('visible tree plot saved as pdf')


'''
泰坦尼克乘客生存的预测
决策树
'''
import pandas as pd

def load_data():
    train_path = os.path.dirname(os.path.realpath(__file__))+'/data/titanic/Titanic_Data-master/train.csv'
    test_path = os.path.dirname(os.path.realpath(__file__))+'/data/titanic/Titanic_Data-master/test.csv'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    '''数据探索'''
    #打印数据的信息，了解数据的基本情况,了解数据的行数，列数，完整度，数据的类型
    # print(train_data.info())
    # print(test_data.info())
    #统计情况
    # print(train_data.describe())
    #查看字符串类型的整体情况
    # print(train_data.describe(include=['O']))

    #使用head查看前几行数据
    # print(train_data.head())
    # print(train_data.tail())
    '''数据清洗'''
    #在进行数据探索的时候，发现age和fare,cabin缺失较大，embarked确实少。
    #使用平均值来填充age的nan值
    train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
    test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
    print('训练数据：')
    print(train_data.info())
    print('测试数据：')
    print(test_data.info())

    #使用平均票价填充缺失的值
    train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
    print('训练数据')
    print(train_data.info())
    print('测试数据：')
    print(test_data.info())

    #cabin为船舱，无法补齐;embarked少量缺少，可以少量补齐
    print(train_data['Embarked'].value_counts())  #通过这个可以看到各个仓的人数，使用s仓的补齐
    train_data['Embarked'].fillna('S',inplace=True)
    test_data['Embarked'].fillna('S',inplace=True)

    '''特征选择
        要找出有用的字段，去掉无用的字段
        1)无关的字段：passengerId,name，Ticket(船票号)，Cabin(有丢失，不要了)
    '''
    # print(train_data.info())
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    train_features = train_data[features]
    train_labels = train_data['Survived']
    test_features = test_data[features]

    #特征值里有一些字符串，不利于后续的运算，需要转化为数值类型，使用dictVectorizer
    from sklearn.feature_extraction import DictVectorizer
    dvec = DictVectorizer(sparse=False)
    train_features = dvec.fit_transform(train_features.to_dict(orient='record'))

    # print(dvec.feature_names_)

    #得到分类数模型,构造ID3分类树
    rf = DecisionTreeClassifier(criterion='entropy')
    #决策树分类
    rf.fit(train_features,train_labels)

    '''模型评估和预测'''
    test_featrues = dvec.fit_transform(test_features.to_dict(orient='record'))
    # pre_labels = rf.predict(test_featrues),测试集没有结果，预测的结果没有对比的数据，因此这个预测就不能用了

    from sklearn.model_selection import cross_val_score
    import numpy as py
    print('cross_val_score的准确性%.4lf：',py.mean(cross_val_score(rf,train_features,train_labels,cv=10)))


    '''决策树可视化'''
    dot_data = StringIO()
    out_file = os.path.dirname(os.path.realpath(__file__))+'/graphviz/vis.pdf'
    tree.export_graphviz(rf,out_file=dot_data,feature_names=dvec.feature_names_)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(out_file)


if __name__ =='__main__':
    load_data()