'''
数据挖掘的步骤：
    1）数据采集，数据分析，特征工程，训练模型，模型评估等
    2）sklearn工具可以方便的进行特征工程和模型训练工作。
    3）特征处理类的三个方法：fit,transform和fit_transform
    4)transform:主要用来对特征进行转换，从可利用信息的角度来说，转换分为无信息转换和有信息转换。无信息转换指不利用其他信息进行转换，比如指数，对数转换等
    5）有信息转换：从是否利用目标值向量可分为无监督转换和有监督转换。
    6）无监督转换指只利用特征的的统计信息的转换，统计信息包括：均值，标准差，边界等，比如标准化，PCA发降维等
    7）有监督指即利用特征信息，有利用目标值信息的转换。比如通过模型选择特征，LDA法降维
    8）有信息转换的通过fit方法实际有用，fit的主要工作是：获取特征信息和目标值信息；从这点来说：fit方法和模型训练时的fit方法就能联系起来：都通过
    分析特征和目标值，提取有价值的信息，对于转换类来说是某些统计量，对于模型来说可能是特征的权值系数。
    9）只有有监督的转换类的fit和transfrom方法才需要特征和目标值两个参数，fit方法无用不代表其没有实现，而是除合法性校验以外，其没有对特征和目标值进行
    任何处理。
'''

from numpy import hstack,vstack,array,median,nan
from numpy.random import choice
from sklearn.datasets import load_iris

def original_data():
    iris = load_iris()
    #特征矩阵加工
    #使用vstack增加一行含有缺失值的样本（nan,nan,nan,nan）
    #使用hstack增加一列表示花颜色（0-白，1-黄，2-红）,花的颜色是堆积的，意味着颜色并不影响花的分类
    print(iris.data.shape)
    print(array([nan,nan,nan,nan]).reshape(1,-1).shape)
    iris.data = vstack([iris.data,array([nan,nan,nan,nan]).reshape(1,-1)])
    #choice:从序列中随机选一个元素返回，并且序列中的值可以设定权重
    iris.data = hstack((choice([0,1,2],size=iris.data.shape[0]).reshape(-1,1),iris.data))
    print(iris.data.shape)

    #目标值向量加工
    #增加一个目标值，对应含缺失值的样本，值为众数
    iris.target = hstack((iris.target,array([median(iris.target)])))
    print(iris.target.shape)

'''
关键技术：
    1）并行处理，流水线处理，自动化调参，持久化是使用sklearn优雅的进行数据挖掘的和兴。并行处理和流水线处理多个特征处理工作，甚至包括模型训练工作组合成一个工作
    ，在组合的前提下，自动化调参技术帮我们省去人工调参的反锁，训练好的模型是储存在内存中的数据，持久化能够将这些数据保存在文件系统中，之后使用时无需在进行训练
    ，直接从文件系统中加载即可。
    2）并行处理使得多个特征处理工作能够并行的进行，根据对特征矩阵的读取方式不同，可分为整体并行处理和部分并行处理。整体并行处理，即并行处理的每个工作的输入都是
    特征矩阵的整体；部分并行处理，即可定义每个工作需要输入的特征矩阵的列。
'''
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import FeatureUnion
def featureUnoin_T():
    #新建，将整体特征矩阵进行对数函数转换的对象
    step2_1 = ('ToLog',FunctionTransformer(log1p))
    #新建将整体特征矩阵进行二值化类的对象
    step2_2 = ('ToBinary',Binarizer())
    #新建整体并行处理对象
    #该对象也有fit和fit_transform方法，fit和transform方法均是并行的调用需要并行处理的对象的fit和transfrom方法
    #参数transfromer_list为需要并行处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
    step2 = ('FeatureUnoin',FeatureUnion(transformer_list=[step2_1,step2_2]))

'''
整体并行处理有缺陷，在一些场景下，我们只需要对特征矩阵的某些列进行转换，而不是所有的列。pipline并没有提供相应的类（仅OneHotEncoder类实现了改功能）,需要在FeatureUnoin
的基础上进行优化，实现部分并行处理
'''
from sklearn.pipeline import FeatureUnion,_fit_one_transformer,_fit_transform_one,_transform_one
from sklearn.externals.joblib import Parallel,delayed
from scipy import sparse
import numpy as np

#部分并行处理，继承FeatureUnoin,这些和FeatureUnion的不同之处是：在取值的时候，选择了需要的列
class FeatureUnoinExt(FeatureUnion):
    #相比FeatureUnoin，多了idx_list参数，其表示每个并行工作需要读取的特征矩阵的列
    def __init__(self,transformer_list,idx_list,n_jobs=1,transformer_weight=None):
        self.idx_list = idx_list
        FeatureUnion.__init__(self,transformer_list=map(lambda trans:(trans[0],trans[1]),transformer_list),n_jobs=n_jobs,transformer_weights=transformer_weight)

    #由于只部分读取特征矩阵，方法fit需要重构
    def fit(self,X,y=None):
        transformer_idx_list = map(lambda trans,idx:(trans[0],trans[1],idx),self.transformer_list,self.idx_list)
        transformers = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入fit方法
            delayed(_fit_one_transformer)(trans,X[:,idx],y) for name,trans,idx in transformer_idx_list)
        self._update_transformer_list(transformers)
        return self

    #由于只读部分读取特征矩阵，方法fit_transfrom 需要重构
    def fit_transform(self, X, y=None, **fit_params):
        transformer_id_list = map(lambda trans,idx:([trans[0],trans[1],idx]),self.transformer_list,self.idx_list)
        result = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入fit_transfrom方法
            delayed(_fit_transform_one(trans,name,X[:,idx],y,self.transformer_weights,**fit_params)) for name,trans,idx in transformer_id_list)
        Xs,transformers= zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    #由于只读取部分特征矩阵，方法transform 需要重构
    def transform(self,X):
        transformer_id_list = map(lambda trans,idx:(trans[0],trans[1],idx),self.transformer_list,self.idx_list)
        Xs = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入transform方法
            delayed(_transform_one)(trans,name,X[:,idx],self.transformer_weights) for name,trans,idx in transformer_id_list)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

'''
在使用iris场景中，我们对特征矩阵的第i列（花的颜色）进行定性特征编码，对第2,3,4列进行对数函数转换，对第5列进行定性特征二值化处理
'''
from numpy import log1p
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
def fearureUnionExt_T():
    #新建将部分特征矩阵进行定性特征编码的对象
    step2_1 = ('OneHotEncoder',OneHotEncoder(sparse=False))
    #将部分特征矩阵进行对数函数转换的对象
    step2_2 = ('ToLog',FunctionTransformer(log1p))
    #将部分特征矩阵进行二值化类的对象
    step2_3 = ('ToBinary',Binarizer())

    #新建部分并行性处理对象
    #参数transformer_list 为需要并行处理的对象列表，改列表为二元组列表，第一元为对象的名称，第二元为对象
    #参数idx_list 为相应的需要读取的特征矩阵的列
    step2 = ('FeatureUnoinExt',FeatureUnoinExt(transformer_list=[step2_1,step2_2,step2_3],idx_list=[[0],[1,2,3],[4]]))


'''
流水线处理：
pipeline包提供了Pipeline类进行流水线处理，流水线上除最后一个工作外，其他都要执行fit_transform方法，且上一个工作输出作为下一个工作的输入，
最后一个工作必须实现fit方法，输入为上一个工作的输出，但是不限定一定有transform方法，以你为流水线的最后一个工作可能是训练。

构建完整的流水线：
'''

from numpy import log1p
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def pipline_T():

    #计算缺失值的对象
    step1 = ('Imputer',Imputer())
    #将部分特征矩阵进行定性特征编码的对象
    step2_1 = ('OneHotEncoder',OneHotEncoder(sparse=False))
    #将部分特征矩阵进行对象函数转换的对象
    step2_2 = ('ToLog',FunctionTransformer(log1p))
    #将部分特征矩阵进行二值化类的对象
    step2_3 = ('ToBinary',Binarizer())
    #将部分并行处理对象，返回值为每个并行工作的输出的合并
    step2 = ('FeatureUnoinExt',FeatureUnoinExt(transformer_list=[step2_1,step2_2,step2_3],idx_list=[[0],[1,2,3],[4]]))
    #新建无量纲化对象
    step3 = ('MinMaxScaler',MinMaxScaler())
    #新建卡法校验特征对象
    step4 = ('SelectKBest',SelectKBest(chi2,k=3))
    #新建PCA降维对象
    step5 = ('PCA',PCA(n_components=2))
    #新建逻辑回归对象，其为待训练的模型作为流水线的最后一步
    step6 = ('LogisticRegression',LogisticRegression(penalty='l2'))
    #新建流水线处理对象
    #参数steps为需要流水线处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二维为对象
    pipeline = Pipeline(steps=[step1,step2,step3,step4,step5,step6])
    return pipeline

'''
自动调参
    1）网格搜索是自动化调参的常见技术之一，GridSearchCV类，对组合好的对象进行训练以及调参
    2)这个类的意义：自动调参，只要把参数输入进去，就能给出最优化的结果和参数。
    3）优点：这是一个贪心算法，拿当前对模型影响最大的参数调优，直到最优化；在拿下一个影响最大的参数调优，直到所有的参数调整完毕。
    4）缺点：（1）只适用于小数据集，一旦数据的量级上去了，就很难得到结果。（2）可能会调到局部最优，不是全局最优，但是省时间省力。
    5）大数据量快速调优的方法:坐标下降
    
'''
from sklearn.model_selection import GridSearchCV
def auto_param_T():
    iris = load_iris()
    #新建网格搜索对象
    #第一参数为待训练的模型
    #param_grid 为待调参的组成的网格，字典格式，键为参数名称（格式“对象名称_子对象名称_参数名称”），值为可去的参数值列表
    pipeline = pipline_T()
    grid_search = GridSearchCV(pipeline,param_grid={'FeatureUnoinExt__ToBinary__threshold':[1.0,2.0,3.0,4.0],'LogisticRegression__C':[0.1,0.2,0.4,0.8]})

    #训练以及调参
    grid_search.fit(iris.data,iris.target)
    return grid_search

'''
持久化： externals.joblib包提供了dump和load方法来持久化和加载内存的数据
'''
from sklearn.externals.joblib import dump,load
def persistence_T():
    #持久化数据
    #第一个参数为内存中的对象
    #第二个参数为保存在文件系统中的名称
    #第三个参数为压缩级别，0为不压缩，3为合适的压缩级别
    grid_search = auto_param_T()
    dump(grid_search,'grid_search.dmp',compress=3)
    #从文件系统中加载数据到内存中
    grid_search = load('grid_search.dmp')






if __name__=='__main__':
    original_data()