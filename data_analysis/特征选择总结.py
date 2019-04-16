'''
scikit_learn 几种常见的特征选择方法

特征选择主要的两个功能：
    1）减少特征数量，降维，使模型泛化能力更强，减少过拟合
    2）增强对特征和特征值之间的理解

1，去掉取值变化小的特征
    假设某特征的的特征取值只有1和0，并且在所有输入样本中，95%的实例的该特征取值都是1，那就可以认为这个特征作用不大。如果100%都是1，那这个特征
    都就没有意义。当特征值是离散型变量的时候，这种方法才可用，连续的变量要离散化才可以使用。在实际中一般不太会有95%以上的都取某个值的特征存在。
    所以这种方法不太好用。可以将他作为特征选择的预处理，先去掉那些取值变化小的特征，在从接下来的特征选择方法中选择合适的进行进一步的特征选择。

2，单变量特征选择
    单变量特征选择能够对每一个特征进行测试，衡量该特征和相应变量之间的关系。根据得分扔掉不好的特征。对于回归和分类问题，采用卡方检验等方式进行特征检验
    （卡方检验：用方差来衡量观测频率和理论频率之间差异性的方法）
    方法简单，易于运行。      
'''


'''
2.1：Pearson相关系数
        （1）皮尔逊相关系数是一种最简单，能帮助理解特征和相应变量之间关系的方法，该方法衡量的是变量之间的线性相关性，结果取值区间是【-1,1】，-1
        表示完全负相关（这个变量下降，那个就会上升），+1表示完全正相关，0表示没有线性关系。
        2）pearson相关系数的明显的缺陷是：作为特征排序机制，只对线性关系敏感，如果关系是非线性的，即便两个变量具有一一对应的关系，相关性
        也可能接近0.
'''
import numpy as np
from scipy.stats import pearsonr
def pearson_t():
    #确保每次取值一样
    np.random.seed(0)
    size= 300
    x = np.random.normal(0,1,size)
    print('low noise',pearsonr(x,x+np.random.normal(0,1,size)))  #根据结果：相关性比较强，p值很小
    print('high noise',pearsonr(x,x+np.random.normal(0,10,size))) #相关性很弱，p值较大

    #数据不相关，数据不是线性相关的
    x1 = np.random.uniform(-1,1,100000)
    print(pearsonr(x1,x1**2)[0])


'''
2.2 互信息和最大信息系数
    想把互信息直接用于特征选择其实不是太方便：
    1）它不属于度量方式，也没有办法归一化，在不同数据级上的结果无法做比较
    2）对于连续变量的计算不是很方便（X,Y都是集合，x，y都是离散的取值），通常变量需要先离散化，而互信息的结果对离散化的方式很敏感。
    3）最大信息系数客克服了这两个问题，首先寻找一种最优的离散化方式，然后把互信息取值转化为一种度量方式，取值区间在[0,1].
    4）缺点：当零假设不成立时，MIC的统计就会收到影响，在有的数据集上不存在这个问题，但又的数据集上就存在这个问题
'''
from minepy import MINE
def mic_t():
    m = MINE()
    x = np.random.uniform(-1,1,10000)
    m .compute_score(x,x**2)
    #计算最大信息系数
    print(m.mic())

'''
距离相关系数：这里使用曼哈顿距离为例
    1)距离相关系数是为了克服pearson相关系数的弱点而生的在x和x^2这个例子中，即便pearson相关系数为0，我们也不能确定这两个变量是独立的（有
    可能是非线性）;但如果距离相关性系数是0，那么我们就可以说这两个变量是独立的。
    2）尽管MIC和距离相关系数关系存在，但是当变量之间的关系接近线性相关的时候，pearson相关系数仍然不可替代。第一：pearson相关系数计算速度快
    ，这在处理大规模的数据的时候很重要；第二：pearson相关系数的取值区间是[-1,1],而MIC和距离相关系数都是[0,1]，这个特点使得pearson行管系数表现的
    更加丰富，符号表示关系的正负，绝对值能够表示强度，pearson相关性的前提是两个变量的变量关系是单调的。
'''
from scipy.spatial.distance import correlation
def dis_euc():
    x = np.random.uniform(-1,1,10000)
    y = x**2
    #使用这个可以查看距离相似系数的关系
    corre = correlation(x,y)
    print(corre)

'''
基于学习模型的特征排序：
    1）思路：直接使用需要用的机器学习算法，针对每个单独的特征和响应变量建立预测模型
    2）pearson相关系数等价于线性回归中的标准化回归系数。假如某个特征和响应变量之间的关系是非线性的，可以用基于树的方法（决策树，随机森林）
    或者扩展的线性模型等。
    3）基于树的方法比较易于使用，对非线性关系的建模比较好，并且不需要太多的调试，但是注意过拟合问题。因此数的深度不要太大，在就是运用交叉验证。
'''
from sklearn.model_selection import cross_val_score,ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
def model_based_ranking():
    boston = load_boston()
    x = boston['data']
    y = boston['target']
    names = boston['feature_names']
    #有20个随机树，最大深度是4
    rf = RandomForestRegressor(n_estimators=20,max_depth=4)
    scores = []

    for i in range(x.shape[1]):
        #scoring 准确度评价标准：r2,最好的结果是1，也可能是赋值；作用是：一个常量模型总是预测y的期望值，忽视输入的特征，结果是0.
        #cv，交叉验证生成器或者可迭代的次数，ShuffleSplit，参数依次是：划分数据的测试，测试数据的个数，训练数据的比例

        # result = ShuffleSplit(len(x),3,0.3)
        # for train,test in result.split(x):
        #     print('--------tain:',x[train],'--------test:',x[test])

        score = cross_val_score(rf,x[:,i:i+1],y,scoring='r2',cv = ShuffleSplit(len(x),3,0.3))
        scores.append((round(np.mean(score),3),names[i]))
    print(sorted(scores,reverse=True))  #reverse==True将结果的值降序排列


'''
线性模型和正则化
    1）单变量特征选择方法独立的衡量每个特征与相应变量之间的关系
    2）另一种特征选择方法是：基于机器学习模型的方法
    3)有些机器学习方法本身就具有对特征进行打分机制，或者很容易将其运用到特征选择任务中，例如回归模型，SVM，决策树，随机森林等。
    4）这种方法叫做wrapper类型，即为：特征排序模型和机器学习模型是耦合在一起的；对于非wrapper类型的特征选择方法叫做filter类型
    下面案例：
        1）使用回归模型的系数来选择特征，越是重要的特征在模型中对应的系数就越大，而根输出变量越是无关的特征对应的系数就会越接近与0，在噪音不多的数据上，或者是数据量
        远远大于特征数的数据上，如果特征之间的相对来说是比较独立的，那么即便是最简单的线性回归模型也一样能取得非常好的效果。
'''
from sklearn.linear_model import LinearRegression
import numpy as np


def model_select():
    np.random.seed(0)
    size = 5000

    #a dataset with 3 features
    x = np.random.normal(0,1,(size,3))
    #y = x0+2*x1+noise
    y = x[:,0]+2*x[:,1] + np.random.normal(0,2,size)

    lr = LinearRegression()
    lr.fit(x,y)
    return lr.coef_

#a helper method for pretty_printing linear models
def pretty_print_linear(coefs,names=None,sort=False):
    if names==None:
        names = ['X%s'% x for x in range(len(coefs))]
    lst = zip(coefs,names)
    if sort:
        lst = sorted(lst,key=lambda x:-np.abs(x[0]))
    return '+'.join('%s * %s'%(round(coef,3),name) for coef,name in lst)

def model_select_coef():
    coefs= model_select()
    print(pretty_print_linear(coefs))





if __name__=='__main__':
    # pearson_t()
    # mic_t()
    # dis_euc()
    model_based_ranking()