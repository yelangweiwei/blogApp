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
    # if all(names)==None:
    if names==None:
        names = ['X%s'% x for x in range(len(coefs))]
    lst = zip(coefs,names)
    if sort:
        lst = sorted(lst,key=lambda x:-np.abs(x[0]))
    return '+'.join('%s * %s'%(round(coef,3),name) for coef,name in lst)

def model_select_coef():
    coefs= model_select()
    print(pretty_print_linear(coefs))


'''
正则化模型
    1）正则化就是把额外的约束或者惩罚项加到已有的模型（损失函数）上，以防止过拟合并提高泛化呢你。损失函数由原来的E(X,Y)变为E(X,Y)+alpha||w||,
    w是模型系数组成的向量（有些地方叫做参数parameter,coeffieients）||.||一般是L1或则L2范数，alpha是一个可调的参数，控制着正则化的强度。
    当用线性模型时，L1正则化和L2正则化也成为Lasso和Ridge.

L1正则化/Lasso
    1)L1正则化经系数w的l1范数作为惩罚项加到损失函数上，由于正则项非零，这就迫使那些弱的特征所对应的系数变为0,。因此L1正则化往往会使用学到的模型很稀疏
    因为系数w经常为0.这个特征使得L1正则化则称为一种很好的特征选择方法。
lasso:（套索算法）
    通过构造一个惩罚函数得到一个较为精炼的模型，使得它的一些回归系数，即强制系数绝对值之和小于某个固定值，同时设定一些回归系数为零，因此保留了子集收缩的优点，
    是一种处理具有复共线性数据的有偏估计。
StandardScaler:
    在训练模型的时候，要输入features,也叫特征，对于同一个特征，不同的样本中的取值可能相差很大，一些异常小或者异常大的数据误导模型的正确训练；另外，如果数据分布很分散
    也会影响训练结果。以上两种情况都体现方差非常大，此时，我们可以将特征中的值进行标准化，即转化为均值为0，方差为1的正态分布。
    
'''
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

def pretty_print_linear(coefs,names=None,sort=False):
    # if all(names)==None:
    if names==None:
        names = ['X%s'% x for x in range(len(coefs))]
    lst = zip(coefs,names)
    if sort:
        lst = sorted(lst,key=lambda x:-np.abs(x[0]))
    return '+'.join('%s * %s'%(round(coef,3),name) for coef,name in lst)
def lass0_t():
    #加载波士顿房价
    boston = load_boston()
    #标准化数据，
    scaler = StandardScaler()
    x = scaler.fit_transform(boston['data'])
    y = boston['target']
    names = boston['feature_names']
    #alpha的作用：通过系数来压缩维度，从而降低维度，达到减小维度复杂度的目的。（有不同意见的请回应，谢谢）
    lasso= Lasso(alpha=0.3)
    lasso.fit(x,y)
    print('lasso model',pretty_print_linear(lasso.coef_,names,sort=True))

'''
l2正则化/ridge regression
l2正则化将系数向量的l2范数添加到损失函数中，由于l2惩罚系数是二次方的，这使得l2和l1有诸多差异，最明显的就是l2正则化会让系数的取值变得平均。对于关联特征，这意味着他们能够获得更相近的对应系数。还是以y= x1+x2,假设x1和x2具有很强的关联，如果用l1正则化，不论学到的模型是y = x1+x2还是y = 2*x1,惩罚都是一样的，都是2alpha,但是对于l2来说，第一个模型的惩罚项是2alpha,第二个模型的是4*alpha.可以看出，系数之和为常数是，各系数相等时惩罚项是最小的。所以才有了l2会让哥哥系数趋于相同的特点。
  l2正则化对于特征选择来说是一种稳定的模型，不想l1正则化那样，系数会因为细微的数据变化而变动。所以l2正则化和l1正则化提供的价值是不同的。l2对于特征理解来说更加有用：表示能力强的特征对应的系数是非零。
当做出结果的时候，会发现ridge的系数更加的稳定，更能反映内部的关系结构。
'''
def pretty_print_linear(coefs,names=None,sort=False):
    if names==None:
        names = ['X%s'% x for x in range(len(coefs))]
    lst = zip(coefs,names)
    if sort:
        lst = sorted(lst,key=lambda x:-np.abs(x[0]))
    return '+'.join('%s * %s'%(round(coef,3),name) for coef,name in lst)


from sklearn.linear_model import Ridge,LinearRegression
from sklearn.metrics import r2_score
def ridge_t():
    size = 100
    for i in range(10):
        print('Random seed is %s'%i)
        np.random.seed(i)
        x_seed = np.random.normal(0,1,size)
        x1 = x_seed+np.random.normal(0,.1,size)
        x2 = x_seed+np.random.normal(0,0.1,size)
        x3 = x_seed+np.random.normal(0,.1,size)
        x = np.array([x1,x2,x3]).T
        y = x1+x2+x3+np.random.normal(0,0.1,size)

        lr =  LinearRegression()
        lr.fit(x,y)
        print('line model:',pretty_print_linear(lr.coef_))

        ridge = Ridge(alpha=10)
        ridge.fit(x,y)
        print('ridge model:',pretty_print_linear(ridge.coef_))



'''
随机森林：
    1）有点：随机森林准确率高，鲁棒性好,易于使用等，是目前最流行的机器学习算法之一。
    2）提供的两种特征选择方法：平均不纯度减少和平均精度减少
    
'''

'''
平均不纯度减少：
    1）随机森林由多个决策树构成。决策树树中的每一个节点都是关于某个特征的条件，
    为的是将数据集按照不同的响应变量一分为二。利用不纯度可以确定节点（最优条件），
    对于分类分类问题，通常采用基尼不纯度或者信息增益，对于回归问题，通常采用的是
    方差或者最小二乘拟合。当训练决策树的时候，可以计算出每个特征减少了多少树的不
    纯度。对于一个决策树森林来说，可以算出每个特征平均减少了多少不纯度，并把它平
    均减少的不纯度作为特征选择的值。
    2）信息熵：一个随机的变量X可以代表n个随机事件，对应的随机变量为X= xi,那么熵
    的定义就是X的加权信息量。
    H(x) =p(x1)log2(1/p(x1))+...+p(xn)log2(1/p(xn))
    其中p(x1)就是xi发生的概率。
    例如：有32个球队，每个队的概率实力相当，那么每一队胜出的概率就是1/32,那么要
    猜对哪个队胜出的概率就比较困难。
    这个时候H(x) = 32 *(1/32)log2(1/(1/32)) = 5
    熵可以作为一个系统的混乱程度的标准。

    3）基尼不纯度：
        基尼不纯度的大概意思是：一个随机事件变成它对立事件的概率。
        例如：一个随机事件x,p(x=0) =0.5, p(x=1) = 0.5;那么基尼不纯度就是
        p(x=0)*(1-p(x))+p(x=1)*(1-p(x=1)) = 0.5
        
        一个随机事件Y，p(y=0) = 0.1,p(y=1) = 0.9
        那么基尼不纯度就为：p(y=0)*(1-p(y=0))+p(y=1)*(1-p(y=1))=0.1*0.9+0.9*0.1 = 0.18
        y=0时发生的概率就比较大，而基尼不纯度就比较小。
        基尼不纯度越低，纯度越高。可以用来衡量系统混乱程度的标准。
'''
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
def random_t():
    #Load boston housing dataset as an example
    boston = load_boston()
    X = boston["data"]
    Y = boston["target"]
    names = boston["feature_names"]
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),reverse=True))

'''
 运行结果后；
    1）这里特征得分实际上采用的是Gini Importance.使用基于不纯度的方法的时候，
        注意：（1）这种方法存在偏向，对具有更多类型的变量会更有利；
        （2）对于存在关联的多个特征，其中任意一个都可以作为指示器（优秀的特征），
        并且一旦某个特征被选择之后，其他的特征的重要度就会急剧下降，因为不纯度已
        经被选中的那个特征降下来了，其他的特征就很难在降低那么多的不纯度了，这样，
        只有先被选中的那个特征的重要程度很高，其他的关联特征的重要度往往较低。在
        理解数据时，就会造成误解，导致错误的认为先被选中的特征是很重要的，而其余
        的特征是不重要的，但实际上这些特征对响应变量的作用确实非常接近。
    2)特征随机选择方法稍微缓解了这个问题，但总的来说并没有完全解决。
'''
def featureSelection():
    size = 100000
    np.random.seed(seed=10)
    x_seed = np.random.normal(0,1,size)
    x0 = x_seed+np.random.normal(0,0.1,size)
    x1 = x_seed+np.random.normal(0,0.1,size)
    x2 = x_seed+np.random.normal(0,0.1,size)
    x = np.array([x0,x1,x2]).T
    y = x0+x1+x2
    #关注这里特征随机选择max_features
    rf = RandomForestRegressor(n_estimators=20,max_features=2)
    rf.fit(x,y)
    result= map(lambda z:round(z,3),list(rf.feature_importances_))
    print('scores for x0,x1,x2:',list(result))

'''
从结果中显示：x1的重要度比x2高，但是，实际他们的重要程度是一样的。
'''

'''
平均精度减少
    1）直接度量每个特征对模型精确率的影响，主要思路是：打乱每个特征的特征值顺序，并且度量顺序变动对模型的精确率
    的影响。很明显，对于不重要的变量来说，打乱顺序对模型的精确率影响不会太大，但是对于重要的变量来说，打乱顺序就
    会降低模型的精确率。
'''
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
def mean_decrease_accruacy():
    boston = load_boston()
    x = boston['data']
    y = boston['target']
    names = boston['feature_names']
    rf = RandomForestRegressor()
    scores = defaultdict(list)
    #混排数据，其中测试数据占30%
    for train_idx,test_idx in ShuffleSplit(len(x),0.3).split(x):
        x_train,x_test  = x[train_idx],x[test_idx]
        y_train,y_test = y[train_idx],y[test_idx]
        r = rf.fit(x_train,y_train)
        acc = r2_score(y_test,rf.predict(x_test))
        #打乱，然后每个特征，就是每一列的数据进行混排，看对总体的影响
        for i in range(x.shape[1]):
            x_t = x_test.copy()
            np.random.shuffle(x_t[:,i])
            shuff_acc = r2_score(y_test,rf.predict(x_t))
            scores[names[i]].append((acc-shuff_acc)/acc)
    print('features sorted by their scores:')
    print(sorted([(round(np.mean(score),4),feat) for feat,score in scores.items()],reverse=True))
'''
在这个例子中，通过运行发现，有的特征对模型的性能影响很大。尽管这些是在所有特征上进行了训练得到的模型，
然后才得到了每个特征的重要性测试，这并不意味着我们扔掉某些或者某个特征后模型的性能一定会下降很多，因
为删掉某个特征后，其关联的特征一样可以发挥作用，让模型能基本上不变。
'''


'''
两种顶层特征选择算法：
    1）之所以叫做顶层，是因为都是建立在基于模型的特征选择基础之上的，在不同的字迹上建立模型，然后汇总最终确定特征得分。
    2）稳定性选择：
        （1）稳定性选择是一种基于二次抽样和选择算法相结合的较新的方法，选择算法可以使回归，SVM或者其他类似的方法。
        它的主要思想是在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果。比如可以统计
        某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）。理想情况下，重要的特征
        的得分会接近100%，稍微弱一点的特征得分会是非0的数，而最无用的特征得分将会接近于0.
'''
from sklearn.linear_model import RandomizedLasso
from sklearn.datasets import load_boston
def stability_selection_t():
    boston = load_boston()
    x = boston['data']
    y = boston['target']
    names = boston['feature_names']
    #alpha=0是岭回归，=1是索回归
    # 当alpha从0变化到1，目标函数的稀疏解（部分变量的系数为0）也从0单调增加到lasso的稀疏解。
    rlasso = RandomizedLasso(alpha=0.025)
    rlasso.fit(x,y)
    print(sorted(zip(map(lambda x:round(x,4),rlasso.scores_),names),reverse=True))
'''
运行结果：值越大，越重要，得分会受到正则化参数alpha的影响，但是sklearnde 
随机lasso能够自动选择最优的alpha.接下来几个特征得分就开始下降。但是下降的不是特别急，这和纯lasso的方法和随机森立的结果不一样。能够看出稳定性选择对于克服过拟合和对数据理解来说是有帮助的；总的来说，好的特征不会因为有相似的特征，关联特征而得分为0.
'''


'''
递归特征消除（Recirsive feature elimination  =>RFE）
 1）主要思想是反复的构建模型，然后选出最好的或者是最差的特征（根据系数来选择），把选出来的特征放在一起，然后在剩余的特征上重复这个过程，直到所有的特征都遍历了。这个过程中特征别消除的次序就是特征的排序。因此，这是一种寻找最优特征子集的贪心算法。
 2）RFE的稳定性很大程度上取决于在迭代的时候底层使用哪种模型。例如，假如RFE
 采用普通的回归，没有经过正则化的回归是不稳定的，那么RFE就是不稳定的；加入采用Ridge,而用Ridge正则化的回归是稳定的,那么RFE就是稳定的。
'''
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
def ref_t():
    boston = load_boston()
    x = boston['data']
    y = boston['target']
    names = boston['feature_names']

    lr = LinearRegression()
    rfe  = RFE(lr,n_features_to_select=1)
    rfe.fit(x,y)
    print(sorted(zip(map(lambda x:round(x,4),rfe.ranking_),names)))





if __name__=='__main__':
    # pearson_t()
    # mic_t()
    # dis_euc()
    # model_based_ranking()
    # lass0_t()
    # ridge_t()
    # random_t()
    # featureSelection()
    # mean_decrease_accruacy()
    # stability_selection_t()
    ref_t()