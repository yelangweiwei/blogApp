import theano
import numpy as np
from theano import tensor as T
rng = np.random
#为了测试，生成10个样本，每个样本是3维的向量，然后用于训练
def theano_logic():
    N= 10
    feats = 3
    D =(rng.randn(N,feats).astype(np.float32),rng.randint(size=N,low=0,high=2).astype(np.float32))
    print(type(D))

    #声明自变量x,以及每个样本对应的标签y（训练标签）
    x = T.matrix('x')
    y = T.vector('y')

    #随机初始化参数w,b=0,为共享变量
    w = theano.shared(rng.randn(feats),name='w')
    b = theano.shared(0.,name='b')

    #构造代价函数
    p_1 = 1/(1+T.exp(-T.dot(x,w)-b))  #s激活函数
    xent = -y*T.log(p_1)-(1-y)*T.log(1-p_1)  #交叉代价函数
    cost = xent.mean()+0.01*(w**2).sum()  #代价函数的平均值+L2正则项以防止过拟合，其中系数为0.01
    gw,gb = T.grad(cost,[w,b])
    #对总代价函数求参数的偏导数
    prediction = p_1>0.5
    #大于0.5预测值为1，否则为0
    train = theano.function(inputs=[x,y],outputs=[prediction,xent],updates=((w,w-0.1*gw),(b,b-0.1*gb))) #训练所需函数
    prediction = theano.function(inputs=[x],outputs=prediction)  #测试阶段的函数
    print(prediction)

    #训练
    training_steps = 1000
    for i in range(training_steps):
        pred,err = train(D[0],D[1])
        print(err.mean())  #查看代价函数下降的变化过程

from theano.ifelse import ifelse
import time
def con_t():
    a,b = T.scalars('a','b')
    x,y = T.matrices('x','y')
    z_switch = T.switch(T.lt(a,b),T.mean(x),T.mean(y))
    z_lazy = ifelse(T.lt(a,b),T.mean(x),T.mean(y))

    #optmizer:optimizer的类型结构图（可以简化计算，增加计算的稳定性）
    #linker：决定使用哪种方式进行编译（c/python）
    f_switch = theano.function([a,b,x,y],z_switch,mode=theano.Mode(linker='vm'))
    f_lazyifelse = theano.function([a,b,x,y],z_lazy,mode=theano.Mode(linker='vm'))
    val1 = 0
    val2 = 1
    big_mat1 = np.ones((1000,100))
    big_mat2 = np.ones((1000,100))
    n_times = 10
    tic = time.clock()
    for i in range(n_times):
        f_switch(val1,val2,big_mat1,big_mat2)
    print('f_switch:%f sec'%(time.clock()-tic))

    tic = time.clock()
    for i in range(n_times):
        f_lazyifelse(val1,val2,big_mat1,big_mat2)
    print('f_lazyifelse:%f sec'%(time.clock()-tic))


'''
theano 循环,递归和跟序列有关的操作可以使用scan来进行

'''
def one_step(coef,power,x):
    return coef *x**power

def scan_t():
    #定义单步的函数，实现a*x^n
    #输入参数的顺序要与下面的scan的输入参数对应
    coefs = T.ivector()  #每步变化的值，系数组成的向量
    powers = T.ivector()  #每步变化的值，指数组成的向量
    x = T.iscalar()  #每步不变的值，自变量
    #seq,out_info,non_seq与one_step函数的参数顺序一一对应   返回的result是每一项的符号表达式组成的list
    result,updates = theano.scan(fn = one_step,sequences=[coefs,powers],outputs_info=None,non_sequences=x)
    #每一项的值和输入的函数关系
    f_ploly = theano.function([x,coefs,powers],result,allow_input_downcast=True)
    coef_val = np.array([2,3,4,6,5])
    power_val = np.array([0,1,2,3,4])
    x_val =10
    print('多项式各项的值:',f_ploly(x_val,coef_val,power_val))

    #scan的返回的result是每一项的值，并没有求和，如果我们只想要多项式的值，可以把f_poly写成这样
    #多项式每一项的和与输入的函数关系
    f_poly = theano.function([x,coefs,powers],result.sum(),allow_input_downcast=True)
    print('多项式的值:',f_poly(x_val,coef_val,power_val))




'''
共享变量:
是实现机器学习算法参数更新的重要机制，返回共享的变量。使用get_value,set_value 来读取或者修改共享变量的值。
'''
from theano import shared
def shared_t():
    #定义一个共享值，并初始化
    state = shared(0)
    inc = T.iscalar('inclar')
    accumulator = theano.function([inc],state,updates=[(state,state+inc)])
    #打印state的初始化的值
    print(state.get_value())
    accumulator(1)
    print(state.get_value())

#-----------------------------------------线性代数
import numpy.linalg as LA  #导入numpy中的线性代数库
def norm_fanshu_t():
    x = np.arange(0,1,0.1)
    x1 = LA.norm(x,1)  #计算1的范数
    x2 = LA.norm(x,2)
    xa = LA.norm(x,np.inf)  #计算无穷范数
    print(x1)
    print(x2)
    print(xa)


'''
一个n阶方阵A能进行特征值分解的充分必要条件是：它含有n个线性无关的特征向量
'''
def diag_t():
    a = np.array([[1,2],[3,4]])  #示例矩阵
    A1 = np.linalg.eigvals(a)  #得到特征值   计算矩阵的特征值
    A2,V1 = np.linalg.eig(a)  #得到特征值和特征向量  返回包含特征值和对应特征向量的元组
    print(A1)
    print(A2)
    print(V1)

'''
奇异值分解：（SVD）
用途：降维，推荐系数，数据压缩等
将矩阵分解为奇异向量和奇异值，我们会得到一些类似特征分解的信息。
每个矩阵都有奇异值分解，但不一定都有特征分解。非方阵的矩阵就没有特征分解，这时我们只能使用奇异值分解。
SVD最有用的性质：可能是拓展矩阵求逆到非方阵上。\
A = UDV^T  u是一个m*m的方阵，D是一个m*n的矩阵  V是一个n*n的方阵。u，V是正交矩阵 ，D不一定是方阵，u的列向量称为左奇异向量，V的列向量称为有奇异向量
'''
def svd_t():
    data = np.mat([[1,1,1,0,0],
                   [2,2,2,0,0],
                   [3,3,3,0,0],
                   [5,5,5,0,0],
                   [0,0,0,3,2],
                   [0,0,0,6,6]])
    u,sigma,vt =np.linalg.svd(data)
    # print(u)
    print(sigma)
    # print(vt)
    #转换为对角矩阵
    diagv = np.mat([[sigma[0],0,0],[0,sigma[1],0],[0,0,sigma[2]]])
    print(diagv)

'''
迹运算：
返回的是矩阵对角线元素的和
用途：迹运算在转置运算下是不变的。多个矩阵相乘得到方阵的迹，与将这些矩阵中最后一个挪到最前面之后相乘得到的迹是相同的。
'''
def trace_t():
    c = np.array([[1,2,3],[4,5,6],[7,8,9]])
    Trc = np.trace(c)
    print(Trc)

    d = c-2
    Trd = np.trace(d)
    print(Trd)

    trcd = np.trace(c.dot(d))
    print(trcd)

    trdc = np.trace(d.dot(c))
    print(trdc)

''''
使用python实现主程序分析  principal component analysis  PCA
是一种主程序分析方法
定义：通过正交变换，将一组可能存在相关性的变量转换为一组线性不相关的变量，转换后的这组变量为主成分
'''
from sklearn.datasets import load_iris
from numpy.linalg import eig
def pca_t():
    iris_data = load_iris()
    X = iris_data.data
    k= 2
    #standardsize by remove average  去中心化，或者叫标准化
    X = X-X.mean(axis=0)

    #Caculate convariance matrix:   计算协方差    协方差：判断变量之间的相关性
    x_cov = np.cov(X.T,ddof=0)
    #caculate eigenvalues and eigenvectors of convariance matrix   计算特征值和特征向量
    eigenvalues,eigenvectors = eig(x_cov)

    #top k large eigenvectors
    klarge_index = eigenvalues.argsort()[-k:][:-1]
    k_eigenvectors = eigenvectors[klarge_index]


    print(np.dot(X,k_eigenvectors.T))


#二项式分布
import matplotlib.pyplot as plt
import math
from scipy import stats

def binomial_t():
    n = 20
    p = 0.3
    k = np.arange(0,41)
    #定义二项分布
    binominal = stats.binom.pmf(k,n,p)

    #二项分布可视化
    plt.plot(k,binominal,'o-')
    plt.title('binomial:n=%i,p =%.2f'%(n,p),fontsize=15)
    plt.xlabel('number of success')
    plt.ylabel('probality of success',fontsize=15)
    plt.grid(True)
    plt.show()

'''
离散型随机变量的分布情况，如果是连续型的随机变量的情况，通过使用概率密度函数来描述。
常用的连续型随机变量的概率密度函数：正态分布或者高斯分布，又叫钟型曲线。
从图上展示：标准差越大，图形越分散
'''
def norm_t():
    #平局值或者期望
    mu = 0
    #标准差
    sigma1 = 1
    sigma2 = 2
    #随机变量的取值
    x = np.arange(-6,6,0.1)
    y1 = stats.norm.pdf(x,0,1)  #定义正态分布的密度函数
    y2 = stats.norm.pdf(x,0,2)
    plt.plot(x,y1,label='sigma is 1')
    plt.plot(x,y2,label='sigma is 2')
    plt.title('normal $\mu$=%.1f,$\sigma$=%.1f or %.1f'%(mu,sigma1,sigma2))
    plt.xlabel('x')
    plt.ylabel('probability density')
    plt.legend(loc='upper left')
    plt.show()

'''
边缘概率
联合概率分布，求其中一个随机变量的概率分布的情况,定义在子集上的概率分布称为边缘概率分布
'''

'''
numpy.var()：求方差
numpy.cov():求协方差
numpy.corrcoef()：求相关系数
'''

'''
信息论：度量信息的几种常用的指标
1，信息量：是度量信息多少的一个物理量，从量上反应具有确定概率的事件发生时所传递的信息。香农把信息看做是一种消除不确定性的量。
    在实际的应用中，信息量通常用概率的负对数来表示，即：I= -log2P（以2为底的对数）；获取的信息量总是正的。
2，信息熵:是对随机变量不确定性的度量，又叫香农熵；用熵来评价整个随机变量X平均的信息量，而平均最好的度量就是随机变量的期望。信息熵越大
    包含的信息越多，那么随机变量的不确定性就越大。
'''
def hxP():
    p = np.arange(0,1.05,0.05)
    Hx = []
    for i in p:
        if i ==0 or i==1:
            Hx.append(0)
        else:
            Hx.append(-i*np.log2(i)-(1-i)*np.log2(1-i))
    plt.plot(p,Hx,label='entropy')
    plt.xlabel('p')
    plt.ylabel('H(x)')
    plt.show()

'''
条件熵：在一个变量x的条件下，（变量x的每个值都会取），另一个变量y熵对x的期望
互信息：又称为信息增益，用来评价一个事件出现对于另一个事件出现所贡献的信息量
I(x,y)= H(y)-H(x)  在决策树的特征选择中，信息增益为主要依据。
相对熵：
交叉熵；
'''
'''
使用交叉熵，及其注意事项
'''
import tensorflow as tf
def cross_entropy():
    #神经网路的输出
    logits = tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])
    #使用softmax的输出
    y = tf.nn.softmax(logits)
    #正确的标签只要一个1
    y_ = tf.constant([[0.0,0.0,1.0],[1.0,0.0,0.0],[1.0,0.0,0.0]])
    #计算交叉熵
    cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
    #直接计算神经网络的输出结果的交叉熵
    cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits))
    with tf.Session() as sess:
        softmax = sess.run(y)
        ce = sess.run(cross_entropy)
        ce2 = sess.run(cross_entropy2)
        print('softmax result=',softmax)
        print('cross_entropy result=',ce)
        print('softmax_cross_entropy_with_logits result=',ce2)





if __name__ == '__main__':

    cross_entropy()
    # binomial_t()
    # trace_t()
    # svd_t()
    # norm_fanshu_t()

    # c = np.array([[1,2,3],[3,4,5]])
    # d= c.T
    # print(c)
    # print(d)


    # shared_t()
    # scan_t()
    # con_t()

    # theano_logic()

    #更新共享变量参数,在第一次的时候，输出值
    # 定义一个共享变量，初始化值为w，w还是原来的初始化的值；在这次初始化的同时，w变为x+x,在调用这个函数的时候，使用的w是新的值
    # w = theano.shared(2)
    # x = theano.tensor.iscalar('x')
    # f = theano.function([x],w,updates=[[w,w+x]])  #定义函数自变量为x,因变量为w,当函数执行时，更新参数w = w+x
    # print(f(2))
    # print(w.get_value())
    # print(f(3))

    #自动求导
    # x = theano.tensor.fscalar('x')
    # y = 1/(1+theano.tensor.exp(-x))
    # dx = theano.grad(y,x)
    # f = theano.function([x],dx)
    # print(f(3))


    #符号计算图模型
    # x,y = theano.tensor.fscalars('x','y')
    # z1 = x+y
    # z2 = x*y
    # f = theano.function([x,y],[z1,z2])
    # print(f(2,3))



    #内置的变量类型
    #自定义数据类型
    '''broadcastable:是True或者False的布尔类型元组，元组的大小等于变量的维度，为True,表示变量在对应的维度上可以进行广播，否则不能进行广播
    
    '''
    # r = T.row()
    # r.broadcastable
    # mtr = T.matrix()
    # mtr.broadcastable
    # f_row = theano.function([r,mtr],[r+mtr])
    # R = np.arange(1,3).reshape(1,2)
    # print(R)
    # M= np.arange(1,7).reshape(3,2)
    # print(M)
    # print(f_row(R,M))

    #类型转换，共享变量
    # data= np.array([[1,2],[3,4]])
    # shared_type = theano.shared(data)
    # print(type(shared_type))





    #初始化张量
    # x =T.scalar(name='input',dtype='float32')
    # w = T.scalar(name='weight',dtype='float32')
    # b = T.scalar(name='bias',dtype='float32')
    # z = w*x+b
    # #编译程序
    # net_input = theano.function(inputs=[w,x,b],outputs=z)
    # #执行程序
    # print('net_input:%2f'% net_input(2.0,3.0,4.0))




