from __future__ import print_function  #其中引用的是新版本的函数，在低版本调用函数的时候，这里要调用高版本函数的特征
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

'''
图像处理：

'''



'''
tensorflow自编码器
'''
import os
import struct
import numpy as py
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input,Dense
from tensorflow.python.keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data

#为避免网络问题，这里我们定义处理本地数据集MINIST 的加载函数
def load_minist(path,kind='train'):
    '''Load MNIST data from path'''
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'%kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'%kind)
    with open(labels_path,'rb') as lbpath:
        magic,n = struct.unpack('>II',lbpath.read(8))   #将读取的字节序列转换为大端格式的无符号整型，将每2个字节转换为无符类型的整型
        labels = np.fromfile(lbpath,dtype=np.uint8)   #更高效的读取文件内容的方式，需要知道数据的类型
    with open(images_path,'rb') as images_path:
        magic,num,rows,cols = struct.unpack('>IIII',images_path.read(16))   #二进制数据转换为无符号int类型
        images = np.fromfile(images_path,dtype=np.uint8).reshape(len(labels),784)   #将4维数据转换为2维数据
    return images,labels

'''
自编码器的作用：
    1，数据去噪
    2，进行可视化降维
    
    设置合适的维度和稀疏约束，自编码器可以学习到PCA等技术更有意思的数据投影
    自编码器能从数据样本中进行无监督学习，这意味着可以将这个算法应用到某个数据集找那个，来取得良好的性能，且不需要任何新的特征工程，只需要适当的训练数据
    
    缺点：
        自编码器在图像压缩方面表现不好，由于在某个给定数据集上训练自编码器，因此它在处理和训练相类似的数据时可达到合理的压缩结果，但是在压缩差异较大的其他图像时效果不佳，这里，像JPEG这样的
        压缩计数在通用图像压缩方面会表现的更好
    
    训练自编码器，可以使输入通过编码器和解码器后，保留尽可能多的信息，但也可以训练自编码器来使新表征具有多种不同的属性，不同类型的自编码器旨在实现不同类型的属性
    四种不同的自编码器：
        1：香草自编码器
        2，多层自编码器
        3，卷积自编码器
        4，正则自编码器
'''

'''
香草自编码器：
    1：在这种自编码器中，只有三个网络层，即只有一个隐藏层的神经网络。它的输入和输出是相同的，可通过使用Adam优化器和均方差损失函数，来学习如何重构。
    隐含层维数是64，小于输入维数784，则称这个编码器是有损的。通过这个约束，来迫使神经网络来学习数据的压缩表征。
'''

def xiangcao_encode():
    input_size = 784
    hidden_size = 64
    output_size = 784

    x = Input((input_size,))
    #Encoder
    h = Dense(hidden_size,activation='relu')(x)
    #Decoder
    r = Dense(output_size,activation='sigmoid')(h)

    autoencoder = Model(input=x,output=r)
    autoencoder.compile(optimizer='adam',loss='mse')

'''
多层自编码器
    可以将自动编码器的隐含层数目进一步提高
    在这里使用了3个隐含层，不只是一个，任意一个隐含层都可以作为特征表征，但是为了使网络对称，我们使用最中间的网络层。
'''
def many_layer_encode():
    input_size = 784
    hidden_size = 128
    code_size = 64

    x = Input(shape=(input_size,))
    #encoder
    hidden_1 = Dense(hidden_size,activation='relu')(x)
    h  = Dense(code_size,activation='relu')(hidden_1)

    #Decoder
    hidden_2 = Dense(hidden_size,activation='relu')(h)
    r = Dense(input_size,activation='sigmoid')(hidden_2)

    autoencoder = Model(input=x,output=r)
    autoencoder.compile(optimizer='adam',loss='mse')

'''
卷积自编码器
    除了全连接层，自编码器应用到卷积层
    原理是一样的，要使用3D矢量（如图像）而不是展平后的一维矢量。对输入图像进行下采样，以提供较小维度的潜在表征，来迫使自编码器从压缩后的数据进行学习。
'''
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D
def conv_encode_decode():
    #输入是3维
    x = Input(shape=(28,28,1))

    #Encoder
    conv1_1 = Conv2D(16,(3,3),activation='relu',padding='same')(x)  #卷积中输出滤波器的数量，卷积窗，激活函数，
    '''
    池化层两个作用：
        1：平移，旋转，尺度不变性
        2：保留主要的特征同时减少参数（降维，效果类似PCA）和计算量，防止过拟合，提高模型泛化能力
    '''
    pool1 = MaxPooling2D((2,2),padding='same')(conv1_1)  #卷积池的窗大小，将输入的数据按照卷积窗的大小进行
    conv1_2 = Conv2D(8,(3,3),activation='relu',padding='same')(pool1)
    pool2 = MaxPooling2D((2,2),padding='same')(conv1_2)
    conv1_3 = Conv2D(8,(3,3),activation='relu',padding='same')(pool2)
    h = MaxPooling2D((2,2),padding='same')(conv1_3)

    #Decoder
    conv2_1 = Conv2D(8,(3,3),activation='relu',padding='same')(h)

    '''
     size:整数tuple,分别为行和列的上采样因子
     作用是：简单的用复制插值对原张量进行修改，也就是平均池化的逆操作。
    '''
    up1 = UpSampling2D((2,2))(conv2_1)
    conv2_2 = Conv2D(8,(3,3),activation='relu',padding='same')(up1)
    up2 = UpSampling2D((2,2))(conv2_2)
    conv2_3 = Conv2D(16,(3,3),activation='relu')(up2)
    up3 = UpSampling2D((2,2))(conv2_3)
    r = Conv2D(1,(3,3),activation='sigmoid',padding='same')(up3)

    autoencoder = Model(input=x,output=r)
    autoencoder.compile(optimizer='adam',loss='mse')

'''
正则自编码器：
    除了施加一个比输入维度小的隐含层，一些其他方法也可以用来约束自编码器的重构，如正则自编码器
    正则自编码器不需要使用浅层的编码器和解码器以及小的编码维数来限制模型的容量，而是使用损失函数来鼓励模型学习其他特征（除了将输入复制到输出），
    这些特性包括系数表征，小导数表征，以及对噪声或输入缺失的鲁棒性。
    常用到两种正则自编码器：稀疏自编码器和降噪自编码器
'''
'''
稀疏自编码器：
    一般用来学习特征，以便用于像分类这样的任务，稀疏正则化的自编码器必须反映训练数据的独特统计特征，而不是简单的充当恒等函数。以这种方式训练，执行附带
     稀疏惩罚的复现任务可以得到能学习有用特征的模型
     还有一种用来约束自动编码器重构的方法，是对其损失函数施加的约束，比如：对损失函数添加一个正则化约束，这样能使自编码器学习到数据的稀疏表征
     在隐含层，我们加入了L1正则化，作为优化阶段中损失函数的惩罚项，与香草自编码器相比，这样操作后的数据表征更为稀疏。
'''
from keras import regularizers
def sparse_encode():
    input_size = 784
    hidden_size = 64
    output_size = 784

    x = Input(shape=(input_size,))
    #Encoder 利用损失函数的惩罚项，学些训练数据集的独特统计特征。
    h = Dense(hidden_size,activation='relu',activity_regularizer=regularizers.l1(10e-5))(x) #施加在输出上的L1正则

    #Decoder
    r = Dense(output_size,activation='sigmoid')(h)

    autoencoder = Model(input=x,output=r)
    autoencoder.compile(optimizer='adam',loss='mse')


'''
降噪自编码器：
    1，这里不是通过对损失函数施加惩罚项，而是通过改变损失函数的重构误差项来学习一些有用的信息
    2，向训练数据加入噪声，并使自编码器学会去除这种噪声来获得没有被噪声污染过的真实输入。因此，这就迫使编码器学习提取最重要的特征并学习输入数据中更加鲁棒性
    的表征。这也是泛化能力比一般编码器强的原因
'''
def remove_noise_encode():
    x = Input(shape=(28,28,1))

    #Encoder
    conv1_1 = Conv2D(32,(3,3),activation='relu',padding='same')(x)
    pool1= MaxPooling2D((2,2),padding='same')(conv1_1)
    conv1_2  = Conv2D(32,(3,3),activation='relu',padding='same')(pool1)
    h = MaxPooling2D((2,2),padding='same')(conv1_2)

    #Decoder
    conv2_1 = Conv2D(32,(3,3),activation='relu',padding='same')(h)
    up1 = UpSampling2D((2,2))(conv2_1)
    conv2_2 = Conv2D(32,(3,3),activation='relu',padding='same')(up1)
    up2 = UpSampling2D((2,2))(conv2_2)
    r = Conv2D(1,(3,3),activation='sigmoid',padding='same')(up2)
    autoencoder = Model(input=x,output=r)
    autoencoder.compile(optimizer='adam',loss='mse')


def zibianma_code():
    #读取训练的数据和测试的数据
    dir_path = 'G:\\20190426\\zhouweiwei\\mygit\\blogApp\\data_analysis\\data\\mnist_data\\'
    x_train,y_train = load_minist(dir_path,kind='train')
    x_test,y_test = load_minist(dir_path,kind='t10k')
    # 将数据转换为浮点类型
    x_train = x_train.reshape(-1,28,28,1).astype('float32')
    x_test = x_test.reshape(-1,28,28,1).astype('float32')
    #归一化数据，使之在[0,1]之间
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    #对x_train展开为-1*784,前边为了做归一化，将数据转换为4维，这里又将数据转换为二维
    x_train = x_train.reshape(len(x_train),np.prod(x_train.shape[1:]))
    x_test = x_test.reshape(len(x_test),np.prod(x_test.shape[1:]))
    #定义输入层节点，隐含层节点数
    input_img = Input(shape=(784,))
    encoding_dim = 32
    #利用keras函数模型
    '''
    限制维度，使其小于输入的维度，这种情况称为有损自编码器，通过训练有损表征，使得自编码器能学到数据中最重要的特征；
    如果潜在表征的维度和输入相同，或是在完备案例中潜在表征的维度大于输入，上述结果也会出现。
    在这些情况下，即使使用线性编码器和线性解码器，也能很好的利用输入重构输出。且无需了解有关数据分布的任何有用信息
    在理想情况下，根据要分配的数据复杂度，来准确的选择编码器和解码器的编码维数和容量，就可以成功的训练任何所需的自编码器器结构。
    '''
    encoded = Dense(encoding_dim,activation='relu')(input_img)
    decoded = Dense(784,activation='sigmoid')(encoded)
    #创建自编码模型
    autoencoder = Model(inputs=input_img,outputs=decoded)
    #创建编码器模型
    encoder = Model(inputs=input_img,outputs=encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    #创建解码器模型
    decoder = Model(inputs=encoded_input,outputs=decoder_layer(encoded_input))
    #编译自编码器模型
    autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
    #训练该模型
    autoencoder.fit(x_train,x_train,epochs=50,batch_size=256,shuffle=True,validation_data=(x_test,x_test))
    #输出预测值
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    #显示10个数字
    n = 10
    plt.figure(figsize=(20,4))
    for i in range(n):
        #可视化输入数据
        ax = plt.subplot(2,n,i+1)
        plt.imshow(x_test[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #可视化自编码器学习的结果
        ax = plt.subplot(2,n,i+1+n)
        plt.imshow(decoded_imgs[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

#使用自编码预测信用卡欺诈
'''
    使用自动编码，进行异常检测
    根据模型进行欺诈预测
    这个自编码器先跳过，看看后边是不是都要使用自编码器 
'''


######################tensorflow实现Word2Vec
'''
    典型的词向量表示法有：独热表示，和分布式表示
    独热表示：独热表示的向量长度为词典的大小，向量的分量只有一个1，其他的全为0,1的位置对应该词在词典中的位置
    文本分类：使用词袋模型，将文章对应的稀疏矩阵合并为一个向量，然后统计每个词出现的频率；优点：存储简洁，
        缺点：容易受维数灾难的困扰，尤其是用于深度学习算法时。
            任何两个词都是孤立的，存在语义鸿沟词（任意两个词之间都是孤立的，不能体现词和词之间的关系）
            克服这种不足，人们提出分布式表示
    分布式表示：
        解决词汇和位置无关的问题，可以通过计算向量之间的距离（欧式距离，余弦距离）来体现词与词的相似性。基本的想法是直接使用一个普通的向量表示一个词。
        词向量的分布式表示解决了词汇和位置无关的问题。
    word2vec并非是深度学习的范畴，但是生成的词向量在很多任务中可以作为深度学习算法的输入，是深度学习在NLP领域的基础。
    
    Word2Vec原理：
        1：根据词汇周围来预测生成的概率，根据上下文来预测目标值，使用的模型CBOW 模型 
        2：根据目标值预测上下文，使用的模型：Skip-Gram模型。
'''

#导入需要的库

import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from urllib.request import urlretrieve
from sklearn.manifold import TSNE

#读取本地的数据
def read_data(file_name):
    '''
    将包含在zip文件中的第一个文件解压为单词列表
    :param file_name: 
    :return: 
    '''
    with zipfile.ZipFile(file_name) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()  #将读取的内容转换为字符串，并进行拆分
        print(data)
    return data

#构建数据集
vacabulary_size = 50000
def build_dataset(words):
    count = ['UNK',-1]
    '''
        collections.Counter:
    '''
    count.extend(collections.Counter(words).most_common(vacabulary_size-1))














if __name__ == '__main__':

    zip_file = 'C:\\Users\\Administrator\\Desktop\\temp\\krk_data.zip'
    read_data(zip_file)

    # zibianma_code()
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




