# -*- coding:utf-8 -*-
'''
利用TFLearn 解决线性回归的问题
'''
import tflearn
import numpy as np
import matplotlib.pylab as plt

def regression_by_tflearn():
    #生成随机数据
    X =np.linspace(-2,2,1000)
    np.random.shuffle(X)
    Y = 0.5*X+np.random.normal(0,0.5,(1000,))

    #定义线性回归模型
    #input_data：用于定义输入层，作为一个占位符，表示一个模型中输入数据的结构
    input_ = tflearn.input_data(shape=[None])
    #单个线性输入层
    linear = tflearn.single_unit(input_)
    #metric  评估函数，用于评估当前训练模型的性能，评价函数的结果不用于训练过程中
    #learning_rate：可理解为每一次梯度下降的补偿，一帮设置学习率小于0.01,
    regression = tflearn.regression(linear,optimizer='sgd',loss='mean_square',metric='R2',learning_rate=0.01)
    m = tflearn.DNN(regression)
    m.fit(X,Y,n_epoch=1000,show_metric=True,snapshot_epoch=False)
    print('\nRegression result:')
    print('Y='+str(m.get_weights(linear.W))+'*X'+str(m.get_weights(linear.b)))
    print('\nTest prediction for x = 3.2,3.3,3.4')
    print(m.predict([3.2,3.3,3.4]))


'''
利用TFLearn进行深度学习
'''
from __future__ import division,print_function,absolute_import
import tensorflow_pratice as tf
import os
import tflearn.datasets.mnist as mnist
from tflearn.layers.core import  input_data,dropout,fully_connected
from tflearn.layers.conv import  conv_2d,max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def tfLearn_deep_pratice():
    X,Y,testX,testY = mnist.load_data(one_hot=True)
    X = X.reshape([-1,28,28,1])
    testX = testX.reshape([-1,28,28,1])

    tf.reset_default_graph()

    network = input_data(shape=[None,28,28,1],name='input')

    #定义卷积层
    network = conv_2d(network,32,3,activation='relu',regularizer='L2')
    #定义最大池化层
    network = max_pool_2d(network,2)

    #进行归一化处理
    network = local_response_normalization(network)
    network = conv_2d(network,64,3,activation='relu',regularizer='L2')
    network = max_pool_2d(network,2)
    network = local_response_normalization(network)

    #全连接层





if __name__=='__main__':
    regression_by_tflearn()

