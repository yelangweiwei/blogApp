import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import keras
from keras.datasets import mnist
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Flatten,Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def data_by_hand():
    #加载数据
    (train_x,train_y),(test_x,test_y) = mnist.load_data()
    #数据探索
    # print(type(train_x))

    #将三维的数据转换为4维的数据
    train_x = train_x.reshape(train_x.shape[0],28,28,1)
    print(train_x.shape)
    test_x = test_x.reshape(test_x.shape[0],28,28,1)
    #将每个图像的值缩小到0~1之间
    train_x = train_x/255
    test_x = test_x/255

    train_y = keras.utils.to_categorical(train_y,10)
    print(train_y.shape)
    test_y = keras.utils.to_categorical(test_y,10)

    #创建序贯模型
    model = Sequential()

    #第一层卷积层，6个卷积核，大小为5*5，relu为激活函数
    model.add(Conv2D(6,kernel_size=(5,5),activation='relu',input_shape=(28,28,1)))

    #第二层为池化层，最大池化
    model.add(MaxPooling2D(pool_size=(2,2)))

    #第三层，16个卷积核，大小为5*5，relu为激活函数
    model.add(Conv2D(16,kernel_size=(5,5),activation='relu'))

    #第四层池化层
    model.add(MaxPooling2D(pool_size=(2,2)))

    #将参数扁平化，在LeNet5中称为卷积层，实际上这一层是一维的向量，和全连接层一样,将多维的数据扁平化处理
    model.add(Flatten())

    #全连接
    model.add(Dense(120,activation='relu'))
    #全连接层
    model.add(Dense(84,activation='relu'))

    #输出层，用sortmax激活函数,计算分类概率
    model.add(Dense(10,activation='softmax'))

    #设置损失函数和优化器的配置
    model.compile(loss=keras.metrics.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    #传入训练数据进行训练
    '''
    batch_size:
    '''
    model.fit(train_x,train_y,batch_size=128,epochs=2,verbose=1,validation_data=(test_x,test_y))

    #对结果进行评估
    score = model.evaluate(test_x,test_y)
    print('误差:%0.4lf'%score[0])
    print('准确率:',score[1])






if __name__=='__main__':
    data_by_hand()





























