#-*- coding:utf-8 -*-
import argparse
import os
import numpy as np
import time
import tensorflow_pratice as tf
from tensorflow_pratice.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
DATA_DIT = os.path.dirname(os.path.realpath('__file__'))+'/data_analysis/data/estimator_data/data/'
MODEL_DIR = os.path.dirname(os.path.realpath('__file__'))+'/data_analysis/data/estimator_data/model/'
NUM_STEPS = 1000
tf.logging.set_verbosity(tf.logging.INFO)
print('using model dir:%s'%MODEL_DIR)

#2）定义模型函数
def cnn_model_fn(features,labels,mode):
    '''
    输入层
    Reshape X to 4-D tensor:[batch_size,width,height,channels]
    MNIST 图片是28*28像素，有一个彩色通道
    :param features: 
    :param labels: 
    :param mode: 
    :return: 
    '''
    input_layer = tf.reshape(features['x'],[-1,28,28,1])

    #卷积层1
    #32个5*5的卷积核，ReLU为激活函数
    #填充格式same
    #输入张量形状：[batch_size,28,28,1]
    #输出张量形状  [batch_size,28,28,32]

    '''activation：
        激活函数与其他层输出生成特征图，对某些运算结果平滑（微分），为神经元网络引入非线性（输入输出曲线关系），刻画输入复杂模型
        训练复杂模型，激活函数主要因素，单调，输出岁输入增长，可用梯度下降法找局部极值点；可微分，定义域内任意一点有倒数，输出可用梯度下降法。
        tf.nn.relu:修正线性单元，斜坡函数，分段线性，输入非负输出相同，输入为负输出为0，不受‘梯度消失’影响，取值范围[0,+∞],较大学习速率时，易受饱和神经元
        影响，损失信息但性能突出，输入秩张量（向量），小于0置0，其余分量不变
        tf.sigmoid:只接收浮点数，返回区间[0.0,1.0]内的值，输入值较大返回接近1.0，输入值较小，返回接近0，适用于真实输出位于[0.0,1.0]，输入接近饱和或变化剧烈，输出范围缩减称为问题
        输入0，输出0.5，sigmoid函数值域中间点。
        tf.tanh:双曲正切函数，值域[-1.0,1.0]，有输出负值能力，值域中间点为0，网络下层期待输入为负值或者0.0，会有问题
        tf.nn.dropout:依据可配置概率输出设0.0，适合少量随机性有助于训练，keep_prob参数指定输出保持概率，每次执行，不同输出，丢弃输出设为0.
    '''

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu
    )
    '''
    batch_normalization:优点：
        1)加快训练速度，这样可以使用较大学习率来训练网络
        2）提高网络的泛化能力
        3）BN层本质上是一个归一化的网络层，可以替代局部响应归一化层（LRN层）
        4）可以打乱样本训练顺序
        training为True，证明模型是在训练模式，否则我推理模式；在训练时，模型的moving_mean和moving_variance是不断变化的；在推理时，这些值是不变的。
    '''
    conv1 = tf.layers.batch_normalization(inputs=conv1,training=mode==tf.estimator.ModeKeys.TRAIN,name='BN1')

    #池化层1
    #使用2*2大小的卷积核，步幅是2
    #输入张量形状为 [batch_size,28,28,32]
    #输出张量形状 [batch_size,14,14,32]
    '''
         2D输入的最大池化层
         pool_size:指定池窗口的大小
         strides:用于指定池操作的步幅，可以是单个整数，用来知道你个所有空间维度的相同值。
         
         2个作用：
            平移，旋转，尺度不变性
            保留主要特征，同时减少参数（降维，效果类似PCA）和计算量，防止过拟合，提高模型泛化能力。
         
    '''
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

    #卷积层2
    #64个大小为5*5的卷积核
    #填充格式为same
    #输入张量形状 [batch_size ,14,14,32]
    #输出张量形状 [batch_size,14,14,64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters= 64,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu
    )

    #池化层2
    #最大池化层使用池化层2大小为2*2的卷积核，步幅为2
    #输入张量：【batch_size,14,14,64】
    #输出张量 [batch_size,7,7,64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

    #把张量展平为向量
    #输入张量形状[batch_size,7,7,64]
    #输出向量形状 [batch_size,7*7*64]

    pool2_flat = tf.reshape(pool2,[-1,7*7*64])

    #全连接层
    #全连接层共有1024个神经元
    #输入张量形状 [batch_size,7*7*64]
    #输出张量形状：[batch_size,1024]
    '''
    inputs:输入该网络层的数据
    unints：输出维度大小，改变inputs的最后一维
    activation:激活函数，即神经网络的非线性变化
    use_bias:使用bias为True(默认使用)，不用bias改为False，是否使用偏置项
    trainable=True:标明该层的参数是否参与训练
    输出结果的最后一维度就等于神经元的个数，即units的数值（神经元的个数）
    进行重新拟合，减少特征信息的损失
    '''
    dense = tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu,name='dense1')

    #增加一个dropout 操作，保持神经元的比率为0.6

    '''
    勤劳的神经元们事无巨细地干活，花大量时间把不重要的特征都学到了（过拟合），按这种方式去做其他事效率就很低（模型泛化能力差）

    现在随机挑选几个神经元组成小团队高效地工作（精兵简政），不仅速度快，效率高，工作的内容更有重点，还消除减弱了神经元节点间的联合适应性，增强了泛化能力。
        
    主要调试rate,查看随机生成的网络结构的数量，使用交叉验证的方式调试参数，查找最有效的参数。
    '''
    dropout = tf.layers.dropout(inputs=dense,rate=0.4,training=mode==tf.estimator.ModeKeys.TRAIN)

    #逻辑层
    #输入张量形状 [batch_size,1024]
    #输出张量形状 [batch_size,10]
    logits = tf.layers.dense(inputs=dropout,units=10)


    predictions = {
        # 产生预测值
        '''
        返回最大的那个值的下标
        input 是输入的矩阵
        axis:0表示按照按照列比较返回最大值的索引，1表示按照行比较返回最大值的索引，注意返回的是索引值
        '''
        'classes':tf.argmax(input=logits,axis=1),
        # 把softmax_tensor 添加到数据流图中，用来记录相关的日志
        '''
        目的：把一个N*1的向量归一化为（0，1）之间的值，由于采用指数运算，使得向量中数值较大的量的特征更加明显。
        其中的每个值表示这个样本属于每个类的概率，输出的向量的每个值范围是0到1   参考https://blog.csdn.net/wgj99991111/article/details/83586508
        '''
        'probabilities':tf.nn.softmax(logits,name='softmax_tensor')
    }
    prediction_output = tf.estimator.export.PredictOutput({
        'classes':tf.argmax(input=logits,axis=1),
        'probabilities':tf.nn.softmax(logits,name='softmax_tensor')
    })

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions,export_outputs={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_output})

    #计算代价函数
    onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,logits=logits
    )

    #Generate some summary info
    #scalar：包含单个值的实数张量
    tf.summary.scalar('loss',loss)
    tf.summary.histogram('conv1',conv1)
    tf.summary.histogram('dense',dense)

    #Configure the training Op(for TRAIN mode)

    if mode==tf.estimator.ModeKeys.TRAIN:
        '''
            减小损失
            实例化一个优化函数，并给予一定的学习率进行优化训练
            对学习率进行自适应约束，
            这个的学习率换成了梯度的均方根，，所以我们可以不设置学习率。
            在模型训练的初期和中期，，加速效果不错，训练速度快；在模型训练的后期，模型会反复的在局部最小值附近抖动。
        '''
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1e-4)
        '''
            不仅可以优化更新训练的模型参数，也可以为全局步骤计数
            该操作不仅可以计算出梯度，而且还可以将梯度作用在变量
            Add operations to minimize `loss` by updating `var_list`
        '''

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

    '''
        添加评估指标
        对比预测的结果和真实的标签的准确度
    '''
    eval_metric_ops = {
        'accuracy':tf.metrics.accuracy(
            labels=labels,predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

#3)定义读取训练数据的输入函数
def generator_input_fn(dataset,batch_size=BATCH_SIZE):
    def _input_fn():
        X = tf.constant(dataset.images)
        Y = tf.constant(dataset.labels,dtype=tf.int32)

        '''
            将队列中的数据打乱后在读取出来
            batch_size :从队列中提取新的批量的大小
            capacity：队列中的元素的最大数量
            min_after_dequeue:出队后队列中元素的最小数量，用于确保元素的混合级别;一定要确保这个参数小于capacity参数的值，否则会出错，这个代表队列中元素
            大于它的时候就会输出乱的顺序的batch。也就是说这个函数的输出结果是一个乱序的样本排列的batch，不是按照顺序排列的。
            enqueue_many:tensors中的张量是否都是一个例子
            
        '''
        image_batch,label_batch = tf.train.shuffle_batch([X,Y],
                                                         batch_size=batch_size,
                                                         capacity=8*batch_size,
                                                         min_after_dequeue=4*batch_size,
                                                         enqueue_many=True)
        return {'x':image_batch},label_batch
    return _input_fn

 # 8)保存模型
def serving_input_receiver_fn():
    feature_tensor = tf.placeholder(tf.float32, [None, 784])
    return tf.estimator.export.ServingInputReceiver({'x': feature_tensor}, {'x': feature_tensor})


if __name__== '__main__':

    # 4)加载训练数据及预测数据
    # 下载并加载数据
    mnist = input_data.read_data_sets(DATA_DIT)
    train_data = mnist.train.images  # return np.array
    # 将输入转换为一个array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    eval_data = mnist.test.images  # return np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    predict_data_batch = mnist.test.next_batch(10)

    # 5)创建Estimator
    #Estimator class to train and evaluate TensorFlow models
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=MODEL_DIR)

    # 设置预测日志
    # 记录预测值
    tensors_to_log = {
        'probabilities': 'softmax_tensor'
    }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=2000)

    print('6-----------------训练模型')
    mnist_classifier.train(
        input_fn=generator_input_fn(mnist.train, batch_size=BATCH_SIZE),
        #设置训练的次数
        steps=NUM_STEPS,
        hooks=[logging_hook]
    )

    print('7-----------------测试模型')
    # 测试模型并打印结果
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        # numpy array object or dict of numpy array objects. If an array,the array will be treated as a single feature
        x={'x': eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    print('8-----------------保存模型')
    exported_model_dir = mnist_classifier.export_saved_model(MODEL_DIR, serving_input_receiver_fn)
    decoded_model_dir = exported_model_dir.decode('utf-8')








