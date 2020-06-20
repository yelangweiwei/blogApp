
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#拟合变量
def fitting_args():
    x_data = np.float32(np.random.rand(2, 100))
    y_data = np.dot([0.100, 0.200], x_data) + 0.300

    # 构造线性模型
    b = tf.Variable(tf.zeros([1]))
    w = tf.Variable(tf.random_uniform([1, 2], -1, 1.0))
    y = tf.matmul(w, x_data) + b

    # 最小化方差
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # 初始化：
    init = tf.initialize_all_variables()

    # 启动图
    sess = tf.Session()
    sess.run(init)

    # 拟合平面
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(w), sess.run(b))

#mnist 入门
from tensorflow.examples.tutorials.mnist import input_data
#softmax回归模型的经典案例
def mnist_pra():
    mnist = input_data.read_data_sets('G:\\20190426\\zhouweiwei\\mygit\\blogApp\\tensorflow_pratice\\mnist_data\\',
                                      one_hot=True)

    sess = tf.InteractiveSession()  #这个比Session更加灵活，可以在运行图的时候，插入一些计算图，在交互是的环境中是有利的。
    x = tf.placeholder('float',[None,784])  #这里的None表示其值大小不定，用以指定batch的大小
    y_ = tf.placeholder('float', [None, 10]) #每一行为一个10维的向量，用于代表对应某一张图片的类别

    w= tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))


    y = tf.nn.softmax(tf.matmul(x,w)+b)

    #交叉熵是用来衡量我们的预测用于描述真相的低效性
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    #使用梯度下降算法
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    sess.run(init)
    #训练模型
    for i in range(10000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        train_step.run(feed_dict={x:batch_xs,y_:batch_ys})

    #评估模型
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    #将获得的bool值转换成浮点值，然后取平均
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print('-------------训练获得的精度: ',accuracy)

    #计算学习到的模型在测试数据集上的正确率
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))


#权重初始化
def weight_variable(shape):
    inital = tf.truncated_normal(shape,stddev=0.1) #加入少量的噪声打破对称性以及避免0梯度
    return tf.Variable(inital)

#偏置项
def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape) #用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题。
    return tf.Variable(inital)

#卷积和池化
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#多层卷积
import os
def conv_pra():
    data_path = os.path.dirname(os.path.realpath(__file__))
    mnist =  input_data.read_data_sets(data_path,one_hot=True)

    sess = tf.InteractiveSession()  # 这个比Session更加灵活，可以在运行图的时候，插入一些计算图，在交互是的环境中是有利的。
    x = tf.placeholder('float', [None, 784])  # 这里的None表示其值大小不定，用以指定batch的大小
    y_ = tf.placeholder('float', [None, 10])

    W_conv1 = weight_variable([5,5,1,32])  #前两维度是batch的大小，接着是通道数目，最后输出通道数目
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x,[-1,28,28,1])  #将x变成一个4d通道，第2,3，维对应的是宽，高，最后一维代表颜色通道，灰色是1，rgb是3

    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #第二层
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #密集连接层
    w_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

    #dropout   减少过拟合，在输出层之前添加dropout.作用是：除了可以屏蔽神经元的输出外，还会自动的处理神经元输出值的scale
    keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    #输出层
    w_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

    #训练和评估模型
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step =tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch = mnist.train.next_batch(50)
        if i%100 ==0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            print('step %d,training_accuracy %g'%(i,train_accuracy))
        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

    #测试
    print('test accuracy %g'%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))







if __name__ == '__main__':
    conv_pra()


