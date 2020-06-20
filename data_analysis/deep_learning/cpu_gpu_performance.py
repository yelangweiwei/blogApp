
'''
"/cpu:":表示cpu
“/gpu:0”:表示第一块gpu
“/gpu:1”：表示第二块gup
多个gpu就是将需要在每个gpu处处理的函数，放在指定的gpu设备中，在最后都要调用tf.Session
'''
import numpy as np
import tensorflow_pratice as tf
from  datetime import datetime

#gpu和cpu对比
def gpu_cpu_t():#验证后，cpu要很慢，gpu快很多，使用tx2实验的
    device_name ='/cpu:0'
    shape = (int(10000),int(10000))
    with tf.device(device_name):
        #形状为shap，元素服从minval 和maxval 之间均匀分布
        random_matrix = tf.random_uniform(shape=shape,minval=0,maxval=1)   #产生shape类型的，值在最小值和最大值之间的随机变量
        dot_operation = tf.matmul(random_matrix,tf.transpose(random_matrix))  #tf.transpose：转置    tf.matmul:求点积
        sum_operation = tf.reduce_sum(dot_operation)  #  求和，这里分按行求和（1）  按列求和（0）
        start_time = datetime.now()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session: #log_device_placement:处理单元日志
            result = session.run(sum_operation)
            print(result)
        print('\n'*2)
        print('Shape:',shape,'Device:',device_name)
        print('Time taken:',datetime.now()-start_time)

def tf_see_t():
    #构建计算图  常量
    a = tf.constant(1.,name='a')
    b = tf.constant(2.,shape=[2,2],name = 'b')
    #创建会话
    sess = tf.Session()
    #执行会话
    result= sess.run([a,b])
    print('result_a:',result[0])
    print('result_b:',result[1])


'''
变量：global_variable_initializer初始化所有的变量
在运算session前会初始化所有的变量
'''
def tf_initializer():
    #初始化所有的变量
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    #保存模型变量
    saver = tf.train.Saver()
    saver.save(sess,'./tmp/model/',global_step=100)
    #恢复模型变量
    #先加载meta graph并回复权重变量
    saver = tf.train.import_meta_graph('./tmp/model/-100.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/tmp/model/'))

#定义卷积神经网络运算规则，其中weights和biases为共享变量
def conv_relu(input,kernel_shape,bias_shape):
    #创建变量weight
    weight = tf.get_variable('weights',kernel_shape,initializer=tf.random_normal_initializer())
    #创建变量biases
    biases = tf.get_variable('biases',bias_shape,initializer=tf.constant_initializer())
    conv = tf.nn.conv2d(input,weight,strides=[1,1,1,1],padding='SAME')
    return tf.nn.relu(conv+biases)  #修正线性

def conv_def():
    with tf.variable_scope('conv1'):
        #创建变量'conv1/weights','conv1/biases
        relu1 = conv_relu(input_images,[5,5,32,32],[32])
    with tf.variable_scope('conv2'):
        relu1 = conv_relu(relu1,[5,5,32,32],[32])

def placeholder_t():
    x = tf.placeholder(tf.float32,shape=(3,2))
    y = tf.reshape(x,[2,3])
    z = tf.matmul(x,y)
    print(z)
    with tf.Session() as sess:
        rand_array_x = np.random.rand(3,2)
        rand_array_y = np.random.rand(2,3)
        print(sess.run(z,feed_dict={x:rand_array_x,y:rand_array_y}))

#数据流图可视化
def visible_graph():
    #定义算子及算子名称
    a = tf.constant(2,name='input_a')
    b = tf.constant(4,name='input_b')
    c = tf.multiply(a,b,name='mul_c')
    d = tf.add(a,b,name='add_d')
    e = tf.add(c,d,name='add_e')
    sess = tf.Session()
    output = sess.run(e)
    print(output)
    #将数据流图写进log文件
    # writer = tf.summary.FileWriter('/home/nvidia/temp_project/result',sess.graph)
    writer = tf.summary.FileWriter(r'G:\20190426\zhouweiwei\mygit\blogApp\data_analysis\deep_learning\result',sess.graph)
    writer.close()#关闭写
    sess.close() #关闭会话


'''
采用作用域来组织运算的封装，在图形展示中，点击右上方的图标，可以查看各个阶段的图形信息
'''
def scope_graph():
    grap = tf.Graph()
    with grap.as_default():
        in_1 = tf.placeholder(tf.float32,shape=[],name='input_a')
        in_2 = tf.placeholder(tf.float32,shape=[],name='input_b')
        const = tf.constant(3,dtype=tf.float32,name='static_value')
        with tf.name_scope('Transformation'): #采用作用域来组织运算封装
            with tf.name_scope('A'):
                A_mul = tf.multiply(in_1,const) #乘法
                A_out = tf.subtract(A_mul,in_1) #减法
                with tf.name_scope('B'):
                    B_mul = tf.multiply(in_2,const)
                    B_out = tf.subtract(B_mul,in_2)
                with tf.name_scope('C'):
                    c_div = tf.div(A_out,B_out)  #除法
                    c_out = tf.add(c_div,const)
                with tf.name_scope('D'):
                    D_div = tf.div(B_out,A_out)
                    D_out = tf.add(D_div,const)
                    out = tf.maximum(c_out,D_out)  #输出两个变量之间的最大的值
    writer = tf.summary.FileWriter(r'G:\20190426\zhouweiwei\mygit\blogApp\data_analysis\deep_learning\result',graph=grap)
    writer.close()

'''
分布式：大幅度的提升性能，分布式并行处理
实现tensorflow分布式处理
'''
def tensorflow_dis():
    #cpu和两个Gpu
    #一版把参数存储及简单操作定义在cpu上，比较复杂操作定义在各自gpu上
    #生成输入数据
    N_GPU = 1
    train_x = np.random.rand(100).astype(np.float32)
    train_y = train_x*0.2+0.3
    #参数存储及简单操作放在cpu上
    with tf.device('/cpu:0'):
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)
        w = tf.Variable(0.0,name='weight')
        b = tf.Variable(0.0,name='reminder')
    #优化操作放在gpu上，采用异步更新的参数
    for i in range(N_GPU):  #这里可以实现多GPU执行，但是要确保你的机器上有相同个数的GPU，  这里的 GPU是处理设备，在tf进行处理的时候，会自动分配数据在各个gpu上运行
        with tf.device('/gpu:%d'%i):
            y = w*X+b
            loss = tf.reduce_mean(tf.square(y-Y))
            init_op = tf.global_variables_initializer()  #全局变量初始化
            train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  #梯度下降优化
    #创建会话，训练模型
    with tf.Session() as sess:
        #初始化参数
        sess.run(init_op)
        for i in range(1000):
            sess.run(train_op,feed_dict={X:train_x,Y:train_y})
            if i%100 ==0:
                print(i,sess.run(w),sess.run(b))
        print(sess.run(w))
        print(sess.run(b))




if __name__=='__main__':
    tensorflow_dis()