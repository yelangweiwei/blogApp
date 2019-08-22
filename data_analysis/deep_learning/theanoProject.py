import theano
import numpy as np
from theano import tensor as T

if __name__ == '__main__':

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
    data= np.array([[1,2],[3,4]])
    shared_type = theano.shared(data)
    print(type(shared_type))





    #初始化张量
    # x =T.scalar(name='input',dtype='float32')
    # w = T.scalar(name='weight',dtype='float32')
    # b = T.scalar(name='bias',dtype='float32')
    # z = w*x+b
    # #编译程序
    # net_input = theano.function(inputs=[w,x,b],outputs=z)
    # #执行程序
    # print('net_input:%2f'% net_input(2.0,3.0,4.0))




