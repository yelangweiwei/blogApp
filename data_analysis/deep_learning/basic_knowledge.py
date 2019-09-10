import os
import numpy as np
import time



if __name__=='__main__':
    pass

    #循环和向量化运算比较  SIMD指令
    # x1 = np.random.rand(1000000)
    # x2 = np.random.rand(1000000)

    #使用循环计算向量点积
    # tic= time.process_time()


    #数据和并和展平
    #一维合并数据
    # a = np.array([1,2,3,4,5])
    # b = np.array([2,3,4,5,6])
    # print(np.append(a,b))
    # print(np.concatenate([a,b]))

    #多维合并
    # a = np.arange(4).reshape([2,2])
    # print(a)
    # b = np.arange(4).reshape([2,2])
    # print(b)
    # print(np.append(a,b,axis=0))

    #矩阵展开
    # nd5 = np.arange(6).reshape(2,-1)
    # print(nd5)
    # #按照列优先，扁平化
    # print(nd5.ravel('F'))
    # #按照行优先,扁平化
    # print(nd5.ravel())

    #矩阵操作
    # nd14 = np.arange(9).reshape([3,3])
    # print(nd14)
    # # #矩阵转置
    # # print(np.transpose(nd14))
    # # #矩阵乘法运算
    # # nd15 = np.arange(12).reshape([3,4])
    # # print(nd14.dot(nd15))
    # # #矩阵的迹,对角线求和
    # # print(np.trace(nd14))
    #
    # #矩阵的行列式
    # print(np.linalg.det(nd14))
    # #矩阵的逆矩阵
    # c = np.random.random([3,3])
    # print(c)
    # print(np.eye(3))
    # print(np.linalg.solve(c,np.eye(3)))



    # #random.choice
    # a = np.arange(1,25,dtype=float)
    # # print(len(a))
    # # print(a)
    # c1 = np.random.choice(a,size=(3,4))  #size指输出的形状
    # print(c1)
    # c2 = np.random.choice(a,size=(3,4),replace=False)  #replace为True,可以重复抽取
    # print(c2)
    # c3 = np.random.choice(a,size=(3,4),p = a/np.sum(a))  #代表每个元素被抽取的概率
    # print(c3)



    #存取元素
    # np.random.seed(2018)
    # nd11 = np.random.random(10)
    # print(nd11)
    #
    # nd22 = np.arange(1,20,2)
    # print(nd22)
    # #截取部分元素
    # print(nd22[(nd22>10)&(nd22<17)])

    #arange
    # print(np.arange(0,10,2))

    #创建多种形态的多维数组
    # a = np.zeros([3,3])
    # print(a)
    # b = np.ones([3,3])
    # print(b)
    # d = np.eye(3)
    # print(d)
    # c = np.diag([1,2,3])
    # print(c)


    #利用random生成ndarray
    #random 生成0-1之间的随机数
    # np.random.seed(10)
    # a  = np.random.random(10)
    # print(a)
    # b = np.random.randn(10)
    # print(b)
    # c = np.random.uniform(0,10,10)
    # print(c)
    # d = np.random.normal(c)
    # print(d)

    # list1 = [1,2,3,4,65,3,2,5,3]
    # array1 = np.array(list1)
    # print(type(array1))
    # print(array1)
    #嵌套list转换为array
    # list2 = [[1,2,3,4],[3,4,5,6,7]]
    # array2 = np.array(list2)
    # print(type(array2))
    # print(array2.shape)
    # print(array2)