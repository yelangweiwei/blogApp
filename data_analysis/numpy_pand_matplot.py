'''
matplot numpy pandas
'''
import os
import random
import matplotlib
import numpy as np
from  matplotlib import pyplot as plt
from matplotlib import font_manager

#fc-list 在linux上使用的，可以查看系统中支持的中文字体
#windows和linux设置字体的方式
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size':12
        }
matplotlib.rc('font',**font)


# myfont= font_manager.FontProperties(fname=r'C:\Windows\Fonts\微软雅黑')


'''

'''
def matplt():
    fig = plt.figure(figsize=(12,10),dpi=80)  #dpi:每英寸点的个数 figsize:宽和高

    y = [random.randint(20,35) for i in range(120)]
    y1 = [random.randint(15,25) for i in range(120)]
    x = range(0,120,1)
    plt.plot(x,y,label='yestoday',color='r',linestyle='--',linewidth=2)
    plt.plot(x,y1,label='today',color='orange',linestyle='-.',linewidth=3)

    #将x轴显示时间字符串
    _xtick_labels = ['10点{}分'.format(i) for i in range(60)]
    _xtick_labels += ['11点{}分'.format(i) for i in range(60)]
    plt.xticks(x[::3],_xtick_labels[::3],rotation=45)  #旋转90度

    plt.yticks(range(min(y),max(y)))
    # 添加描述信息
    plt.title('室外温度趋势图')
    plt.xlabel('时间')
    plt.ylabel('温度 单位(℃)')
    #画网格 alpha 控制透明度
    plt.grid(alpha=0.5,linestyle='-')
    plt.legend(loc='best')

    plt.show()
    #图片的保存
    save_path = os.path.dirname(os.path.realpath(__file__))+'/data/fig/'
    #可以将图片保存为.svg的格式，这样就可以保证图片避免失真
    # plt.savefig(save_path+'temp.svg')


'''
折线图：表示统计数量的增减变化的统计图，反应事物的变化情况
直方图：绘制连续的数据，展示一组或者多组数据的分布情况  一个范围内的值有多少
条形图：绘制离散的数据，能够一眼看出各个数据的大小，比较数据之间的差别
散点图：判断变量之间是否存在数量关联趋势，展示离群点
'''
'''
绘制散点图 
不同条件，维度之间的内在的关联关系
'''
def sca_plt():
    #3月
    ya = [2,2,3,5,6,8,4,7,9,3,8,9,3,7,4,7,3,6,2,2,3,5,6,8,4,7,9,3,8,9,3]
    #10月
    yb = [7,8,5,4,7,8,9,7,5,7,7,8,9,7,5,4,7,9,7,8,5,4,7,8,9,7,5,7,7,8,9]
    fig = plt.figure(figsize=(35,15),dpi=80)
    xa = list(range(1,32,1))
    xb = list(range(35,66,1))
    plt.scatter(xa,ya,label='3 month',linewidths=3)
    plt.scatter(xb,yb,label='10 month',linewidths=1)
    #调整x轴刻度
    _x = list(xa)+list(xb)
    _xtick_labels = ['3 month {} day'.format(i) for i in xa]
    _xtick_labels +=['10 month {} day'.format(i-34) for i in xb]
    plt.xticks(_x[::3],_xtick_labels[::3],rotation=45)
    plt.xlabel('month')
    plt.ylabel('温度℃')
    plt.legend(loc='best')
    plt.show()

'''
条形图:
绘制横着的条形图
频数统计，数量统计
'''

def bar_pra():
    movie_name_list = ['你好，李焕英','唐人街探案3','刺杀小说家','熊出没,狂野大陆\n','人潮汹涌','新神榜：哪吒重生','侍神令']
    money_list = [20,16,12,19,14,14,0.3]
    money_list1 = [24,20,18,14,17,13,0.8]
    money_list2 = [20,13,16,18,15,12,1]
    figure = plt.figure(figsize=(20,8),dpi=80)
    plt.barh(range(len(movie_name_list)),money_list,height=0.2,color='red',label='first_day')
    plt.barh([x+0.2 for x in range(len(movie_name_list))],money_list1,height=0.2,color='blue',label='second_day')
    plt.barh([x+0.4 for x in range(len(movie_name_list))],money_list2,height=0.2,color='yellow',label='third_day')

    plt.yticks([x+0.2 for x in range(len(movie_name_list))],movie_name_list,rotation=45)
    plt.xlabel('movie score')
    plt.ylabel('movie name')
    plt.legend(loc='best')
    plt.title('movie compare')
    plt.show()


'''
直方图
    使用的是没有进行统计过的数据，可以使用直方图进行统计，统计后的直方图不可以在进行统计绘制
    年龄的分布状态
    一段时间内用户点击次数的分布状态
    用户活跃时间的分布状态
'''
def hist_pra():
    a = [random.randint(90,180) for x in range(250)]
    print(len(a))
    fig = plt.figure(figsize=(20,8),dpi=80)
    bin_width = 10
    bin_num = (max(a)-min(a))//bin_width
    plt.hist(a,bins=bin_num)  #后边的bins可以是分的组的个数，也可以是list，其中的值代表了每个hist的宽度
    # plt.hist(a,bins=np.array([91,102,138,152,180]))  #后边的bins可以是分的组的个数，也可以是list，其中的值代表了直方图的开始和结束位置
    plt.xlabel('movie time length')
    plt.ylabel('movie num')
    plt.xticks(range(min(a),max(a)+bin_width,bin_width),rotation=45)
    plt.title('hist movie score')
    plt.grid()
    plt.show()

'''
numpy:
    数组和数字进行计算，是可以计算的
    数组和数组 列数有一样或者行数一样（单行或者单列），活着行数和列数都一样，或者后者行，列小的数组是大的数组的一个子模块，是可以计算的
    广播原则：如果两个数组的后援维度，即从尾部开始算起的维度的轴长度相符或其中一方的长度为1，则认为是广播兼容的，广播会在缺失或者长度为1的维度上进行。
    
'''
def np_pra():
    # t1 = np.arange(12)
    # print(t1.shape)
    # t2 = np.array([[1,2,3],[3,4,5]])
    # print(t2.shape)
    # t3 = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
    # print(t3)
    # print(t3.shape)
    # t4 = np.arange(12)
    # print(t4)
    # print(t4.reshape(3,4))
    #
    # t5 = np.arange(24).reshape((2,3,4))
    # print(t5.reshape(4,6))
    # print(t5.reshape(24,))
    # print(t5.reshape(1,24))
    # print(t5.flatten())  #将多维数组展开
    # print(t5+2)  #把计算应用到数组中的每一个数字上，分别进行计算
    # print(t5/0)  #会出现警告，但是不会出错
    # t5 = np.arange(24).reshape(4,6)
    # print(t5)
    # t6 = np.arange(100,124).reshape(4,6) #对应位置的值分别进行计算
    # print(t5+t6)
    # t7 = np.array([0,1,2,3.,4,5])
    # print(t5-t7)
    # t8 = np.arange(4).reshape((4,1))
    # print(t8)
    # print(t5+t8)

    # t9 = np.array([0,1,2,3,4,5,6,7,8,9])  #形状不一致的时候，不能进行计算
    # print(t9+t5)

    # a = np.array([1,2,3,4,5,6])
    # print(type(a[0]))
    # print(a)
    #
    # t4 = np.array(range(1,4),dtype=float)
    # print(t4)
    # print(t4.dtype)

    # t5 = np.array([1,1,1,0,0,1],dtype=bool)
    # print(t5)
    # print(t5.dtype)
    #
    # #调整数据类型
    # print(t5.astype('int8'))
    #
    t6 = np.array([random.random() for i in range(10)],dtype=float)
    print(t6)
    print(t6.dtype)
    print(np.round(t6,2))

    #转置
    t = np.arange(24).reshape(4, 6)
    print(t)
    print(t.transpose())
    print(t.T)
    print(t.swapaxes(0, 1))


''''
一般不使用numpy读数据，但是这个也有读数据的方法
从csv中读取数据
索引和切片操作，只想要其中的一行数据
'''
def num_read_data():
    fname =os.path.dirname(os.path.realpath(__file__))+'/data/csvdata/temp_data.csv'
    # data1 = np.loadtxt(fname,dtype="int",delimiter=',',skiprows=0,usecols=None,unpack=True)
    #skiprows:跳过行数 usercols：读取指定的列，unpack:为True,将原来的值进行转置，默认的情况是有多少行就会展示多少行，转置后，行和列就进行转置了
    # print(data1)
    data = np.loadtxt(fname,dtype="int",delimiter=',',skiprows=0,usecols=None,unpack=False) #skiprows:跳过行数 usercols：读取指定的列，unpack:为True，写入不同的属性组
    print(data)

    # print(data.shape)
    # #取行
    # # print(data[1])
    # #取不连续的多行
    # print(data[[2,3,4]])
    # #取不连续的多列
    # print(data[:,[1,3]])
    # #取连续的多列
    # print(data[:,1:])
    # #取多行，多列
    # print(data[2:,2:])
    # #取多个不相邻的点
    # print(data[[0,2],[0,1]])

    #对对应的位置的值的修改
    # data[2,3]=10
    # print(data)
    #布尔索引
    # print(data<10)
    # print(data[data>5])  #对为true的位置的值进行取值
    #替换where
    # result=np.where(data<5,1,20)
    # print(result)
    #clip 剪裁,将小于3的替换为3，将大于10的替换为10，处在3和5之间的不变
    print(data.clip(3,5))

'''
数组的拼接
随机方法
'''
def numpy_combine():
    # fname = os.path.dirname(os.path.realpath(__file__)) + '/data/csvdata/temp_data.csv'
    # data = np.loadtxt(fname,dtype="int",delimiter=',',skiprows=0,usecols=None,unpack=True)
    # print(data)
    # data1 = data
    #竖直拼接和分割是对应的，水平方向截取
    # print(np.vstack((data,data1)))
    #水平拼接，水平方向截取
    # print(np.hstack((data,data1)))

    #列交换
    # data[:,[1,2]] = data[:,[2,1]]
    # print(data)
    # #行交换
    # data[[1,2],:]=data[[2,1],:]
    # print(data)

    #构造全为0的数组
    # print(np.zeros((3,4)))
    # print(np.ones((3,4)))
    # print(np.eye(3))
    # print(data.shape)
    # #返回最大值的位置
    # print(np.argmax(data,axis=0)) #列
    # print(np.argmax(data,axis=1)) #行
    # print(np.argmin(data,axis=0)) #列
    # print(np.argmin(data,axis=1)) #行
    #
    # #生成随机数
    # print(np.random.rand(1,2,3))  #均匀随机分布
    # print(np.random.seed(4)) #这个控制紧挨着的随机值，得到的值是一样的，在后边有一个就不在起作用了，比如1）起作用，但是2）就不起作用了
    # print(np.random.randint(10,20,(3,4)))  #标准正态分布1）
    # print(np.random.randint(10,20,(3,4)))  #2）

    #copy view  赋值和视图、
    # b = data
    # a = b[:] #视图的操作，一种切片，会创建新的对象a,但是a的数据完全有b保管，他们两个的数据变化是一致的
    # a = b # 完全不复制，a和b相互影响
    # a = b.copy()  #复制，a和b互不影响
    #
    # #nan 表示缺失  inf表示无穷  nan不是一个数字
    # print(np.nan is np.nan)
    # #返回非0元素的行索引和列索引，行索引和列索引分开写
    # print(np.nonzero(data))
    # #计算nan的个数,返回非0元素的个数
    # print(np.count_nonzero(data))
    # #判断各个值是不是nan
    # print(np.isnan(data))
    # #求和
    # print(np.sum(data)) #求所有数据的和
    # print(np.sum(data,axis=0))  #按照列求和
    # print(np.sum(data,axis=1))  #按照行求和

    # print(data)
    # 计算每一列的平均值
    # print(np.average(data,axis=0))
    # print(data.mean(axis=0))
    # print(data.max(axis=0))
    # print(data.min(axis=0))
    # print(data.ptp(axis=0)) #极值
    # print(data.std(axis=0)) #标准差

    #填充nan
    t  = np.arange(12).reshape(3,4).astype('float')
    t[1,2:]=np.nan
    print(t)
    for i in range(t.shape[1]):
        temp_code = t[:,i]
        if np.count_nonzero(temp_code!=temp_code)==0: #非0的个数为0，没有nan
            continue
        else:
            temp_not_nan_col = temp_code[temp_code==temp_code]  #将不是nan的值提取出来
            print(temp_not_nan_col)
            temp_code[np.isnan(temp_code)]= temp_not_nan_col.mean()
    print(t)



'''
pandas
    处理非数值型的数据
    series：一维带标签的数组
    dataframe:二维的series数组
    
'''
import pandas as pd
import string
from pymongo import MongoClient
def pd_pra():
    # t = pd.Series(np.arange(10),index=list(string.ascii_uppercase[:10]))
    # print(t)
    #
    # t1 = pd.Series(np.arange(10),index = list('adcdefghij'))
    # print(t1)
    #
    # task_dict = {'name':'xiaoming','age':30,'sex':0,'tel':18330239298}
    # t2 = pd.Series(task_dict)
    # print(t2)
    # print(t2.dtype)
    #
    # #series 的切片和索引
    # # print(t2['name'])
    # # print(t2[0])
    # # print(t2[[0,2,3]])
    # # print(t2[['age','sex','name']])
    # # print(t2[['tel']])
    # #布尔索引
    # # print(t1[t1<4])
    #
    # #属性
    # print(list(t2.index))
    # print(type(t2.index))
    # print(list(t2.values))
    # print(type(t2.values))
    # print(t1.where(t1>3))  #将<=3的变为0

    #读取外部数据
    # csv_path = os.path.dirname(os.path.realpath(__file__)) + '/data/csvdata/temp_data.csv'
    # data = pd.read_csv(csv_path)
    # print(data)

    #dataframe,
    t = pd.DataFrame(np.arange(12).reshape(3,4),)
    print(t.shape)
    print(type(t))
    print(t)







if __name__=='__main__':
    pd_pra()




