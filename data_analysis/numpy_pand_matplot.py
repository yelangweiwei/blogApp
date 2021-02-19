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





if __name__=='__main__':
    hist_pra()




