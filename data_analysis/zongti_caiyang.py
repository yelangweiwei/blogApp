#采样偏差影响对总体的预测，可以减小采样误差，但是避免不了采样误差，
#标准差，用来表征数据的分散程度   样本-均值的平方和的均值的开方

#置信区间：用总体统计量的估计区间话标准差，用于表征数据的波动范围
#抽样方法   1）蒙特卡洛模拟
import seaborn
import numpy as np
#boostrap采样的实现
def boostrap(data,num_sample,statistics,alpha):
    n = len(data)
    idx = np.random.randint(0,n,size=(num_sample,n))
    samples = data[idx]
    stat = np.sort(statistics(samples,1))
    return (stat[int((alpha/2)*num_sample)],stat[int((1-alpha/2)*num_sample)])

if __name__ == '__main__':
    data = seaborn.load_dataset('iris')
    data1,data2 = boostrap(data, 2, data, 1)
    print(data1)
    print(data2)






