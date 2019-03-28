#采样偏差影响对总体的预测，可以减小采样误差，但是避免不了采样误差，
#标准差，用来表征数据的分散程度   样本-均值的平方和的均值的开方

#置信区间：用总体统计量的估计区间话标准差，用于表征数据的波动范围
#抽样方法   1）蒙特卡洛模拟
import seaborn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#boostrap采样的实现
def boostrap_sample():
    data = seaborn.load_dataset('iris')
    # seaborn.distplot(data['sepal_length'],color='gray')

    boost_sample = []
    for i in range(1000):
        boost_sample.append([])
    boost_mean = []
    for i in range(1000):
        boost_sample[i].append(pd.Series.sample(data['sepal_length'],replace=True))
        sample_data = pd.Series.sample(data['sepal_length'],replace=True)
        boost_mean.append(np.mean(sample_data))
    seaborn.distplot(boost_mean,color='gray')
    #求置信区间
    print(boost_mean)
    start_percentile = np.percentile(boost_mean,2.5)
    end_percentile = np.percentile(boost_mean,97.5)
    print([start_percentile,end_percentile])
    plt.show()



if __name__ == '__main__':
    boostrap_sample()




