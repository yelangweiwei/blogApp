'''
这种聚类的问题，一种是不能什么都要进行可视化，特征的选择要还是要通过计算获得特征。？？？？？？？？？
EM聚类，也叫最大期望算法
三个主要的步骤，初始化参数，观察预期，重新估计；前两个是期望步骤，最后一个是最大化步骤。
'''
import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns
#高斯混合模型
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def get_data(data_path):
    data_content = pd.read_csv(data_path,encoding='GBK')
    #将获得数据
    print(type(data_content))
    print(data_content.head(5))
    #打印dataframe的列
    print(data_content.columns)
    #通过查看前5条数据，获得数据的features
    features = list(data_content.columns)
    features.remove('英雄')
    features.remove('主要定位')
    features.remove('次要定位 ')
    print(features)

    '''通过可视化，查看各个特征之间的相关性，并在相关的特性之间选择一个，实现降维'''
    #设置正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  #用来显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号
    #使用热力图显示各个变量之间的相关性
    data = data_content[features]
    data = data[features]
    data['最大攻速'] = data['最大攻速'].apply(lambda x: float(x.strip('%')) / 100)
    data['攻击范围'] = data['攻击范围'].map({'远程': 1, '近战': 0})
    corr = data.corr()
    plt.figure(figsize=(14,14))
    sns.heatmap(corr,annot=True)
    plt.show()

    #特征选择
    #通过热力图显示各个变量之间的相关性；通过热力图,将相关性比较大的特征去掉,这个通过查看视图去掉相似特征，可以用过计算获得结果
    features.remove('生命成长')
    features.remove('法力成长')
    features.remove('初始法力')
    features.remove('物攻成长')
    features.remove('物防成长')
    features.remove('每5秒回血成长')
    features.remove('最大每5秒回血')
    features.remove('每5秒回蓝成长')

    '''数据清洗'''
    data= data[features]

    '''数据标准换'''
    ss = StandardScaler()
    data = ss.fit_transform(data)
    '''构造GMM聚类'''
    gmm = GaussianMixture(n_components=8,covariance_type='full',random_state=0)
    prediction = gmm.fit_predict(data)
    print(prediction)

    #聚类结果进行评估,这个值越大，证明聚类结果越好
    from sklearn.metrics import calinski_harabasz_score
    print(calinski_harabasz_score(data,prediction))



    #将聚类的结果写进csv文件
    # data_content.insert(0,'分组',prediction)
    # save_path = 'G:\\20190426\\zhouweiwei\\mygit\\blogApp\\data_analysis\\data\\EM\\result\\emResult.csv'
    # data_content.to_csv(save_path)












def hero_em_t():
    #读取数据
    data_path = os.path.dirname(os.path.realpath(__file__))+'/data/EM/EM_data/heros.csv'
    get_data(data_path)




if __name__=='__main__':
    hero_em_t()