#! /usr/bin/env python
# -*-coding:utf-8 -*-
# Author:zhouweiwei

'''
kmeans：聚类算法：确定k类的中心，找到了.0这些k类的中心，也就完成了聚类
'''
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
def k_means_t():
    data_path = os.path.dirname(os.path.realpath(__file__))+'/data/kmeans-master/data.csv'
    data = pd.read_csv(data_path,encoding='gbk')

    #进行数据探索
    print(data.columns)
    print(data.head(5))
    print(data.dtypes)
    #经过数据探索，发现其中的值变化比较大，进行数据标准化
    features = ['2019年国际排名','2018世界杯','2015亚洲杯']
    train_data = data[features]
    ss = MinMaxScaler()
    train_data = ss.fit_transform(train_data)
    #训练模型
    k_model  = KMeans(n_clusters=3)
    k_model.fit(train_data)
    predict_y = k_model.predict(train_data)
    #将聚类的结果合并到原始数据中
    result  = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
    result.rename({0:u'聚类'},axis=1,inplace=True)
    print(result)


'''使用k_means进行图像分割'''
import PIL.Image as images
import numpy as py
def read_pic(pic_path):
    #得到图片的像素
    f =open(pic_path,'rb')
    img = images.open(f)
    #得到图像的尺寸
    width,height = img.size
    data_list = []
    for x in range(width):
        for y in range(height):
            #得到点的三个通道
            c1,c2,c3 = img.getpixel((x,y))
            data_list.append([c1,c2,c3])
    f.close()
    #将数据进行标准化
    max_min=MinMaxScaler()
    data_list = max_min.fit_transform(data_list)
    print('data type:',type(data_list),'___data_list___:',data_list)
    print('data_type:',type(py.mat(data_list)),'___matrix__:',py.mat(data_list))
    return py.mat(data_list),width,height

def k_mean_2():
    pic_path = os.path.dirname(os.path.realpath(__file__)) + '/data/kmeans-master/'
    #先进行加载数据
    img,width,height = read_pic(pic_path+'weixin.jpg')
    #训练聚类.进行二分类
    k_means_model = KMeans(n_clusters=2)
    #这个结果是一维的
    k_means_model.fit(img)
    label_result = k_means_model.predict(img)
    #将聚类结果进行转换为图像尺寸的矩阵
    label_result = label_result.reshape([width,height])
    #创建新图像，用来保存聚类的结果，并设置不同的灰度值，因为获得的结果是0,1，这两个值不好看，就转换为颜色用255代表0,127代表1
    pic_mark = images.new("L",(width,height))
    for x in range(width):
        for y in range(height):
            pic_mark.putpixel((x,y),int(256/(label_result[x][y]+1))-1)
    pic_mark.save(pic_path+'pic_mark.jpg','JPEG')


def load_data(pic_path):
    path = pic_path+'weixin.jpg'
    f = open(path,'rb')
    img = images.open(f)
    #获得图形的宽，长
    weight,height = img.size
    #获得图形的像素
    data_list = []
    for i in range(weight):
        for j in range(height):
            c1,c2,c3 = img.getpixel((i,j))
            data_list.append([c1,c2,c3])
    f.close()

    #将通道信息标准化
    min_max = MinMaxScaler()
    data_list = min_max.fit_transform(data_list)
    return py.mat(data_list),weight,height



def load_data2(pic_path):
    path = pic_path+'weixin.jpg'
    f = open(path,'rb')
    img = images.open(f)
    #获得图形的宽，长
    weight,height = img.size
    #获得图形的像素
    data_list = []
    for i in range(weight):
        for j in range(height):
            c1,c2,c3 = img.getpixel((i,j))
            data_list.append([(c1+1)/256.0,(c2+1)/256.0,(c3+1)/256.0])
    f.close()

    return py.mat(data_list),weight,height


#将图像分类后的结果转换为颜色数值
from skimage import color
def kmeans_skimage_16():
    #加载数据
    pic_path = os.path.dirname(os.path.realpath(__file__)) + '/data/kmeans-master/'
    #获取图形的内容，宽和高
    imag,wight,height = load_data(pic_path)

    #进行聚类分类
    k_means_model = KMeans(n_clusters=16)
    lable_result = k_means_model.fit_predict(imag)
    #获得结果是一维的，要将这个数据转换为矩阵
    lable_result = lable_result.reshape([wight,height])
    #将结果转换为不同颜色的矩阵,这里是将label进行转换,每个RGB的数值都在0~255之间
    label_color = (color.label2rgb(lable_result)*255).astype(py.uint8)
    #数据需要进行转换，不然图形是倒置的
    label_color = label_color.transpose(1,0,2)
    #通过矩阵来生成图片
    image = images.fromarray(label_color)
    image.save(pic_path+'pic_label_16.jpg','JPEG')

def label_to_pic():
    # 加载数据
    pic_path = os.path.dirname(os.path.realpath(__file__)) + '/data/kmeans-master/'
    # 获取图形的内容，宽和高
    imag, wight, height = load_data2(pic_path)

    # 进行聚类分类
    k_means_model = KMeans(n_clusters=16)
    lable_result = k_means_model.fit_predict(imag)
    # 获得结果是一维的，要将这个数据转换为矩阵
    lable_result = lable_result.reshape([wight, height])

    save_image =images.new('RGB',(wight,height))
    #将结果展示在图形中,相当于原来的图形，使用质心中的值来替代。
    for i in range(wight):
        for j in range(height):
            print(k_means_model.cluster_centers_[lable_result[i,j]])
            c1 = k_means_model.cluster_centers_[lable_result[i,j],0]
            c2 = k_means_model.cluster_centers_[lable_result[i,j],1]
            c3 = k_means_model.cluster_centers_[lable_result[i,j],2]
            save_image.putpixel((i,j),(int(c1*256)-1,int(c2*256)-1,int(c3*256)-1))
    save_image.save(pic_path+'pic_mark_16.jpg','JPEG')









if __name__=='__main__':
    # k_mean_2()
    # kmeans_skimage_16()
    label_to_pic()


