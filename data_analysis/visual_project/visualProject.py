'''
用户画像
话题的关键词
统计高频出现的词
目的：爬取数据，分析数据，实现可视化
'''
import matplotlib.pyplot as plt
import jieba
import numpy as np
import urllib
import json
import requests
from wordcloud import WordCloud
from PIL import Image
from lxml import etree

f = '数据分析全景图及修炼指南' \
        '学习数据挖掘的最佳学习路径是什么？' \
        'Python 基础语法：开始你的Python之旅' \
        'Python 科学计算：NumPy ' \
        'Python 科学计算：Pandas ' \
        '学习数据分析要掌握哪些基本概念？' \
        '用户画像：标签化就是数据的抽象能力 ' \
        '数据采集：如何自动采集数据？' \
        '数据采集：如何使用八爪鱼采集微博上的“D&G”评论？' \
        'Python爬虫：如何自动化下载王祖贤的海报？'

def remove_stop_words(f):
    stop_words = ['什么','就是','之旅','哪些','开始','最佳','使用','如何','下载','基础','王祖贤','路径','修炼']
    for word in stop_words:
        f = f.replace(word,'')
    return f

#生成词云
def create_word_cloud(f):
    print("根据词频计算词云")
    #使用全模式，将要分析的词，将句子中所有可以成词的词语都扫描出来。
    f = remove_stop_words(f)
    text = " ".join(jieba.cut(f,cut_all=False,HMM=True))
    #创建词云
    wc = WordCloud(
            #这个font_path要小写
            font_path='simhei.ttf',
            max_words=100,
            width= 2000,
            height=1200,
    )

    #使用这个方法，将扫描的文字生成词云
    wordcloud = wc.generate(text)
    #写词云图片
    wordcloud.to_file('wordcloud.jpg')
    #显示词云文件
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

headers = {
    'Referer':'https://music.163.com',
    'Host':'music.163.com',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
}

#爬取毛不易的歌
def get_names_and_ids(artist_id):
    #获取指定音乐人的页面内容
    url = 'https://music.163.com/#/artist?id='+str(artist_id)
    page_content = requests.request('GET',url,headers = headers)
    #将获得的内容，使用HTML进行解析
    html = etree.HTML(page_content)
    #使用xpath解析前50手歌,使用xpath解析元素和属性
    #打印指定路径下a标签的属性
    href_xpath = "//*[@id='hotsong-list']/a/@href"
    #获取a标签的内容
    title_xpath = "//*[@id='hotsong-list']/a/text()"
    href_list = html.xpath(href_xpath)
    title_list = html.xpath(title_xpath)
    print(list(href_list))
    print(list(href_list))
    id_list = []
    name_list = []
    for href,name in zip(href_list,title_list):
        id_list.append(href[9:])
        name_list.append(name)
    return name_list,id_list

#根据歌的名字和id读取每一首歌
def get_song_lyric(name,id):
    song_href =
    res = requests.request('GET',song_href,headers=headers)
    content = json.loads(res)
    print(content)



def maoYiCloud(artist_id):
    name_list, id_list = get_names_and_ids(artist_id)
    for name,id in zip(name_list,id_list):
        get_song_lyric()






if __name__=='__main__':

    #连续使用wordcloud
    # create_word_cloud(f)

    #毛不易的歌词制作词云
    artist_id = '12138269'
    maoYiCloud(artist_id)



