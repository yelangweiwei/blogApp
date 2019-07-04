#! /usr/bin/env python
# -*-coding:utf-8 -*-
# Author:zhouweiwei
import requests
from selenium import webdriver
from lxml import etree

def get_data_by_xpath():
    query = '王祖贤'
    url_request = 'https://www.douban.com/j/search_photo?q='+query+'&limit=20&start='
    for i in range(25036):
        url = url_request+str(i)
        #模拟浏览器
        driver = webdriver.Chrome()
        #获得数据
        content = driver.get(url)
        html = etree.HTML(content)
        srcs  = html.xpath(src_path)
        titles = html.xpath(title_path)
        for src,title in zip(srcs,titles):
            content = requests.get(src,timeout=10)
            save_path = 'F:\\blog\\blogApp\\爬虫\\xpath_data\\'+title+'.jpg'
            with open(save_path,'wb') as wh:
                wh.write(content)




