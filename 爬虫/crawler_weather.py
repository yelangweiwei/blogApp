#! /usr/bin/env python
# -*-coding:utf-8 -*-
# Author:zhouweiwei

'''
通过网页爬取和api爬取数据，将快剪辑中的内容进行爬取，
'''

import urllib.request as urlrequest
# import urllib3.request as urlrequest
import numpy as np
import sys
import json
import time
import logging
import http.cookiejar
from bs4 import BeautifulSoup

def get_vidio_address(web_address):

    #模拟浏览器
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}
    request = urlrequest.Request(headers=headers,url=web_address)

    #添加cookie处理对象
    cookie_support = urlrequest.HTTPCookieProcessor(http.cookiejar.CookieJar())
    opener = urlrequest.build_opener(cookie_support)
    #将opener设置为全局变量
    urlrequest.install_opener(opener)
    #读取数据
    try:
        # 读取网页内容
        page_content = urlrequest.urlopen(request).read().decode('utf8')

        #将要打印的内容使用beautifulsoup进行解析
        bfs = BeautifulSoup(page_content,'html.parser')
        print(bfs)

        #将读取的页面的内容，使用json 进行转换

    except:
        print('read page error')
        return 0








if __name__=='__main__':
    #网页的地址
    web_address= 'http://kuai.360.cn'
    get_vidio_address(web_address)