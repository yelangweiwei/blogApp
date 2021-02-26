# -*-conding:utf-8-*-
# !/usr/bin/env

'''
通用爬虫
    通常指搜索引擎的爬虫
    工作原理：
        实现和百度新闻一样的网站，爬那些新闻数据，
        种子url地址是如何确定的

聚焦爬虫
    针对某类网站的爬虫

robots协议：那些内容可以抓取，那些内容不可以抓取
爬取的时候，先去请求这个网站是不是能爬
道德层次的要求，在技术上是可以实现的
'''

import requests
import os
import time


class TiebaSpider(object):
    def __init__(self, tieba_name):
        self.tieba_name = tieba_name
        self.url_temp = 'https://tieba.baidu.com/f?kw=' + tieba_name + '&ie=utf-8&pn={}'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36 SE 2.X MetaSr 1.0'}

    def get_url_list(self):
        return [self.url_temp.format(50 * i) for i in range(1000)]

    def pasr_url(self, url):
        reponse = requests.get(url, headers=self.headers)
        return reponse.content.decode('utf-8')

    def save_html(self, html_strm, page_num):
        path = os.path.dirname(os.path.realpath(__file__)) + '/data/{}-第{}页.html'.format(self.tieba_name, page_num)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_strm)

    def run(self):  # 实现主要逻辑
        # 1，构造url列表
        url_list = self.get_url_list()
        # 2，遍历发送请求，获取响应
        for url in url_list:
            time.sleep(2)
            content = self.pasr_url(url)
            # 3，保存
            page_num = url_list.index(url) + 1
            self.save_html(content, page_num)



def get_pra():
    tieba_name = '李毅'
    tieba_spider = TiebaSpider(tieba_name)
    tieba_spider.run()

'''
代理地址：
    
'''
def post_pra():

    #官方进行了防爬机制
    # post_data = {
    #     'from': 'zh',
    #     'to': 'en',
    #     'query': '人生苦短,我用python',
    #     # 'token': '1118c8eb90a65d1d166811d9bf260c0f',
    # }
    # headers = {"User-Agent":"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Mobile Safari/537.36"}
    # post_url = "https://fanyi.baidu.com/basetrans"
    # r = requests.post(post_url,headers=headers,data=post_data)
    # print(r.content.decode())


    #根据汉字和英文自动进行转换






if __name__ == '__main__':
    post_pra()