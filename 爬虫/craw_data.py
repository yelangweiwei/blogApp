'''
自动化下载
爬虫实际上用浏览器访问的方式，模拟了访问网站的过程,打开网页，提取数据，保存数据
'''
import requests
import os
import json
from lxml import etree
from selenium import webdriver
def down_wangzuxian_pic():
    query= '王祖贤'
    for i in range(0,25133,20):
        url = 'https://www.douban.com/j/search_photo?q='+query+'&limit=20&start='+str(i)
        html= requests.get(url).text  #获得返回的结果
        response = json.loads(html,encoding='utf-8') #将json转换为python对象
        for image in response['images']:
            print(image["src"])
            #将图片下载到指定的位置
            dir = os.path.dirname(os.path.realpath(__file__))+'/craw_data_dir/'+str(image['id'])+'.jpg'
            pic = requests.get(image["src"],timeout=10)
            try:
                with open(dir,'wb') as wh:
                    wh.write(pic.content)
            except:
                print('cannot download the pic')

'''
当网页使用js请求数据，只有js加载完之后，才能获得完整的html文件，xpath可以不受限制的加载，帮我们定位想要的元素
'''
def down_by_xpath():
    query = '王祖贤'
    for i in range(0,6):
        url = 'https://movie.douban.com/subject_search?search_text='+query+'&cat=1002&start='+str(i*15)
        #模拟浏览器
        chrom_exe = 'G:\\20190426\\zhouweiwei\\mygit\\blogApp\\data_analysis\\data\\apriori\\browser\\chromedriver.exe'
        driver = webdriver.Chrome(executable_path=chrom_exe)
        driver.get(url)
        #解析html
        html = driver.find_element_by_xpath("//*").get_attribute("outerHTML")
        html =etree.HTML(html)
        src_xpath = "//div[@class='item-root']/a[@class='cover-link']/img[@class='cover']/@src"
        title_xpath = "//div[@class='item-root']/div[@class='detail']/div[@class='title']/a[@class='title-text']"
        srcs = html.xpath(src_xpath)
        titles = html.xpath(title_xpath)

        for src,title in zip(srcs,titles):
            dir = os.path.dirname(os.path.realpath(__file__)) + '/craw_data_dir1/' +title.text+ '.webp'

            pic = requests.get(src, timeout=10)
            try:
                with open(dir, 'wb') as wh:
                    wh.write(pic.content)
            except:
                print('cannot download the pic')






if __name__=='__main__':
    # down_wangzuxian_pic()
    down_by_xpath()







