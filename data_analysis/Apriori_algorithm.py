'''
重要概念：
支持度：指的是某个商品出现的次数与总次数之间的比例，支持度越高，代表这个组合出现的频率越大
置信度：指的是在A发生的情况下，B发生的概率；是个条件概率
提升度：代表的是：商品A的出现，对商品B出现概率的提升的程度;提升度（a->b） = 置信度（a-b）/支持度(b);用这个公式来衡量A出现的情况下，时候对b出现的
概率有所提升；
1）提升度（a-b）>1:有提升
2）提升度（a->b）=1;没有提升，也没有下降
3）提升度（a->b）<1;代表有下降
apriori算法的原理
1）实际就是查找频繁项集的过程：就是支持度大于等于最小支持度阈值的项集，所以小于最小支持度的项目就是非频繁项集，而大于等于最小支持度的项集就是频发项集
2）项集：可以是单个商品，也可以是商品组合
流程：
1）k=1;计算K项集的支持度
2）筛掉小于最小支持度的项集
3）如果项集为空；则对应k-1项集的结果为最终结果；否则k= k+1,重复1~3步

apriori:的改进算法：FP-Growth算法
apriori的缺点：
1）可能产生大量的候选集，因为采用排列组合的方式，把可能的项集都组合出来了
2）每次计算都需要重新扫描数据集，来计算每个项集的支持度。

FP-Growth (频繁模式树)特点:
1)创建一棵FP 树来存储频繁项集，在创建之前对不满足最小支持度的项进行删除，减少了存储空间。
2）整个生成过程只遍历2次，大大减少了计算量。
实际应用中，比较常用
'''

from efficient_apriori import apriori

def ariori_t():
    # 设置数据集
    # data = [('牛奶', '面包', '尿布'),
    #         ('可乐', '面包', '尿布', '啤酒'),
    #         ('牛奶', '尿布', '啤酒', '鸡蛋'),
    #         ('面包', '牛奶', '尿布', '啤酒'),
    #         ('面包', '牛奶', '尿布', '可乐')]
    data = [['牛奶', '面包', '尿布'],
            ['可乐', '面包', '尿布', '啤酒'],
            ['牛奶', '尿布', '啤酒', '鸡蛋'],
            ['面包', '牛奶', '尿布', '啤酒'],
            ['面包', '牛奶', '尿布', '可乐']]

    # 挖掘频繁项集和频繁规则
    itemsets, rules = apriori(data, min_support=0.5, min_confidence=0.5)
    print(itemsets)
    print(rules)



#从网页下载数据并进行关联分
import time,os
import csv
from selenium import webdriver
from lxml import etree

#指定下载页面的数据
def download(request_url,dirver,director,flags,csv_writer):
    dirver.get(request_url)
    time.sleep(1)
    html = dirver.find_element_by_xpath("//*").get_attribute("outerHTML")
    html = etree.HTML(html)
    #设置电影名称，导演演员的xpath
    movie_lists = html.xpath("/html/body/div[@id='wrapper']/div[@id='root']/div[1]//div[@class='item-root']/div[@class='detail']/div[@class='title']/a[@class='title-text']")
    name_lists = html.xpath("/html/body/div[@id='wrapper']/div[@id='root']/div[1]//div[@class='item-root']/div[@class='detail']/div[@class='meta abstract_2']")
    #获得返回的数据个数
    num  = len(movie_lists)
    if num>15:#第一页有16条数据
        movie_lists = movie_lists[1:]
        name_lists = name_lists[1:]
    for(movie,name_list) in zip(movie_lists,name_lists):
        #会存在数据为空的情况
        if name_list.text is None:
            continue
        #显示演员的名称
        print(name_list.text)
        names = name_list.text.split('/')
        #判断导演是否为指定的director
        if names[0].strip() == director and movie.text not in flags:
            #将第一个字段设置为电影名称
            names[0] = movie.text
            flags.append(movie.text)
            csv_writer.writerow(names)
    print('OK') #代表这也数据下载成功
    print(num)
    if num >=14: #有一页可能会有14个电影
        #继续下一页
        return True
    else:
        #没有下一页
        return False


from selenium.webdriver.chrome.options import Options
def get_web_pagea():
    #模拟浏览器,在模拟浏览器的时候，这里的浏览器要指定浏览器的exe地址
    chrom_exe = 'G:\\20190426\\zhouweiwei\\mygit\\blogApp\\data_analysis\\data\\apriori\\browser\\chromedriver.exe'
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-setuid-sandbox")
    dirver = webdriver.Chrome(executable_path=chrom_exe)
    director = u'张艺谋'
    file_name = os.path.dirname(os.path.realpath(__file__))+'/data/apriori/'+director+'.csv'
    base_url = 'https://movie.douban.com/subject_search?search_text='+director+'&cat=1002&start='
    out = open(file_name,'w',newline='',encoding='utf-8-sig')  #为了防止中文编码混乱
    csv_writer = csv.writer(out,dialect='excel')

    flags = []
    #开始的ID为0，每页增加15
    start = 0
    while start<10000:
        request_url = base_url+str(start)
        flag = download(request_url, dirver, director, flags, csv_writer)
        if flag:
            start= start+15
        else:
            break
    out.close()
    print('finished')

#将获得的数据进行分析
def analysis_actor_by_ariori():
    director = u'张艺谋'
    file_name= os.path.dirname(os.path.realpath(__file__))+'/data/apriori/'+director+'.csv'
    #加载数据G:\20190426\zhouweiwei\mygit\blogApp\data_analysis\data\apriori\宁浩.csv
    lists = csv.reader(open(file_name,'r',encoding='utf-8-sig'))
    name_list = []
    for names in lists:
        name_new = []
        for name in names:
            name_new.append(name.strip())
        print(list(name_new[1:]))
        name_list.append(name_new[1:])

    #使用挖掘规则进行关联分析
    itemsets,rules = apriori(name_list,min_support=0.05,min_confidence=1)
    # itemsets,rules = apriori(name_list,min_support=0.1,min_confidence=1)
    print(itemsets)
    print(rules)

if __name__=='__main__':
    # ariori_t()
    # get_web_pagea()
    analysis_actor_by_ariori()






































