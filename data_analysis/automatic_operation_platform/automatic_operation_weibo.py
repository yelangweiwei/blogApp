'''
1,掌握selenium自动化测试工具，以及元素的定位的方法
2，学会编写微博自动化功能模块，加关注，写评论，发微博
3，对微博自动化做自我总结
'''
from  selenium import webdriver
import time
chrom_exe = 'G:\\20190426\\zhouweiwei\\mygit\\blogApp\\data_analysis\\data\\apriori\\browser\\chromedriver.exe'
brower = webdriver.Chrome(executable_path=chrom_exe)

#微博登录
def weibo_login(username,passwd):
    #打开微博的首页，还没有登录之前的页面
    brower.get('https://weibo.com')
    brower.implicitly_wait(5)

    #在打开登录页面中输入用户名和密码进行登录
    brower.find_element_by_id('loginname').clear()
    time.sleep(1)
    brower.find_element_by_id('loginname').send_keys(username)
    time.sleep(1)
    brower.find_element_by_css_selector("[name='password']").clear()
    time.sleep(1)
    brower.find_element_by_css_selector("[name='password']").send_keys(passwd)
    time.sleep(1)
    #点击登录
    brower.find_element_by_css_selector("[node-type='submitBtn']").click()
    time.sleep(1)



#添加指定用户
def add_follow(uid):
    brower.get('https://m.weibo.com/u/'+str(uid))
    time.sleep(1)
    follow_button = brower.find_element_by_xpath('//div[@class="m-add-box m-followBtn m-btn m-btn-block m-btn-blue"]')
    follow_button.click()
    time.sleep(1)
    #选择分组,这里没有分组
    group_button = brower.find_element_by_xpath('//div[@class="m-btn m-btn-white m-btn-text-black"]')
    group_button.click()
    time.sleep(1)

#给指定的某条微博添加内容,这里查找是网页版的
def add_comment(weibo_url,content):
    brower.get(weibo_url)
    brower.implicitly_wait(5)
    comment_content = brower.find_element_by_css_selector('textarea.W_input').clear()
    comment_content = brower.find_element_by_css_selector('textarea.W_input').send_keys(content)
    time.sleep(2)
    comment_button = brower.find_element_by_css_selector('.W_btn_a').click()
    time.sleep(1)

#发文字微博
def post_weibo(content):
    #跳转到用户的首页
    brower.get('https://weibo.com')
    brower.implicitly_wait(5)
    #点击右上角的发布按钮
    post_button = brower.find_element_by_css_selector("[node-type='publish']").click()
    #添加要发表的内容
    content = brower.find_element_by_css_selector('textarea.W_input').send_keys(content)
    time.sleep(2)
    #点击进行提交
    post_button = brower.find_element_by_css_selector("[node-type='submit']").click()
    time.sleep(1)


if __name__=='__main__':
    #网页版微博登录
    username = '18330239296'
    passwd = '1991321456asdf'
    weibo_login(username=username,passwd=passwd)

    #搜索微博用户

    # #每天学点心理学
    # uid = '1890826225'
    # # add_follow(uid)
    #
    # #添加评论
    # weibo_url = 'https://weibo.com/1890826225/HjjqSahwl'
    # content = 'Good Luck! 好运以上路'
    # add_comment(weibo_url, content)

    #给指定的微博写评论
    # content = '李彦宏 百度 AI'
    # post_weibo(content)





