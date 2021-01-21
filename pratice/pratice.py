#! /usr/bin/env python
# -*-coding:utf-8 -*-
# Author:zhouweiwei
#生成器的使用，生成器每次产生的值都进行保存，每次调用都从上次继续执行
def odds(start=1):
    if start%2==0:
        start+=1
    while True:
        yield start
        start+=2
import json
def save_chinese():
    a = json.dumps({'name': '张三'}, ensure_ascii=False)
    print(a)

def combine_txt():
    a = r'G:\20190426\zhouweiwei\mygit\blogApp\pratice\txt_dir\a.txt'
    b = r'G:\20190426\zhouweiwei\mygit\blogApp\pratice\txt_dir\b.txt'
    c = r'G:\20190426\zhouweiwei\mygit\blogApp\pratice\txt_dir\c.txt'
    with open(a,'r') as rh:
        a_content = rh.readline()

    with open(b,'r') as rh:
        b_content = rh.readline()

    c_content = a_content+b_content
    print(c_content)
    c_result = sorted(c_content.strip())
    print(c_result)

    with open(c,'a+') as wh:
        wh.write(''.join(c_result))

import datetime
def get_n_date():
    '''
    获得n天后的日期
    :return:
    '''
    now = datetime.datetime.now()
    print(now)
    new_date = now+datetime.timedelta(days=3)
    print(new_date.strftime('%Y%M%d'))

#闭包函数
def mul_opera(n):
    def get_data(val):
        return val*n
    return get_data

#输出所有的偶数
def get_even():
    # a_data = [x for x in range(1,101)]
    # print(a_data)
    # a_even = [y for y in a_data if y%2==0]
    # print(a_even)
    print(list(range(2,101,2)))


if __name__=='__main__':
    get_even()

