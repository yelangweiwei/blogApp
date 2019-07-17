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

if __name__=='__main__':
   for n in odds():
       if n>5:
            break
       else:
           print(n)
