#! /usr/bin/env python
# -*-coding:utf-8 -*-
# Author:zhouweiwei

import datetime

def date_transform():
    # date =datetime.date(2019,1,3)
    # print(str(date))
    # print(type(str(date)))
    # print(date.year)
    # time = datetime.time(9,20,34)
    # print(time)
    # print(datetime.datetime.now())
    # str_time = str(datetime.datetime.now().date())
    # print(type(str_time))
    # print(datetime.datetime.strptime(str_time,'%Y-%m-%d'))
    # print(datetime.datetime.now().strftime('%Y-%m-%d'))

    import pandas as pd
    # index = pd.date_range('2019-03-01','2019-04-01',freq='H')
    index = pd.date_range('2019-03-01',periods=3)
    print(index.shift(2))


if __name__=='__main__':
    date_transform()