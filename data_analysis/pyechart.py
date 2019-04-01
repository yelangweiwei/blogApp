#! /usr/bin/env python
# -*-coding:utf-8 -*-
# Author:zhouweiwei

import numpy as py
import pandas as pd
import pyecharts
from pyecharts import Page
import matplotlib.pyplot as plt
def scatter_plot():
    page = Page()
    x = [10,20,30,40,50,60]
    y = [10,20,30,40,50,60]
    scatter = pyecharts.Scatter('散点图示例')
    scatter.add('A',x,y,is_visualmap=True,visual_type='size',visual_range=[10,60])
    page.add(scatter)
    page.render()

def line_graph():
    page = Page()
    years = [1950,1960,1970,1980,1990,2000,2010]
    gdb = [300.2,545.3,1075.9,2862.5,5979.6,10289.7,14958.3]
    line = pyecharts.Line('折线图')
    line.add("GDP",years,gdb,mark_point=['average'],is_symbol_show=False,is_smooth=False,is_stack=False,is_step=False,is_fill=True,area_color='#000',area_opacity=0.1)
    page.add(line)
    page.render()

def bar_graph():
    page = Page()
    data = [23,85,72,43,52]
    data1 = [25,75,82,60,82]
    labels= ['A','B','C','D','E']
    bar = pyecharts.Bar('柱状图')
    bar.add('one',labels,data,bar_category_gap='10%',is_stack=False,is_convert=False,mark_point=['max'],mark_line=['min','max'])
    bar.add('two',labels,data1,bar_category_gap='10%',is_stack=False,is_convert=False,mark_point=['max'],mark_line=['min','max'])
    page.add(bar)
    page.render()

def pie_graph():
    page = Page()
    data = [23, 85, 72, 43, 52]
    labels = ['A', 'B', 'C', 'D', 'E']
    pie = pyecharts.Pie('饼状图')
    pie.add('',labels,data,is_label_show=True,radius=[20,75],center=[25,50],rosetype='radius') #用半径来显示数值的大小
    pie.add('',labels,data,is_label_show=True,radius=[20,75],center=[75,50],rosetype='area')  #用面积来显示数值的大小
    page.add(pie)
    page.render()

from sklearn.datasets import load_iris
def box_graph():
    page = Page()
    iris = load_iris()
    iris.columns = ['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm']
    x = list(iris.columns[0:4])
    y = [list(iris.sepal_length_cm),list(iris.sepal_width_cm),list(iris.petal_length_cm),list(iris.petal_width_cm)]
    box = pyecharts.Boxplot('箱状图')
    y_data = box.prepare_data(y)  #将数据进行转换
    box.add('',x,y_data)
    page.add(box)
    page.render()




if __name__=='__main__':
    box_graph()