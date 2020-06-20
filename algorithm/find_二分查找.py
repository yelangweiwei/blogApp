
import numpy as np

'''
二分查找
运算速度
复杂度 log（N）的对数，这个就是二分查找使用的查找到需要元素的位置，log这里指的就是log2;
特点，检测的次数很少，但是检测的数据必须是有序的才可以。所以对于无需的列表，必须先进行排序

最多的猜测的次数和列表的数据的长度相同，这被称为线性时间
二分查找运行的时间是对数时间

大O表示法：指出算法的速度有多快
    大O表示法计算的是操作数
    检查某个元素需要n次，使用大O表示，这个运行时间是O(n),大O表示法并非以s为单位的速度，大O表示法能都让你比较操作数。指出了算法运算时间的增速
    比如：检查长度为n的列表，二分查找需要执行的logn次的操作，使用大O表示法，这个表示方式是：O(logn)


'''

def find_by_binary(data_list,item):
    low_index = 0
    high_index = len(data_list)-1

    while low_index<=high_index:
        mid = (low_index + high_index) // 2  # python会自动将平均值向下取整
        if item<data_list[mid]:
            high_index = mid-1
            continue
        elif item>data_list[mid]:
            low_index = mid+1
            continue
        else:
            return mid
    return None  #没有找到的情况下








if __name__=="__main__":
    data_list = range(100)
    print(data_list)
    item = 30
    item_index = find_by_binary(data_list,item)
    print(item_index)




