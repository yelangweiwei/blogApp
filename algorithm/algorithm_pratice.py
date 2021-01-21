
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
        简单查找，就是把所有的数据都找一遍，这种算法的运行时间不可能超过O(n)

几种常见的大O运行时间
    1：O(logn) 对数时间，这样的算法包括二分查找
    2：O(n) 线性时间，这样的算法包括简单查找
    3：O(n*logn) 比如快速排序，一种较快的排序算法
    4：O(n^2) 一种比较慢的排序算法
    5：O(n!) 一种非常慢的算法  比如商旅问题解决方案

算法的速度指的并非时间，而是操作数的增速


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

def binary_tes():
    data_list = range(100)
    print(data_list)
    item = 30
    item_index = find_by_binary(data_list, item)
    print(item_index)


#工作需要，将某个文件夹下的文件的名字进行更改
import os
def update_file_name():
    out_first_path = 'G:\\20190426\\济南设备嵌入式分析\\data\\漏检的长程数据\\截取后longdata\\'
    dir_list = os.listdir(out_first_path)
    for dir in dir_list:
        file_dir = out_first_path+dir+'\\'
        file_list = os.listdir(file_dir)
        for file in file_list:
            if file.endswith('F.DAT'):
                continue
            ori_B_file = file_dir +file
            file = file.split('-')[0]+'B.DAT'
            new_B_file = file_dir+file
            print('------------------------------:',new_B_file)
            with open(ori_B_file,'rb') as rh:
                B_content = rh.read()
            with open(new_B_file,'wb') as wh:
                wh.write(B_content)


#统计
event_list = ['filer_data','DetectSquareWave','qrs_detect_merge','DetectVF','QRS_width_calculation',
              'detect_master_beat_with_rpos','GetEcgTestTemplateMeasure','Ladder_warning','RR_Diff_warning',
              'Long_RR_warning','QRS_width_warning','QT_duration_warning','ST_T_segment_change_warning',
              'Get_square_wave','Get_vf_wave','CalQRS_TemplateAvg_V4','Cal_TemplateAvg_Param',
              'Yocaly_Warning','AIEcgAnalysisWarn','GetEcgTestTemplateMeasure____000',
              'GetEcgTestTemplateMeasure____111','qrs_detect_merge___111','qrs_detect_merge___222','qrs_detect_merge___000','AnalysisWarn']#,'Yocaly_Warning','AIEcgAnalysisWarn']

def statistic_event_time():
    file_path = 'G:\\20190426\\济南设备嵌入式分析\\data\\time_log\\time_log.txt'
    with open(file_path,'r') as rh:
        lines = rh.readlines()

    event_time_dict = {}
    for key in event_list:
        event_time_dict[key] =[]
    for line in lines:
        key = line.split(":")[1].strip()
        cost_time = float(line.split(':')[0])
        if key not in event_list:
            continue
        event_time_dict[key].append(cost_time)



    with open('G:\\20190426\\济南设备嵌入式分析\\data\\time_log\\result.txt','w') as wh:
        for key in event_list:
            time_list = event_time_dict[key]
            if not len(time_list):
                continue
            wh.write('-----------------key:'+key)
            #将运行时间大意100的提取出来
            time_g_100_list = [x for x in time_list if x>200]
            for time in time_g_100_list:
                wh.write(str(time) + ',')
            wh.write('\n')
            wh.write('------------------g_100_num:'+str(len(time_g_100_list))+'\n')
            wh.write('max_time:'+str(max(time_list))+'\n')
            wh.write('min_time:'+str(min(time_list))+'\n')
            wh.write('avg_time:'+str(sum(time_list)/len(time_list))+'\n')




if __name__=="__main__":
    statistic_event_time()




