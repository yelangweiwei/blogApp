'''
做一个日志的分析统计：分析的数据包括：

分析nginx的log日志：
每条数据id，分析的日期，记录状态，分析花费的时间t1,t2，分析的机器，状态

'''

import os
import pandas as pd
import matplotlib.pylab as plt

def analysis_ai_time(log_ai_list):
    ai_15_count = 0  # 统计出现在1.5s以下的数据有多少
    ai_18_count = 0  # 统计出现在1.5~1.8s之间的数据有多少
    ai_2_count = 0  # 统计出现在1.8~2s之间的数据有多少
    ai_25_count = 0  # 统计出现在2~2.5S之间的数据有多少
    ai_3_count = 0  # 统计出现在2.5~3s之间的数据有多少
    ai_35_count = 0  # 统计出现在3~3.5s之间的数据有多少
    ai_4_count = 0  # 统计出现在3.5~4.5s的数据有多少
    ai_more_4_cout=0 #统计超过4.5s的数据有多少

    for tem_time in log_ai_list:
        if tem_time=='-': #这个相当于没有返回
            ai_more_4_cout+=1
            continue
        tem_time = float(tem_time)
        if tem_time <= 1.5:
            ai_15_count += 1
        elif tem_time > 1.5 and tem_time <= 1.8:
            ai_18_count += 1
        elif tem_time > 1.8 and tem_time <= 2:
            ai_2_count += 1
        elif tem_time > 2 and tem_time <= 2.5:
            ai_25_count += 1
        elif tem_time > 2.5 and tem_time <= 3:
            ai_3_count += 1
        elif tem_time > 3 and tem_time <= 3.5:
            ai_35_count += 1
        elif tem_time > 3.5 and tem_time<=4.8:
            ai_4_count += 1
        else:
            ai_more_4_cout+=1

    print('统计出现在1.5s以下的数据有:',ai_15_count)
    print('统计出现在1.5~1.8s之间的数据有:',ai_18_count)
    print('统计出现在1.8~2s之间的数据有:',ai_2_count)
    print('统计出现在2~2.5S之间的数据有:',ai_25_count)
    print('统计出现在2.5~3s之间的数据有:',ai_3_count)
    print('统计出现在3~3.5s之间的数据有:',ai_35_count)
    print('统计出现在3.5~4.5的数据有:',ai_4_count)
    print('统计出现在4.5以上的数据有:',ai_more_4_cout)

    cout_list = [ai_15_count/len(log_ai_list),ai_18_count/len(log_ai_list),ai_2_count/len(log_ai_list),
                 ai_25_count/len(log_ai_list),ai_3_count/len(log_ai_list),ai_35_count/len(log_ai_list),
                 ai_4_count/len(log_ai_list),ai_more_4_cout/len(log_ai_list)]
    colums_list = ['<1.5','1.5<time<=1.8','1.8<time<=2','2<time<=2.5','2.5<time<=3','3<time<=3.5','3.5<time<=4.8','4.8<time']

    return colums_list,cout_list







def statistics_by_column(pb_log,nginx_log_path):
    #统计ai分析没有成功的数据
    ai_analysis_cout_list = []
    for i in range(len(pb_log['status_uwsgi'])):
        data_ai_status = pb_log['status_uwsgi'][i]
        if data_ai_status!='200':
            ai_analysis_cout_list.append(pb_log['data_id'][i])
    print('ai 分析出现错误占所有数据的比例:',len(ai_analysis_cout_list)/len(pb_log['status_uwsgi']))
    print('ai 分析出现错误的个数:',len(ai_analysis_cout_list))
    # print(list(ai_analysis_cout_list))


    #统计没有成功返回客户端的数据
    nginx_client_list = []
    for j in range(len(pb_log['status_nginx_client'])):
        client_status = pb_log['status_nginx_client'][j]
        if client_status!='200':
            nginx_client_list.append(pb_log['data_id'][j])
    print('nginx client 没有成功的占所有数据的比例:',len(nginx_client_list)/len(pb_log['status_uwsgi']))
    print('nginx client 没有成功的个数:',len(nginx_client_list))
    # print(list(nginx_client_list))

    #将异常的数据记录下来
    save_path = nginx_log_path+'result/nginx.txt'
    with open(save_path,'w') as wh:
        for file in nginx_client_list:
            wh.write(file+'\n')





    #统计所有数据ai分析时间的柱状图，统计几个时间段的数据量，查看数据多分布在哪个范围内
    log_ai_list = pb_log['time_ai']
    time_ai_columns_list, time_ai_cout_list = analysis_ai_time(log_ai_list)

    # 统计所有数据返回客户端的柱状图，统计几个时间段的数据量，查看数据多分布在哪个范围内
    log_client_list = pb_log['time_client']
    log_client_columns_list, log_client_cout_list = analysis_ai_time(log_client_list)

    #两个图分开对比
    # plt.figure(figsize=(10, 6))
    # plt.subplot(211)
    # plt.bar(time_ai_columns_list, time_ai_cout_list)
    # plt.title('log_ai_show')
    # plt.subplot(212)
    # plt.bar(log_client_columns_list,log_client_cout_list)
    # plt.title('log_client_show')
    # plt.show()

    #并列柱状图
    width = 0.4
    x = list(range(len(time_ai_columns_list)))
    plt.bar(x,time_ai_cout_list,width=width,label='time_ai',fc = 'y')
    for i in range(len(x)):
        x[i] = x[i]+width
    plt.bar(x,log_client_cout_list,width=width,label='time_client',fc='r',tick_label=log_client_columns_list)
    plt.legend()
    plt.show()


def analysis_nginx_log(nginx_log_path):
    nginx_log_file_list = os.listdir(nginx_log_path)

    '''
        date:分析数据的日期
        data_id:要分析的数据id
        status_nginx_client: nginx 分析并返回客户端的状态码
        machine: 使用的是哪个机器
        time_client: 返回客户端的时间
        time_ai:AI分析的时间
        status_uwsgi:AI分析状态码
    '''
    head_list = ['date','data_id','status_nginx_client','time_client','time_ai','machine','status_uwsgi']

    #读取每个nginx日志的内容
    for nginx_file in nginx_log_file_list:
        nginx_file_path  = nginx_log_path+nginx_file
        with open(nginx_file_path,'r') as rh:
            nginx_content = rh.readlines()

        #获取数据中指定的字段
        main_content_list = []
        for line_content in nginx_content:
            single_line_list =[]
            content = line_content.split(' ')
            # print(list(content))
            date = content[3].split('[')[1]    #遗留的问题，这里的日期不好转换
            single_line_list.append(date)

            data_id = content[7].split('=')[1]
            single_line_list.append(data_id)

            status_nginx_client = content[9]
            single_line_list.append(status_nginx_client)

            if content[20].split('"')[1]=='-':
                single_line_list.append('-')
            elif content[20].split('"')[1].find(',')!=-1:
                continue
            else:
                time_client = float(content[20].split('"')[1])
                single_line_list.append(time_client)

            if content[21].split('"')[1]=='-':
                single_line_list.append('-')
            elif content[21].split('"')[1].find(',')!=-1:
                continue
            else:
                time_ai = float(content[21].split('"')[1])
                single_line_list.append(time_ai)

            machine = content[22].split('"')[1]
            single_line_list.append(machine)

            status_uwsgi = content[23].split('"')[1]
            single_line_list.append(status_uwsgi)
            main_content_list.append(single_line_list)

        #将每一行的数据按照要分析的字段进行区分
        pb_log = pd.DataFrame(main_content_list,columns=head_list)

        # 所有数据的一个统计图,太大了，画不出
        # pb_log.plot(kind='bar')
        # plt.show()

        # print(pb_log[:10])
        #将获得数据结构根据需要的字段进行统计分析
        statistics_by_column(pb_log,nginx_log_path)







if __name__=='__main__':
    nginx_log_path = 'G:\\20190426\\济南检测\\2018-01\\real_test_data\\test\\20191122实时线上日志分析\\nginx\\'
    analysis_nginx_log(nginx_log_path)