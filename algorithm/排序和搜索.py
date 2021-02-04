import sys

'''
冒泡排序
外层控制循环的次数
内存控制每次循环各个值进行比较
复杂度：最优的情况，都是有序上升的，O(n^2);最坏的情况：都是降序的，O(n^2);但是最坏的情况下要移动值，增加了复杂度；最优的情况下，只是遍历的次数多。
稳定性：稳定的，相同的值，不会出现位置的改变
'''
def bubule_sort(alist):
    for i in range(len(alist)):
        for j in range(0,len(alist)-1-i):
            if alist[j]>alist[j+1]:
                alist[j],alist[j+1] = alist[j+1],alist[j]

#最优状况，复杂度还是O(n)，只是进行了内层的一次循环，最坏的还是O(n^2)
def bubule_sort_update(alist):
    for i in range(len(alist)):
        flag= False
        for j in range(0,len(alist)-1-i):
            if alist[j]>alist[j+1]:
                alist[j],alist[j+1] = alist[j+1],alist[j]
                flag = True
        if flag==False:#不需要在排序了，走了一圈，数据顺序没变，不需要交换，直接跳出就可以了
            break

def bubul_sort_pra():
    alist = [2, 3, 6, 3, 6, 4, 7, 3, 34, 56, 89, 90]
    bubule_sort(alist)
    print(alist)

'''
选择排序
    始终在无序的序列中选择最小的，放在前边排好序的,选择排序注重从后边没有排好序的序列中取得最小值
    复杂度：o(n^2)
    稳定性：不稳定，相同的值会发生改变，比如按照升序排列，前边的值会先排在后边
'''
def select_sort(alist):
    for j in range(len(alist)-1):
        min_index = j
        for i in range(j+1,len(alist)):
            if alist[i]<alist[min_index]:
                min_index = i
        alist[j],alist[min_index]=alist[min_index],alist[j]

def select_sort_pra():
    alist = [2, 3, 6, 3, 6, 4, 7, 3, 34, 56, 89, 90]
    select_sort(alist)
    print(alist)


'''
插入算法
把数据分为两部分，一部分是是排好序的，一部分是没有排好序的
后边无序的元素和前边有序的元素进行比较，交换，插进有序序列，  之后的操作依次类推
先假设第一个元素是最小值，
时间复杂度：最好的情况是：O(n),内层复杂度是O(1)，外层是n;
    最坏的情况： o(n),内存复杂度O(n),所以是O(n^2)
稳定性：对于同样的数，不会出现顺序的改变，是稳定性的。

'''


def insert_sort(alist):
    '''
    插入排序
    :param alist:
    :return:
    '''
    # 从后边的无序集合中取值
    for j in range(1, len(alist)):
        # i代表内存循环起始值
        i = j
        # 执行的是，从右边的无需序列去取出第一个元素，然后将其和左侧的有序集合中，进行比较，插入。
        while i > 0:
            if alist[i] < alist[i - 1]:
                alist[i - 1], alist[i] = alist[i], alist[i - 1]
                i -= 1
            else:
                break


def insert_sort_pra():
    alist = [2, 3, 6, 3, 6, 4, 7, 3, 34, 56, 89, 90]
    insert_sort(alist)
    print(alist)


'''
希尔排序
   是插入排序的一种，也称缩小增量排序，是直接插入排序算法的一种更高效的改进版，希尔排序是非稳定排序算法，希尔排序是把记录按下标的一定增量分组
   对每组使用直接插入排序算法排序；随着增量逐渐减小，每组包含的关键词越来越多，当增量减小至1时，这个文件恰被分为一组，算法终止
   
排序过程：
    基本思想是：将数组列在一个表中，并对列分别进行插入排序，重复这个过程，不过每次用更长的列来进行，（步长更长了，列数更少了），最后整个表就只有一列了
    将数组转换至表是为了更好的理解这算法，算法本身还是使用数组进行排序 
'''
def shell_sort(alist):
    '''

    :return:
    '''







if __name__=='__main__':
    select_sort_pra()