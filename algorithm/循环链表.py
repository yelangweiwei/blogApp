from  timeit import Timer
import time

def abc_algori():
    start_time = time.time()
    for a in range(1001):
        for b in range(0,1001-a):
            c = 1000-a-b
            if a**2+b**2==c**2:
                print('a b c:%d,%d,%d'%(a,b,c))
    end_time = time.time()
    print('elapsed:%f'%(end_time-start_time))


def tes1():
    li = []
    for i in range(10000):
        li.append(i)

def tes2():
    li = []
    for i in range(10000):
        li +=[i]

def tes3():
    li = [i for i in range(10000)]

def tes4():
    li = list(range(10000))

def tes5():
    li = []
    for i in range(10000):
        li.extend([i])

def tes6():
    li = []
    for i in range(10000):
        li.insert(0,i)

def tes_time():
    time1 = Timer('tes1()', "from __main__ import tes1")
    print('time1:', time1.timeit(1000))

    time2 = Timer('tes2()', 'from __main__ import tes2')
    print('time2:', time2.timeit(1000))

    time3 = Timer('tes3()', 'from __main__ import tes3')
    print('time3:', time3.timeit(1000))

    time4 = Timer('tes4()', 'from __main__ import tes4')
    print('time4:', time4.timeit(1000))

    time5 = Timer('tes5()', 'from __main__ import tes5')
    print('time5:', time5.timeit(1000))

    time6 = Timer('tes6()', 'from __main__ import tes6')
    print('time6:', time6.timeit(1000))


''''
顺序表
'''
def add_para():
    pass


'''
链表
'''
# 链表节点
class Node(object):
    def __init__(self,data,element=None):
        self.data =data #保存数据
        self.next = element #保存节点

#定义链表结构
class LinkList(object):
    def __init__(self,node=None):
        self.__head = node #头节点
        if node:  #节点不为空，这里节点要指向自己
            node.next = node
    #判断链表是否为空
    def is_empyty(self):
        if self.__head==None:
            return True
        else:
            return False

    #求链表的长度,尾结点不指向开始
    def get_linkList_length(self):
        if self.is_empyty():
            return 0
        else:
            current = self.__head
            num = 1
            while current.next!=self.__head:
                num+=1
                current  = current.next
            return num

    #遍历整个链表
    def travel_linkList(self):
        current = self.__head
        num = 0
        while current.next != self.__head:
            print(current.data,end=' ')
            num+=1
            current = current.next
        print(current.data,end=' ')  #打印最后一个节点信息
        if num==0:
            print('------------linkList is empty')

    #链表的头部添加元素
    def add_node(self,item):
        if self.is_empyty():
            self.__head=item
            item.next = self.__head
            return True
        else:
            #寻找尾结点
            current = self.__head
            while current.next!=self.__head:
                current = current.next
            item.next = self.__head
            self.__head = item
            current.next = self.__head
            return True

    #链表的尾部添加元素
    def append_node(self,item):
        if self.__head==None:
            self.__head = item
            item.next = self.__head
            return True
        else:
            current = self.__head
            while current.next!=self.__head:
                current = current.next
            current.next = item
            item.next = self.__head
            return True

    #在指定的位置添加元素
    def insert_node(self,item_index,item):
        pre = self.__head
        cout = 0
        if item_index<0: #默認是頭插法
            print(' index is less zero')
            self.add_node(item)
            return True
        elif item_index>=self.get_linkList_length():#默認是尾插法
            self.append_node(item)
            return True
        else:
            while cout<item_index:
                pre = pre.next
                cout+=1
            item.next = pre.next
            pre.next = item
            return True


    #删除节点
    def remove_node(self,item):
        if self.is_empyty():
            print('linkList is empty')
            return False
        else:
            if item.data==self.__head.data:
                self.__head = self.__head.next
                return True
            else:
                current = self.__head
                while current.next.data!=item.data:
                    current = current.next
                current.next = current.next.next
                return True

    #查找节点是否存在
    def search_node(self,item):
        if self.is_empyty():
            print(' there is no the item')
        else:
            current = self.__head
            while current.data!=item.data and current!=None:
                current = current.next
            if current.data==item.data:
                print('--------------there is ok')
                return True
            else:
                print('--------------there is error')
                return False

def single_linkeList():
    node_link = LinkList()
    # 判断节点链表是不是为空
    print('linkList is empty or not:', node_link.is_empyty())

    # 添加节点
    for i in range(10):
        node = Node(i)
        node_link.append_node(node)
    # 遍历节点
    # node_link.travel_linkList()
    # 计算节点的个数
    print('---------linkList count:', node_link.get_linkList_length())

    # 判断节点链表是不是为空
    print('linkList is empty or not:', node_link.is_empyty())

    # 在头节点插入元素
    # node_link.add_node(Node(333))
    # #遍历
    node_link.travel_linkList()

    # 在指定位置插入元素
    # node_link.insert_node(11,Node(110))
    # node_link.travel_linkList()

    # 判断某个值是否存在
    node_link.search_node(Node(1))
    # 删除某个节点
    node_link.remove_node(Node(1))
    node_link.travel_linkList()


if __name__=='__main__':










