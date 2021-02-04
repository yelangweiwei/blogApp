from  timeit import Timer
import time

'''
链表
'''
# 链表节点
class Node(object):
    def __init__(self,data,pre_element=None,next_element=None):
        self.pre = pre_element
        self.data =data #保存数据
        self.next = next_element #保存节点

#定义链表结构
class DoubleLinkList(object):
    def __init__(self,node=None):
        self.__head = node #头节点

    #判断链表是否为空
    def is_empyty(self):
        if self.__head==None:
            return True
        else:
            return False

    #求链表的长度
    def get_linkList_length(self):
        current = self.__head
        num = 0
        while current!=None:
            num+=1
            current  = current.next
        return num

    #遍历整个链表
    def travel_linkList(self):
        current = self.__head
        num = 0
        while current != None:
            print(current.data,end=' ')
            num+=1
            current = current.next
        if num==0:
            print('------------linkList is empty')

    #链表的头部添加元素
    def add_node(self,item):
        item.next = self.__head
        item.next.pre  = item
        self.__head = item


    #链表的尾部添加元素
    def append_node(self,item):
        if self.__head==None:
            self.__head = item
        else:
            current = self.__head
            while current.next!=None:
                current = current.next
            current.next = item
            item.pre = current


    #在指定的位置添加元素
    def insert_node(self,item_index,item):
        pre = self.__head
        cout = 0
        if item_index<=0: #默認是頭插法
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
            pre.next.pre = item
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
                current.next.pre = current
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

def double_linkeList():
    node_link = DoubleLinkList()
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
    double_linkeList()









