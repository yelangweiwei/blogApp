from  timeit import Timer


class Stack(object):
    def __init__(self):
        self.__stack_list = []

    def push(self,item):#入栈
        self.__stack_list.append(item)

    def pop(self): #弹出栈顶元素,栈中就没有这个元素了
        if self.__stack_list:
            return self.__stack_list.pop()
        else:
            return None

    def peek(self): #栈顶出栈，栈中还有这个元素
        if self.__stack_list:
            return self.__stack_list[-1]
        else:
            return None

    def is_empty(self): #判空
        if self.__stack_list:
            return False
        else:
            return True


    def stack_size(self): #返回栈的大小
        return len(self.__stack_list)


def stackPra():
    stack = Stack()
    print('--------------is_empty:', stack.is_empty())
    print('--------------stack_size:', stack.stack_size())
    stack.push(1)
    stack.push(2)
    stack.push(3)
    stack.push(4)

    print('-------stack.pop:', stack.pop())

    print('------stack.peek:', stack.peek())
    print('------stack.peek:', stack.peek())

    print('--------------is_empty:', stack.is_empty())
    print('--------------stack_size:', stack.stack_size())


class Queue(object):
    def __init__(self):
        self.__list = []

    def enqueue(self,item):#入队 O(1)
        self.__list.append(item)

    def dequeue(self):#出队O（n）
        if self.__list:
            return self.__list.pop(0)
        else:
            return None

    def is_empty(self):
        if self.__list:
            return False
        else:
            return True

    def queue_size(self):
        return len(self.__list)
def queueP():
    q = Queue()
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    q.enqueue(4)

    print(q.is_empty())
    print(q.queue_size())

    print(q.dequeue())
    print(q.dequeue())


class Deque(object):
    def __init__(self):
        self.__list = []

    def add_front(self,item):#入队 O(1)
        self.__list.insert(0,item)

    def add_rear(self,item):
        self.__list.append(item)

    def get_front(self):#出队O（n）
        if self.__list:
            return self.__list.pop(0)
        else:
            return None
    def get_rear(self):
        if self.__list:
            return self.__list.pop()
        else:
            return None

    def is_empty(self):
        if self.__list:
            return False
        else:
            return True

    def queue_size(self):
        return len(self.__list)

if __name__=='__main__':
    queueP()



