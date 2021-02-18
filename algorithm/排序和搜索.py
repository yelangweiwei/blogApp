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
    gap：每次按照折半的情况进行
    
时间复杂度：
    最优时间复杂度：根据步长序列的不同而不同
    最坏时间复杂度：步长为1时，O(n^2)
稳定性：因为值会被分为几部分，所以相同的值会发生位置的改变，所以是不稳定的。
    
'''
def shell_sort(alist):
    '''

    :return:
    '''
    #gap变化到1之前，插入算法执行的次数
    gap = len(alist) // 2
    while gap>=1:
        for j in range(gap,len(alist)):  #和插入排序类似只是步长变为了gap
            i = j
            while i>0:
                if alist[i]<alist[i-gap]:
                    alist[i],alist[i-gap] = alist[i-gap],alist[i]
                    i-=gap
                else:
                    break
        gap//=2 #步长每次都缩短

def shell_sort_pra():
    alist = [2, 3, 6, 3, 6, 4, 7, 3, 34, 56, 89, 90]
    shell_sort(alist)
    print(alist)


'''
快速排序
    将第一个值找出来，通过这个值，将整个序列分为两部分，两部分一边都比这个值小，一边都比这个值大

时间复杂度：
    最优时间复杂度：

稳定性：
'''
def quick_sort(alist,first,last):
    if first>=last:
        return 1
    low = first
    high = last
    middle_value = alist[low]
    while low<high:
        #让high的游标左移
        while low<high and alist[high]>=middle_value:
            high-=1
        alist[low]=alist[high]
        #让low的游标右移
        while low<high and alist[low]<middle_value:
            low+=1
        alist[high]=alist[low]
    #对low左边的进行排序
    quick_sort(alist,first,low-1)
    #对low右边的进行排序
    quick_sort(alist,low+1,last)

def quick_sort_pra():
    alist = [2, 3, 6, 3, 6, 4, 7, 3, 34, 56, 89, 90,90]
    quick_sort(alist,0,len(alist)-1)
    print(alist)








'''
二叉树：
    二叉树的性质：
        1，在二叉树的第i层上至多有2^（i-1）个节点（i>0）
        2,深度为k的二叉树至多有2^k-1个节点（k>0）
        3,对于任意一棵二叉树，如果其叶节点数为N0,而度为2的节点总数为N2,则N0=N2+1
        4，具有n个节点的完全二叉树的深度必为log2(n+1)
        5,对于完全二叉树，从上至下，从左至右，编号为i的节点，其左孩子的编号为2*i,右孩子的编号为2*i+1,其双亲的编号必为i/2(i=1除外)
'''


class Node(object):
    def __init__(self,item):
        self.elem =item
        self.lchild = None
        self.rchild = None


class Tree(object):
    def __init__(self,root=None):
        self.root = root

    def add_item(self,item):
        node = Node(item)
        if self.root == None:
            self.root = node
            return
        queue = []
        queue.append(self.root)
        while queue:
            curr_node = queue.pop(0) #提取最左边的值
            if curr_node.lchild is None:
                curr_node.lchild= node
                return
            else:
                queue.append(curr_node.lchild)

            if curr_node.rchild is None:
                curr_node.rchild = node
                return
            else:
                queue.append(curr_node.rchild)

    def breadth_travel(self):
        '''广度遍历'''
        if self.root is None:
            return
        queue = [self.root]
        while queue:
            curr_node = queue.pop(0)
            print(curr_node.elem,end=',')
            if curr_node.lchild is not None:
                queue.append(curr_node.lchild)
            if curr_node.rchild is not None:
                queue.append(curr_node.rchild)

    def preorder(self,node):
        '''
        深度遍历
        先序遍历  根，左，右
        采用递归完成
        :return:
        '''
        if node is None:
            return
        #遍历头节点
        print(node.elem,end=',')
        #左节点
        self.preorder(node.lchild)
        #右节点
        self.preorder(node.rchild)
    def midorder(self,node):

        '''
         中序遍历：先遍历左节点，在遍历根节点，后遍历右节点
        :param node:
        :return:
        '''
        if node is None:
            return
        self.midorder(node.lchild)
        print(node.elem,end=',')
        self.midorder(node.rchild)

    def postOrder(self,node):
        '''
         后序遍历， 先遍历左左节点，在遍历右节点，最后根节点
        :param node:
        :return:
        '''

        if node is None:
            return
        self.postOrder(node.lchild)
        self.postOrder(node.rchild)
        print(node.elem,end=',')


def tree_pra():
    tree = Tree()
    for i in range(10):
        tree.add_item(i)

    # 广度遍历
    tree.breadth_travel()
    print('\n')
    # 先序遍历
    tree.preorder(tree.root)
    print('\n')
    # 中序遍历
    tree.midorder(tree.root)
    print('\n')
    # 后续遍历
    tree.postOrder(tree.root)


def sift_big(li,low,high):
    '''
    堆排序，特殊的完全二叉树
    大根堆：一棵完全二叉树，任何一个节点都比孩子节点大，排出来是升序的的
    小根堆： 满足任何一个节点都比孩子节点小。排出来是降序的
    当根节点的左右子树都是堆时，可以通过一次向下的调整来将其变换成一个堆
    li 表示树，low:表示树根，high表示最后一个节点的位置

    时间复杂度：O(nlog2N)
    https://www.cnblogs.com/jingmoxukong/p/4303826.html
    '''
    tmp = li[low]
    i = low  #指向根
    j = 2*i+1 #j指向两个孩子
    while j<=high: #当j>high，也就是到了子节点时，跳出
        if j+1<=high and li[j]<li[j+1]:#完全二叉树，左子树也要小于右子树,指向右孩子
            j=j+1
        if li[j]>tmp:
            li[i] = li[j]
            i=j
            j = i*2+1
        else:
            break  #子节点比当前值小
    li[i]=tmp


def sift_small(li,low,high):
    '''
    堆排序，特殊的完全二叉树
    大根堆：一棵完全二叉树，任何一个节点都比孩子节点大，排出来是升序的的
    小根堆： 满足任何一个节点都比孩子节点小。排出来是降序的
    当根节点的左右子树都是堆时，可以通过一次向下的调整来将其变换成一个堆
    li 表示树，low:表示树根，high表示最后一个节点的位置

    时间复杂度：O(nlog2N)
    https://www.cnblogs.com/jingmoxukong/p/4303826.html

    这个主要用在前k个值的排序
    '''
    tmp = li[low]
    i = low  #指向根
    j = 2*i+1 #j指向两个孩子
    while j<=high: #当j>high，也就是到了子节点时，跳出
        if j+1<=high and li[j]>li[j+1]:#完全二叉树，左子树也要小于右子树,指向右孩子
            j=j+1
        if li[j]<tmp:
            li[i] = li[j]
            i=j
            j = i*2+1
        else:
            break  #子节点比当前值小
    li[i]=tmp


def heap_sort(li):
    n = len(li)
    #构造堆,初始化堆,构造大顶堆
    for low in range(n//2-1,-1,-1):
        sift_small(li,low,n-1)  #开始的位置和最后一个数据的位置
    print(li)
    print('----------------------')
    #将大顶堆的值放在叶子节点处,最后的位置不断的向前移动
    for high in range(n-1,-1,-1):
        li[0],li[high]=li[high],li[0]  #退休棋子调整
        sift_small(li,0,high-1)
    print(li)

def heap_pra():
    alist = [2, 3, 6,4, 7,34, 56, 89, 90]
    heap_sort(alist)

#求前k个最大的，或者最小的值
def heap_k_pra():
    blist = [2, 3, 6, 4,34, 56, 89, 90,5,7,47]
    k =11
    alist= blist[:k]
    #初始化堆
    for low in range(k//2-1,-1,-1):
        sift_small(alist,low,k-1)
    #此时堆顶是最小的值
    for data in blist[k:]:
        if data>alist[0]:
            alist[0]=data
            sift_small(alist,0,k-1)

    #将堆顶的值进行输出
    for high in range(k-1,-1,-1):
        alist[0],alist[high]=alist[high],alist[0]
        sift_small(alist,0,high-1)
    print(alist)





if __name__=='__main__':
    heap_k_pra()

