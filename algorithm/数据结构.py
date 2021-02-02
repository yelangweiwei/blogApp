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





if __name__=='__main__':


