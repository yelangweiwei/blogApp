import numpy as py
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor  #(通过若分类器迭代产生强分类器的算法)

def dtRegression():
    rng = py.random.RandomState(0)
    x = py.linspace(0,6,100)[:,py.newaxis]
    y = py.sin(x).ravel()+py.sin(6*x).ravel()+rng.normal(0,0.1,x.shape[0])

    regr1 = DecisionTreeRegressor(max_depth=4)
    regr2 =AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300,random_state=rng)
    regr1.fit(x,y)
    regr2.fit(x,y)

    y_1 = regr1.predict(x)
    y_2 = regr2.predict(x)

    plt.figure()
    plt.scatter(x,y,c='k',label='training samples')
    plt.plot(x,y_1,c='g',label='n_estimators=1',linewidth = 2)
    plt.plot(x,y_2,c='r',label = 'n_estimators=300',linewidth=2)
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('boosted decision tree regression')
    plt.legend()
    plt.show()

def decision_regression():
    rng = py.random.RandomState(0)
    x = py.sort(5*rng.rand(80,1),axis=0)
    y = py.sin(x).ravel()
    y[::5]+=3*(0.5-py.random.rand(16))

    regr1 = DecisionTreeRegressor(max_depth=6)
    regr2 = DecisionTreeRegressor(max_depth=10)


    regr1.fit(x,y)
    regr2.fit(x,y)

    x_test = py.arange(0.0,5.0,0.01)[:,py.newaxis]
    y_1 = regr1.predict(x_test)
    y_2 = regr2.predict(x_test)

    plt.figure()
    plt.scatter(x,y,s=20,edgecolor='black',c='darkorange',label='data')
    plt.plot(x_test,y_1,color='cornflowerblue',label='max_depth=2',linewidth=2)
    plt.plot(x_test,y_2,color='yellowgreen',label='max_depth=5',linewidth=2)
    plt.xlabel('data')
    plt.ylabel('target')
    plt.legend()
    plt.show()





if __name__=='__main__':
    decision_regression()