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

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
def decision_tree_structer():

    iris = load_iris()
    x = iris.data
    y = iris.target
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
    estimator = DecisionTreeClassifier(max_leaf_nodes=3,random_state=0)
    estimator.fit(x_train,y_train)

    n_nodes = estimator.tree_.node_count
    print('n_nodes:',n_nodes)
    child_left = estimator.tree_.children_left
    print('child_left:',child_left)
    child_right = estimator.tree_.children_right
    print('child_right:',child_right)
    feature = estimator.tree_.feature
    print('feature:',feature)
    threshold = estimator.tree_.threshold
    print('threshold:',threshold)

    node_depth = py.zeros(shape=n_nodes,dtype=py.int64)
    is_leaves = py.zeros(shape=n_nodes,dtype=bool)
    stack = [(0,-1)]  #seed is the root node id and its parent depth
    while len(stack)>0:
        node_id,parent_depth = stack.pop()
        node_depth[node_id] = parent_depth+1

        if (child_left[node_id]!=child_right[node_id]):
            stack.append((child_left[node_id],parent_depth+1))
            stack.append((child_right[node_id],parent_depth+1))
        else:
            is_leaves[node_id]=True







if __name__=='__main__':
    decision_tree_structer()