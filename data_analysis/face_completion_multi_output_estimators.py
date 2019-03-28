import os
import numpy as py
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

print(__doc__)
def completion_output_estimators():
    #load datasets
    data_path = os.path.dirname(os.path.realpath(__file__))+'/data/'
    data = fetch_olivetti_faces(data_home=data_path)
    targets = data.target
    print(targets)
    print(len(targets))
    #将三维的数据，（元素的个数，图像的像素，图像的像素），转换为（元素的个数，像素的乘积）
    data = data.images.reshape((len(data.images),-1))
    train = data[targets<30]
    test = data[targets>=30]

    #test on a subset of people
    n_faces = 5
    rng = check_random_state(4) #随机状态实例，每次划分测试集和训练集合时，随机分派，但是每次分配的结果是确定的
    face_ids = rng.randint(test.shape[0],size=(n_faces,))
    test = test[face_ids,:]

    #取训练的数据和测试的数据
    n_pixels = data.shape[1]
    #upper half of the faces
    x_train = train[:,:(n_pixels+1)//2]
    #lower half of the faces
    y_train = train[:,n_pixels//2:]
    x_test = test[:,:(n_pixels+1)//2]
    y_test = test[:,n_pixels//2:]

    #fit estimators
    ESTIMATORS = {
        'Extra trees':ExtraTreesRegressor(n_estimators=10,max_features=32,random_state=0),
        'K-nn':KNeighborsRegressor(),
        'Linear regression':LinearRegression(),
        'Ridge':RidgeCV(),
    }

    y_test_predict = dict()
    for name,estimator in ESTIMATORS.items():
        estimator.fit(x_train,y_train)
        y_test_predict[name] = estimator.predict(x_test)

    #plot the completed faces
    image_shape = (64,64)
    n_cols = 1+len(ESTIMATORS)
    plt.figure(figsize=(2.*n_cols,2.26*n_faces))
    plt.suptitle('Face completion with multi-output estimators',size = 16)

    for i in range(n_faces):
        true_face = py.hstack((x_test[i],y_test[i]))
        if i:
            sub = plt.subplot(n_faces,n_cols,i*n_cols+1)
        else:
            sub = plt.subplot(n_faces,n_cols,i*n_cols+1,title='true_faces')

        sub.axis('off')
        sub.imshow(true_face.reshape(image_shape),cmap=plt.cm.gray,interpolation='nearest')

        for j,est in enumerate(sorted(ESTIMATORS)):
            completed_face = py.hstack((x_test[i],y_test_predict[est][i]))
            if i :
                sub = plt.subplot(n_faces,n_cols,i*n_cols+2+j)
            else:
                sub = plt.subplot(n_faces,n_cols,i*n_cols+2+j,title=est)

            sub.axis('off')
            sub.imshow(completed_face.reshape(image_shape),cmap=plt.cm.gray,interpolation='nearest')
    plt.show()

from sklearn import neighbors
def nearestNeighborsRegression():
    py.random.seed(0)
    x= py.sort(5*py.random.rand(40,1),axis=0)
    t = py.linspace(0,5,500)[:,py.newaxis]
    y = py.sin(x).ravel()

    #add noise to targets
    y[::5]+=1*(0.5-py.random.rand(8))

    #fit regression model
    n_neighbors = 5
    for i ,weights in enumerate(['uniform','distance']):
        knn = neighbors.KNeighborsRegressor(n_neighbors,weights=weights)
        y_ = knn.fit(x,y).predict(t)

        plt.subplot(2,1,i+1)
        plt.scatter(x,y,c='k',label='data')
        plt.plot(t,y_,c='g',label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("NeighborsRegression(k=%i,weights='%s')"%(n_neighbors,weights))
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    # completion_output_estimators()
    nearestNeighborsRegression()