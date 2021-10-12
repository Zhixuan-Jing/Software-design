#Machine Learning 3-3 Logistic Regression
import torch
import numpy as np

def sigmoid(x):
    y=1.0/(1+np.exp(-x))
    return y

def flag(n):
    if(n>0.5):
        return 1
    elif(n==0.5):
        return 0
    else:
        return -1

def test(dataset, labelset, w):
    
    hat=np.dot(dataset,w)
    cnt=0
    size=np.size(hat,axis=0)
    for i in range (0,size):
        if(flag(hat[i])==labelset[i]):
            cnt=cnt+1
    return cnt/np.size(dataset,axis=0)

def train(dataset, labelset, train_data, train_label):
    w=np.ones((len(dataset[0])+1,1))
    data=np.mat(dataset)
    label=np.mat(labelset).transpose()

    a=np.ones((len(dataset),1))
    data=np.c_[data,a]
    n=0.5
    rate=0
    for i in range (1,2000) :
        # L(y) = (Y-y)^2, 
        # dL(y)/dy = 2y-2Y
        c = sigmoid(np.dot(data,w))
        b = c-label
        c=np.asarray(c)
        b=np.asarray(b)
        k=c*(1-c)*b
        grad = np.dot(np.transpose(data),k)
        w = w-grad*n
        rate = test(data, train_label, w)
    print(rate)
    return w

if __name__ == "__main__":
    dataset=[
        [0.697,0.460],
        [0.774,0.376],
        [0.634,0.264],
        [0.608,0.318],
        [0.556,0.215],
        [0.403,0.237],
        [0.481,0.149],
        [0.437,0.211],
        [0.666,0.091],
        [0.243,0.267],
        [0.245,0.057],
        [0.343,0.099],
        [0.639,0.161],
        [0.657,0.198],
        [0.360,0.370],
        [0.593,0.042],
        [0.719,0.103]
    ]

    labelset=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]

    test_data=[
        [0.697,0.460],
        [0.774,0.376],
        [0.634,0.264],
        [0.608,0.318],
        [0.556,0.215],
        [0.403,0.237],
        [0.481,0.149],
        [0.437,0.211],
        [0.666,0.091],
        [0.243,0.267],
        [0.245,0.057],
        [0.343,0.099],
        [0.639,0.161],
        [0.657,0.198],
        [0.360,0.370],
        [0.593,0.042],
        [0.719,0.103]
    ]

    test_label=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]

    w=train(dataset,labelset,test_data,test_label)
    print(w)



