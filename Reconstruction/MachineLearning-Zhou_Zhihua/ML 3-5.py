import numpy as np


def train(Xa, Xb, ua, ub, Ea, Eb):
    w=np.ones((2,1))
    Sw=Ea+Eb   
    Sb=np.dot((ua-ub),np.transpose(ua-ub))

    w=np.dot(np.matrix(Sw).I,(ua-ub))
    return w

if __name__ == "__main__":

    Xa=[
        [0.697,0.460],
        [0.774,0.376],
        [0.634,0.264],
        [0.608,0.318],
        [0.556,0.215],
        [0.403,0.237],
        [0.481,0.149],
        [0.437,0.211]
    ]
    Xb=[
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
    ua=np.mean(Xa,axis=0)
    ub=np.mean(Xb,axis=0)
    Ea=np.cov(Xa,rowvar=False)
    Eb=np.cov(Xb,rowvar=False)
    res=train(Xa, Xb, ua, ub, Ea, Eb)
    print(res)
