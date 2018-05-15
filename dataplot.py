import numpy as np
from matplotlib import pyplot as plt

def plot_interval(X,Y,ratio=1):
    m = int(len(X) * ratio)
    X = X[0:m]
    Y = Y[0:m]
    c = list()
    for idx in range(m):
        if Y[idx]==1:
            c.append('r')
        else:
            c.append('b')
    fig = plt.figure()
    plt.scatter(X,Y,c=c)
    plt.savefig('image/interval.eps')
    plt.close(fig)
    return 1

def plot_circle(X,Y,ratio=1):
    m = int(len(X) * ratio)
    shufflelist = np.random.choice(len(X),size=m,replace=False)
    X = X[shufflelist,:]
    Y = Y[shufflelist]
    B = X[np.where(Y==-1)]
    R = X[np.where(Y==1)]
    fig = plt.figure()
    plt.scatter(B[:,0],B[:,1],c='b',marker='x')
    plt.scatter(R[:,0],R[:,1],facecolors='none',edgecolors='r',marker='o')
    circle = plt.Circle((0,0),1,fill=False)
    plt.gcf().gca().add_artist(circle)
    plt.axis('equal')
    plt.savefig('image/circle.eps')
    plt.close(fig)
    return 1
