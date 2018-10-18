import numpy as np
import matplotlib.pyplot as plt
import datagen, dataplot
import rff

X,Y = datagen.unit_ball_ideal(2,0.2,0.9,50)
X1 = np.array([X[idx,:] for idx in range(len(Y)) if Y[idx]==1])
X2 = np.array([X[idx,:] for idx in range(len(Y)) if Y[idx]==-1])
plt.scatter(X1[:,0],X1[:,1],c='r')
plt.scatter(X2[:,0],X2[:,1],c='b')
plt.show()
X,Y = datagen.unit_circle_ideal(0.2,0.9,50)
X1 = np.array([X[idx,:] for idx in range(len(Y)) if Y[idx]==1])
X2 = np.array([X[idx,:] for idx in range(len(Y)) if Y[idx]==-1])
plt.scatter(X1[:,0],X1[:,1],c='r')
plt.scatter(X2[:,0],X2[:,1],c='b')
plt.show()
