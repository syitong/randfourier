import numpy as np
import matplotlib.pyplot as plt
import rff
import tensorflow as tf

samplesize = np.arange(1000,60001,5000)
orfsvm = np.zeros((10,len(samplesize)))
urfsvm = np.zeros((10,len(samplesize)))
for idx in range(1,11,1):
    orfsvm[idx-1,:] = np.loadtxt('result/ORFSVM'+str(idx))
    urfsvm[idx-1,:] = np.loadtxt('result/URFSVM'+str(idx))

orfmean = np.mean(orfsvm,axis=0)
urfmean = np.mean(urfsvm,axis=0)
orfstd = np.std(orfsvm,axis=0)
urfstd = np.std(urfsvm,axis=0)

plt.title("opt vs unif feature selection on MNIST")
plt.xlabel('sample size (k)')
plt.ylabel('accuracy')
plt.xticks(samplesize/1000)
plt.errorbar(samplesize/1000,orfmean,yerr=orfstd,fmt='rx-')
plt.errorbar(samplesize/1000,urfmean,yerr=urfstd,fmt='bx-')
plt.savefig('image/opt_vs_unif.eps')
