import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# my module
import datagen, dataplot
import rff

### set up parameters
datarange = 0.5
overlap = 0
samplesize = 10000
trials = 1

### generate train and test dataset
X,Y = datagen.unit_circle(datarange,
                          overlap,
                          samplesize)
X_train,X_test,Y_train,Y_test = train_test_split(X,
                                                 Y,
                                                 test_size = 0.3,
                                                 random_state=0)
gamma = datagen.gamma_est(X_train,portion = 
                          min(len(X_train),100)/len(X_train))
reg = 0
p = 0.5
sampler = rff.myReLUSampler(len(X[0]),gamma,10)
#sampler = rff.myRBFSampler(len(X[0]),gamma,20)
HyperModel = rff.HyperRFSVM(sampler,p,reg)
kscore = list()
ksparsity = list()
#for idx in range(trials):
#    clf = svm.SVC(C=reg,gamma=gamma)
#    clf.fit(X_train,Y_train)
#    kscore.append(clf.score(X_test,Y_test))
#    print kscore

for idx in range(trials):
    train_score = HyperModel.train(6,X_train,Y_train)
#    cum_score = np.cumsum(train_score)
#    average_score = [cum_score[i]/(i+1) for i in range(len(cum_score))]
#    print HyperModel.w
#    plt.plot(average_score)
#    plt.show()

l = len(Y_test)
output = HyperModel.test(X_test)
score = 0
for idx in range(l):
    if output[idx] == Y_test[idx]:
        score = score + 1
print(float(score) / l)
