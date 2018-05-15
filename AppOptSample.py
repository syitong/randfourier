"""
This code is for comparing the performance of different
ways of sampling random features. In particular, it shows
the test performance of KSVM, RFSVM with uniform feature
distribution, and RFSVM with approximate optimized feature
distribution.
"""
from sys import argv
import csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
# my module
import rff

#### set up data parameters
#gap = 0.2
#label_prob = 0.8
#samplesize = 1500
#logclist = np.arange(-3,7,1)
#trials = 10
#
#### set up feature parameters
#X_pool_fraction = 0.3
#feature_pool_size = 300
#n_components = 3
#
#### generate train and test dataset
#X,Y = rff.unit_circle_ideal(gap,label_prob,samplesize)
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33,random_state=0)
#
#### estimate gamma in the rbf kernel. gamma here is actually 1/variance
#gamma = rff.gamma_est(X_train)

### load data and parameters
timepoint = list()
tasklist = list()
task = 'load data'
tasklist.append(task)
timepoint.append(time.process_time())
X_test = np.loadtxt('data/ideal_Xtest.txt')
Y_test = np.loadtxt('data/ideal_Ytest.txt')
X_train_p = np.loadtxt('data/ideal_Xtrain.txt')
Y_train_p = np.loadtxt('data/ideal_Ytrain.txt')
with open('data/ideal_parameter.csv','r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pass
    gap = float(row['gap'])
    datasize = float(row['samplesize'])
    label_prob = float(row['label_prob'])

### set up experiment parameters
#logsamplesize = 2 + int(argv[1])
logsamplesize = 3
logla_start = -8.0
logla_end = 1.0
logclist = np.arange(-logla_end - logsamplesize,
		     -logla_start - logsamplesize,1)
samplesize = 10**logsamplesize
trials = 10
fileprefix = '{:.1e}'.format(samplesize)

### set up feature parameters
X_pool_fraction = 0.3
#n_components = int(logsamplesize**2 / 3) + 1
#feature_pool_size = n_components * 100

print(task + ' done')
### shuffle the training set
task = 'shuffle data'
tasklist.append(task)
timepoint.append(time.process_time())

shufflelist = np.random.choice(len(X_train_p),size=samplesize,replace=False)
X_train = X_train_p[shufflelist,:]
Y_train = Y_train_p[shufflelist]
gamma = rff.gamma_est(X_train)
print(task + ' done')
### accurate KSVM

kscore = list()
ksparsity = list()
loglambda = np.empty(len(logclist))
for idx in range(len(logclist)):
    loglambda[idx] = -logclist[idx] - np.log10(samplesize)
    task = 'ksvm loglambda = {:1f}'.format(loglambda[idx])
    tasklist.append(task)
    timepoint.append(time.process_time())
    C = 10**logclist[idx]
    clf = svm.SVC(C=C,gamma=gamma)
    clf.fit(X_train,Y_train)
    kscore.append(clf.score(X_test,Y_test))
    ksparsity.append(clf.n_support_)
    print(task + ' done')
with open('result/ksvm_score'+fileprefix+'.csv','w',newline='') as csvfile:
    datawriter = csv.writer(csvfile)
    datawriter.writerow(kscore)

for n_components in [1,3,5,10,20]:
    print('N = ',n_components)
    ### rfsvm with optimized feature distribution and uniform distribution
    rfsvm_list = list()
    rfsvm_opt_list = list()
    opt_w_list = list()
    max_w_list = list()
    fileprefix = '{0:.1e} N {1}'.format(samplesize,n_components)
    feature_pool_size = n_components * 100
    m = samplesize

    for idx in range(trials):
        rfsvm_score = list()
        rfsvm_opt_score = list()
        opt_w = list()
        max_w = list()
        ### random features sampled uniformly
        for jdx in range(len(logclist)):
            task = 'trial {} unif feature gen'.format(idx)
            tasklist.append(task)
            timepoint.append(time.process_time())
            rbf_feature = rff.myRBFSampler(gamma=gamma,
                                           n_old_features=X_train.shape[1],
                                           n_components=n_components)
            X_train_til = rbf_feature.fit_transform(X_train)
            X_test_til = rbf_feature.fit_transform(X_test)
            print(task + ' done')
            Lambda = 10**loglambda[jdx]
            task = 'trial {0}, loglambda = {1:.1f}, unif, train'.format(idx,loglambda[jdx])
            tasklist.append(task)
            timepoint.append(time.process_time())
            rfsvm = SGDClassifier(loss='hinge',
                                  penalty='l2',
                                  alpha=Lambda,
                                  tol=10**(-5),
                                  max_iter=10**6 / m)
            rfsvm.fit(X_train_til,Y_train)
            print(task + ' done')
            task = 'trial {0}, loglambda = {1:.1f}, unif, test'.format(idx,loglambda[jdx])
            tasklist.append(task)
            timepoint.append(time.process_time())
            rfsvm_score.append(rfsvm.score(X_test_til,Y_test))
            print(task + ' done')
        ### random features sampled with optimized distribution
        for jdx in range(len(logclist)):
            opt_feature = rff.optRBFSampler(X_train.shape[1],
                                            feature_pool_size,
                                            gamma=gamma,
                                            n_components=n_components)
            Lambda = 10**loglambda[jdx]
            task = 'trial {} opt feature gen'.format(idx)
            tasklist.append(task)
            timepoint.append(time.process_time())
            opt_feature.reweight(X_train, X_pool_fraction, Lambda=Lambda)
            opt_w.append(np.sum(opt_feature.Weight) / feature_pool_size)
            max_w.append(max(opt_feature.Weight))
            X_train_til = opt_feature.fit_transform(X_train)
            X_test_til = opt_feature.fit_transform(X_test)
            print(task + ' done')
            task = 'trial {0}, loglambda = {1:.1f}, opt, train'.format(idx,loglambda[jdx])
            tasklist.append(task)
            timepoint.append(time.process_time())
            rfsvm_opt = SGDClassifier(loss='hinge',penalty='l2',alpha=Lambda,tol=10**(-5))
            rfsvm_opt.fit(X_train_til,Y_train)
            print(task + ' done')
            task = 'trial {0}, loglambda = {1:.1f}, opt, test'.format(idx,loglambda[jdx])
            tasklist.append(task)
            timepoint.append(time.process_time())
            rfsvm_opt_score.append(rfsvm_opt.score(X_test_til,Y_test))
            print(task + ' done')
        rfsvm_list.append(np.array(rfsvm_score))
        rfsvm_opt_list.append(np.array(rfsvm_opt_score))
        opt_w_list.append(np.array(opt_w))
        max_w_list.append(np.array(max_w))

    tasklist.append('end')
    timepoint.append(time.process_time())
    logcontent = str()
    for idx,value in enumerate(tasklist):
        if value != 'end':
            line = value + ' {:.4f}\n'.format(timepoint[idx+1]-timepoint[idx])
            logcontent = logcontent + line
        else:
            pass
    with open('log/rate_log'+fileprefix+'.txt','w') as logfile:
        logfile.write(logcontent)

    rfsvm_list = np.array(rfsvm_list)
    rfsvm_opt_list = np.array(rfsvm_opt_list)
    with open('result/unif_score'+fileprefix+'.csv','w',newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        datawriter.writerows(rfsvm_list)
    with open('result/opt_score'+fileprefix+'.csv','w',newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        datawriter.writerows(rfsvm_opt_list)

    max_w_list = np.array(max_w_list)
    opt_w_list = np.array(opt_w_list)
    with open('result/unif_w'+fileprefix+'.csv','w',newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        datawriter.writerows(max_w_list)
    with open('result/opt_w'+fileprefix+'.csv','w',newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        datawriter.writerows(opt_w_list)
    ### plot the performance vs different lambda
    rff.plot_circle(X_train,Y_train,ratio=50/samplesize)
    rfsvm_mean = np.sum(rfsvm_list,axis=0) / trials
    rfsvm_opt_mean = np.sum(rfsvm_opt_list,axis=0) / trials
    rfsvm_std = np.std(rfsvm_list,axis=0)
    rfsvm_opt_std = np.std(rfsvm_opt_list,axis=0)
    fig = plt.figure()
    plt.plot(loglambda,kscore,'r-o',fillstyle='none',label='ksvm')
    plt.errorbar(loglambda,rfsvm_mean,rfsvm_std,fmt='g:x',fillstyle='none',label='unif')
    plt.errorbar(loglambda,rfsvm_opt_mean,rfsvm_opt_std,fmt='b--s',fillstyle='none',label='opt')
    plt.yticks(np.arange(0.40,1.05,0.05))
    plt.legend(loc=3)
    plt.xlabel('$\log(\lambda)$')
    plt.ylabel('accuracy')
    plt.savefig('image/opt_results'+fileprefix+'.eps')
    plt.close(fig)

### plot the weight vs different lambda
max_mean = np.sum(max_w_list,axis=0) / trials
max_std = np.std(max_w_list,axis=0)
opt_mean = np.sum(opt_w_list,axis=0) / trials
opt_std = np.std(opt_w_list,axis=0)
fig = plt.figure()
plt.errorbar(loglambda,max_mean,max_std,fmt='r-o',fillstyle='none',label='max')
plt.errorbar(loglambda,opt_mean,opt_std,fmt='b--s',fillstyle='none',label='opt')
plt.legend(loc=1)
plt.xlabel('$\log(\lambda)$')
plt.ylabel('$d_{\max}(q,\lambda)$')
plt.savefig('image/weight'+fileprefix+'.eps')
plt.close(fig)
