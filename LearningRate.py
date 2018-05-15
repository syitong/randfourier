"""
This code is for estimating the learning rate of RFSVM with
uniform feature sampling and optimized feature sampling.
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

fileprefix = ' trial {} '.format(argv[1])
logSMPlist = np.arange(2,6,0.5)
unif_best_score = list()
opt_best_score = list()
for kdx in range(len(logSMPlist)):
### set up experiment parameters
    logsamplesize = logSMPlist[kdx]
    logla_start = -6.0
    logla_end = -1
    logclist = np.arange(-logla_end - logsamplesize,
 	                 -logla_start - logsamplesize,0.5)
    samplesize = int(10**logsamplesize)

    ### set up feature parameters
    X_pool_fraction = 0.3
    n_components = int(logsamplesize**2 / 3) + 1
    feature_pool_size = n_components * 100

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
    loglambda = -logclist - np.log10(samplesize)

    ### rfsvm with optimized feature distribution and uniform distribution
    m = samplesize
    rfsvm_score = list()
    rfsvm_opt_score = list()
    opt_w = list()
    max_w = list()

    ### random features sampled uniformly
    for jdx in range(len(logclist)):
        task = 'unif feature gen'
        tasklist.append(task)
        timepoint.append(time.process_time())
        rbf_feature = rff.myRBFSampler(gamma=gamma,
                                       n_old_features=X_train.shape[1],
                                       n_components=n_components)
        X_train_til = rbf_feature.fit_transform(X_train)
        X_test_til = rbf_feature.fit_transform(X_test)
        print(task + ' done')
        Lambda = 10**loglambda[jdx]
        task = 'loglambda = {0:.1f}, unif, train'.format(loglambda[jdx])
        tasklist.append(task)
        timepoint.append(time.process_time())
        rfsvm = SGDClassifier(loss='hinge',
                              penalty='l2',
                              alpha=Lambda,
                              tol=10**(-5),
                              max_iter=10**6 / m)
        rfsvm.fit(X_train_til,Y_train)
        print(task + ' done in {} epochs'.format(rfsvm.n_iter_))
        task = 'loglambda = {0:.1f}, unif, test'.format(loglambda[jdx])
        tasklist.append(task)
        timepoint.append(time.process_time())
        rfsvm_score.append(rfsvm.score(X_test_til,Y_test))
        print(task + ' done')
    unif_best_score.append(np.max(rfsvm_score))

    ### random features sampled with optimized distribution
    for jdx in range(len(logclist)):
        opt_feature = rff.optRBFSampler(X_train.shape[1],
                                        feature_pool_size,
                                        gamma=gamma,
                                        n_components=n_components)
        Lambda = 10**loglambda[jdx]
        task = 'opt feature gen'
        tasklist.append(task)
        timepoint.append(time.process_time())
        opt_feature.reweight(X_train, X_pool_fraction, Lambda=Lambda)
        opt_w.append(np.sum(opt_feature.Weight) / feature_pool_size)
        max_w.append(max(opt_feature.Weight))
        X_train_til = opt_feature.fit_transform(X_train)
        X_test_til = opt_feature.fit_transform(X_test)
        print(task + ' done')
        task = 'loglambda = {0:.1f}, opt, train'.format(loglambda[jdx])
        tasklist.append(task)
        timepoint.append(time.process_time())
        rfsvm_opt = SGDClassifier(loss='hinge',penalty='l2',alpha=Lambda,tol=10**(-5))
        rfsvm_opt.fit(X_train_til,Y_train)
        print(task + ' done in {} epochs'.format(rfsvm_opt.n_iter_))
        task = 'loglambda = {0:.1f}, opt, test'.format(loglambda[jdx])
        tasklist.append(task)
        timepoint.append(time.process_time())
        rfsvm_opt_score.append(rfsvm_opt.score(X_test_til,Y_test))
        print(task + ' done')
    opt_best_score.append(np.max(rfsvm_opt_score))
    print('logsample {:.1} done'.format(logsamplesize))

tasklist.append('end')
timepoint.append(time.process_time())
logcontent = str()
for idx,value in enumerate(tasklist):
    if value != 'end':
        line = value + ' {:.4f}\n'.format(timepoint[idx+1]-timepoint[idx])
        logcontent = logcontent + line
    else:
        logcontent = logcontent + 'total {:.2f}\n'.format(timepoint[-1] - timepoint[0])
with open('log/learn_rate_log'+fileprefix+'.txt','w') as logfile:
    logfile.write(logcontent)
### save the best score among different lambdas in one trial
with open('result/unif_best_score'+fileprefix+'.csv','w',newline='') as csvfile:
    datawriter = csv.writer(csvfile)
    datawriter.writerow(unif_best_score)
with open('result/opt_best_score'+fileprefix+'.csv','w',newline='') as csvfile:
    datawriter = csv.writer(csvfile)
    datawriter.writerow(opt_best_score)
