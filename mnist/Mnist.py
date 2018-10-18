"""
This code is used to solve the famous handwritten number
recognition problem via RFSVM.
"""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import rff
import log
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
import itertools
import time
from sys import argv

def read_MNIST_data(filepath,obs=1000):
    """
    This method is sufficiently general to read in any
    data set of the same structure with MNIST
    """
    with open(filepath,'rb') as f:
        data = f.read()
        offset = 0
        if data[2] < 10:
            length = 1
        elif data[2] < 12:
            length = 2
        elif data[2] < 14:
            length = 4
        elif data[2] < 15:
            length = 8
        dim = list()
        offset = offset + 4
        for idx in range(data[3]):
            dim.append(int.from_bytes(data[3+idx*4+1:3+idx*4+5],'big'))
            offset = offset + 4
        l = length
        for idx in range(1,len(dim)):
            l = l * dim[idx]
        if obs > 0 and obs < dim[0]:
            dim[0] = obs
        X = np.empty((dim[0],l))
        for idx in range(dim[0]):
            for jdx in range(l):
                index = offset + idx * l + jdx
                X[idx,jdx] = data[index]
        if l == 1:
            X = X[:,0]
        return X

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_train_test_data(train_size=-1,test_size=-1):
    # read in MNIST data set
    Xtrain = read_MNIST_data('data/train-images.idx3-ubyte',train_size)
    Ytrain = read_MNIST_data('data/train-labels.idx1-ubyte',train_size)
    Xtest = read_MNIST_data('data/t10k-images.idx3-ubyte',test_size)
    Ytest = read_MNIST_data('data/t10k-labels.idx1-ubyte',test_size)
    scaler = StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    return Xtrain,Ytrain,Xtest,Ytest

def ORFSVM_MNIST(m=1000,n_components=1000,trial_num=None):
    # set up timer and progress tracker
    mylog = log.log('log/ORFSVM_MNIST_Best_Perform_{:.2e}.log'.format(n_components),
    'ORFSVM MNIST classification starts')
    Xtrain,Ytrain,Xtest,Ytest = get_train_test_data(train_size=m,test_size=int(m/3))
    mylog.time_event('data read in complete')

    # set up parameters
    LogLambda = np.arange(-12.0,-2,1)
    Gamma = 10.**(-3)
    X_pool_fraction = 0.3
    feature_pool_size = n_components * 10

    # hyper-parameter selection
    best_score = 0
    best_Lambda = 1
    result = {'Gamma':[],'Lambda':[],'score':[]}
    for jdx in range(len(LogLambda)):
        Lambda = 10**LogLambda[jdx]
        opt_feature = rff.optRBFSampler(Xtrain.shape[1],
            gamma=Gamma,
            feature_pool_size=feature_pool_size,
            n_components=n_components)
        opt_feature.reweight(Xtrain,X_pool_fraction,Lambda=Lambda)
        mylog.time_event('Gamma={0:.1e} and Lambda={1:.1e}\n'.format(Gamma,Lambda)
                         +'features generated')
        Xtraintil = opt_feature.fit_transform(Xtrain)
        mylog.time_event('data transformed')
        # n_jobs is used for parallel computing 1 vs all;
        # -1 means all available cores
        clf = SGDClassifier(loss='hinge',penalty='l2',alpha=Lambda,
            tol=10**(-3),n_jobs=-1,warm_start=True)
        clf.fit(Xtraintil,Ytrain)
        mylog.time_event('training done')
        result['Gamma'].append(Gamma)
        result['Lambda'].append(Lambda)
        Xtesttil = opt_feature.fit_transform(Xtest)
        Ypred = clf.predict(Xtesttil)
        score = np.sum(Ypred == Ytest) / len(Ytest)
        print('score = {:.4f}'.format(score))
        mylog.time_event('testing done')
        result['score'].append(score)
        if score > best_score:
            best_score = score
            best_Gamma = Gamma
            best_Lambda = Lambda
            best_Sampler = opt_feature
            best_clf = clf

    # performance test
    # Xtesttil = best_Sampler.fit_transform(Xtest)
    # mylog.time_event('best model trained')
    # Ypred = best_clf.predict(Xtesttil)
    # C_matrix = confusion_matrix(Ytest,Ypred)

    # write results and log files
    classes = range(10)
    results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(results)
    results = results + 'Gamma    Lambda    score\n'
    for idx in range(len(result['Gamma'])):
        results = (results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(result['Gamma'][idx],
                                                         result['Lambda'][idx],
                                                         result['score'][idx]))
    mylog.record(results)
    mylog.save()
    return best_score

    # plot confusion matrix
    # fig = plt.figure()
    # plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    # plt.savefig('image/ORFSVM_MNIST_{}-cm.eps'.format(n_components))
    # plt.close(fig)

def URFSVM_MNIST(m=1000,n_components=1000):
    # set up timer and progress tracker
    mylog = log.log('log/URFSVM_MNIST_Best_Perform_{:.2e}.log'.format(n_components),
    'URFSVM MNIST classification starts')
    Xtrain,Ytrain,Xtest,Ytest = get_train_test_data(train_size=m,test_size=int(m/3))
    mylog.time_event('data read in complete')

    # set up parameters
    LogLambda = np.arange(-12.0,-2,1)
    Gamma = 10.**(-3.)

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 1
    best_Lambda = 1
    result = {'Gamma':[],'Lambda':[],'score':[]}
    for jdx in range(len(LogLambda)):
        Lambda = 10**LogLambda[jdx]
        unif_feature = rff.myRBFSampler(Xtrain.shape[1],
            gamma=Gamma,n_components=n_components)
        mylog.time_event('Gamma={0:.1e} and Lambda={1:.1e}\n'.format(Gamma,Lambda)
                         +'features generated')
        Xtraintil = unif_feature.fit_transform(Xtrain)
        mylog.time_event('data transformed')
        # n_jobs is used for parallel computing 1 vs all;
        # -1 means all available cores
        clf = SGDClassifier(loss='hinge',penalty='l2',alpha=Lambda,
            tol=10**(-3),n_jobs=-1,warm_start=True)
        clf.fit(Xtraintil,Ytrain)
        mylog.time_event('training done')
        result['Gamma'].append(Gamma)
        result['Lambda'].append(Lambda)
        Xtesttil = unif_feature.fit_transform(Xtest)
        Ypred = clf.predict(Xtesttil)
        score = np.sum(Ypred == Ytest) / len(Ytest)
        print('score = {:.4f}'.format(score))
        mylog.time_event('testing done')
        result['score'].append(score)
        if score > best_score:
            best_score = score
            best_Gamma = Gamma
            best_Lambda = Lambda
            best_Sampler = unif_feature
            best_clf = clf

    # performance test
    # Xtesttil = best_Sampler.fit_transform(Xtest)
    # Ypred = best_clf.predict(Xtesttil)
    # C_matrix = confusion_matrix(Ytest,Ypred)
    # mylog.time_event('test done')

    # write results and log files
    classes = range(10)
    output_results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(output_results)
    output_results = output_results + 'Gamma    Lambda    score\n'
    for idx in range(len(result['Gamma'])):
        output_results = (output_results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(result['Gamma'][idx],
                                                         result['Lambda'][idx],
                                                         result['score'][idx]))
    mylog.record(output_results)
    mylog.save()
    return best_score

    # plot confusion matrix
    # fig = plt.figure()
    # plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    # plt.savefig('image/URFSVM_MNIST_{}-cm.eps'.format(n_components))
    # plt.close(fig)

def KSVM_MNIST(m=1000,trainsize=1000):
    # set up timer and progress tracker
    mylog = log.log('log/KSVM_MNIST_{}.log'.format(trainsize),'KSVM MNIST classfication starts')

    # read in MNIST data set
    Xtr = read_MNIST_data('data/train-images.idx3-ubyte',trainsize)
    Ytr = read_MNIST_data('data/train-labels.idx1-ubyte',trainsize)
    Xtest = read_MNIST_data('data/t10k-images.idx3-ubyte',-1)
    Ytest = read_MNIST_data('data/t10k-labels.idx1-ubyte',-1)
    mylog.time_event('data read in complete')

    # extract a smaller data set
    Xtrain = Xtr[:m]
    Ytrain = Ytr[:m]
    scaler = StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtr = scaler.transform(Xtr)
    Xtest = scaler.transform(Xtest)

    # set up parameters
    LogLambda = np.arange(-12.0,-2,1)
    gamma = rff.gamma_est(Xtrain)
    LogGamma = np.arange(-0.2,0.8,0.1)
    LogGamma = np.log10(gamma) + LogGamma
    cv = 5 # cross validation folds

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 1
    best_Lambda = 1
    crossval_result = {'Gamma':[],'Lambda':[],'score':[]}
    # for idx in range(len(LogGamma)):
    #     Gamma = 10**LogGamma[idx]
    #     for jdx in range(len(LogLambda)):
    #         Lambda = 10**LogLambda[jdx]
    #         C = 1 / Lambda / ((cv - 1) * m / cv)
    #         clf = svm.SVC(C=C,gamma=Gamma)
    #         score = cross_val_score(clf,Xtrain,Ytrain,cv=cv,n_jobs=-1)
    #         mylog.time_event('Gamma={0:.1e} and Lambda={1:.1e}\n'.format(Gamma,Lambda)
    #                          +'crossval done')
    #         crossval_result['Gamma'].append(Gamma)
    #         crossval_result['Lambda'].append(Lambda)
    #         avg_score = np.sum(score) / 5
    #         print('score = {:.4f}'.format(avg_score))
    #         crossval_result['score'].append(avg_score)
    #         if avg_score > best_score:
    #             best_score = avg_score
    #             best_Gamma = Gamma
    #             best_Lambda = Lambda

    best_Lambda = 10**LogLambda[0]
    best_Gamma = 10**LogGamma[1]
    # performance test
    C = 1 / best_Lambda / len(Xtr)
    best_clf = svm.SVC(C=C,gamma=best_Gamma)
    best_clf.fit(Xtr,Ytr)
    mylog.time_event('best model trained')
    Ypred = best_clf.predict(Xtest)
    C_matrix = confusion_matrix(Ytest,Ypred)
    score = np.sum(Ypred == Ytest) / len(Ytest)
    mylog.time_event('test done')

    # write results and log files
    classes = range(10)
    results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(results)
    results = results + 'Gamma    Lambda    score\n'
    for idx in range(len(crossval_result['Gamma'])):
        results = (results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(crossval_result['Gamma'][idx],
                                                         crossval_result['Lambda'][idx],
                                                         crossval_result['score'][idx]))
    mylog.record(results)
    mylog.save()

    # plot confusion matrix
    fig = plt.figure()
    plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    plt.savefig('image/KSVM_MNIST_{}-cm.eps'.format(trainsize))
    plt.close(fig)

def tfURF2L_MNIST(m=1000,n_components=1000):
    # set up timer and progress tracker
    mylog = log.log('log/tfnnRF2Ldropout_MNIST_{}.log'.format(n_components),'MNIST classification starts')

    # read in MNIST data set
    Xtr = read_MNIST_data('data/train-images.idx3-ubyte',-1)
    Ytr = read_MNIST_data('data/train-labels.idx1-ubyte',-1)
    Xtest = read_MNIST_data('data/t10k-images.idx3-ubyte',-1)
    Ytest = read_MNIST_data('data/t10k-labels.idx1-ubyte',-1)
    mylog.time_event('data read in complete')

    # extract a smaller data set
    Xtrain = Xtr[:m]
    Ytrain = Ytr[:m]
    scaler = StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtr = scaler.transform(Xtr)
    Xtest = scaler.transform(Xtest)

    # set up parameters
    LogLambda = np.arange(-12.0,-2,1.)
    gamma = rff.gamma_est(Xtrain)
    LogGamma = np.arange(-0.2,0.8,.1)
    LogGamma = np.log10(gamma) + LogGamma
    params = {
        'n_old_features': len(Xtrain[0]),
        'n_components': n_components,
        # 'Lambda': np.float32(10.**(-6)),
        'Lambda': np.float32(0.),
        'Gamma': np.float32(10.**LogGamma[2]),
        'classes': [0,1,2,3,4,5,6,7,8,9],
    }
    fit_params = {
        'mode': 'layer 2',
        'batch_size': 1,
        'n_iter': 5000
    }

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 10.**LogGamma[2]
    # best_Lambda = 10.**(-6)
    best_Lambda = 0.
    crossval_result = {'Gamma':[],'Lambda':[],'score':[]}
    # for idx in range(len(LogGamma)):
    #     Gamma = np.float32(10**LogGamma[idx])
    #     for jdx in range(len(LogLambda)):
    #         Lambda = np.float32(10**LogLambda[jdx])
    #         params['Lambda'] = Lambda
    #         params['Gamma'] = Gamma
    #         clf = rff.tfRF2L(**params)
    #         score = cross_val_score(clf,Xtrain,Ytrain,fit_params=fit_params,cv=5)
    #         mylog.time_event('Gamma={0:.1e} and Lambda={1:.1e}\n'.format(Gamma,Lambda)
    #                          +'crossval done')
    #         crossval_result['Gamma'].append(Gamma)
    #         crossval_result['Lambda'].append(Lambda)
    #         avg_score = np.sum(score) / 5
    #         print('score = {:.4f}'.format(avg_score))
    #         crossval_result['score'].append(avg_score)
    #         if avg_score > best_score:
    #             best_score = avg_score
    #             best_Gamma = Gamma
    #             best_Lambda = Lambda
    #             best_clf = clf

    # performance test
    best_clf = rff.tfRF2L(**params)
    best_clf.log = True
    # best_clf.fit(Xtr,Ytr,mode='layer 2',batch_size=100,n_iter=7000)
    # best_clf.fit(Xtr,Ytr,mode='layer 1',batch_size=100,n_iter=7000)
    best_clf.fit(Xtr,Ytr,mode='over all',batch_size=100,n_iter=7000)
    mylog.time_event('best model trained')
    Ypred,_ = best_clf.predict(Xtest)
    C_matrix = confusion_matrix(Ytest,Ypred)
    score = np.sum(Ypred == Ytest) / len(Ytest)
    mylog.time_event('test done')

    # write results and log files
    classes = range(10)
    results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(results)
    results = results + 'Gamma    Lambda    score\n'
    for idx in range(len(crossval_result['Gamma'])):
        results = (results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(crossval_result['Gamma'][idx],
                                                         crossval_result['Lambda'][idx],
                                                         crossval_result['score'][idx]))
    mylog.record(results)
    mylog.save()

    # plot confusion matrix
    fig = plt.figure()
    plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    plt.savefig('image/tfnnRF2Ldropout_MNIST_{}-cm.eps'.format(n_components))
    plt.close(fig)

def main():
    prefix = argv[1]
    # URFSVM_MNIST(m=1000,n_components=500)
    # ORFSVM_MNIST(m=1000,n_components=500)
    uscore_list = []
    oscore_list = []
    for m in range(1000,60001,5000):
        score = URFSVM_MNIST(m=m,n_components=int(np.sqrt(m)))
        uscore_list.append(score)
        score = ORFSVM_MNIST(m=m,n_components=int(np.sqrt(m)))
        oscore_list.append(score)
    np.savetxt('result/URFSVM'+str(prefix),np.array(uscore_list))
    np.savetxt('result/ORFSVM'+str(prefix),np.array(oscore_list))
    # KSVM_MNIST(m=1000,trainsize=60000)
    # URFMLR_MNIST(m=1000,n_components=2000)
    # tfRFLM_MNIST(m=1000,n_components=2000)
    # tfURF2L_MNIST(m=1000,n_components=2000)

if __name__ == '__main__':
    main()
