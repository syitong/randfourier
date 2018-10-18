import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class HRFSVM_binary:
    """
    This class implements the RFSVM with random drop out for each round
    of subgrad descent.
    """
    def __init__(self,n_old_features,n_components=20,gamma=1,p=0,
        alpha=0,max_iter=5,tol=10**(-3)):
        self.sampler = myRBFSampler(n_old_features=n_old_features,
            n_components=n_components,gamma=gamma)
        # self.feature_type = self.sampler.name
        self.classes_ = [1,-1]
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.w = np.zeros(2 * self.sampler.n_components)
        # if self.feature_type == 'rbf':
        #     self.w = np.zeros(2*self.sampler.n_components)
        # else:
        #     self.w = np.zeros(self.sampler.n_components)

    def fit(self,X,Y):
        """
        We run the cyclic subgradient descent. cycle is the number of
        repeats of the cycles of the dataset.
        """
        self.classes_ = list(set(Y))
        if self.classes_ != [-1,1] and self.classes_ != [1,-1]:
            for idx in range(len(Y)):
                if Y[idx] == self.classes_[0]:
                    Y[idx] = 1
                elif Y[idx] == self.classes_[1]:
                    Y[idx] = -1
        n = len(Y)
        T = 0
        score = [1000]
        for idx in range(self.max_iter):
            jlist = np.random.permutation(n)
            for jdx in range(n):
                T = jdx+idx*n+1
                score.append(self.partial_fit(X[jlist[jdx]],Y[jlist[jdx]],T))
                if len(score) > 1:
                    if score[-2] - score[-1] < self.tol:
                        break
            if len(score) > 1:
                if score[-2] - score[-1] < self.tol:
                    break
        return score

    def partial_fit(self,Xrow,y,T):
        if np.random.rand() < self.p:
            n_components = self.sampler.n_components
            w_norm = np.empty(n_components)
            if self.sampler.name == 'rbf':
                for idx in range(n_components):
                    w_norm[idx] = self.w[idx]**2+self.w[idx+n_components]**2
                update_idx = np.argmin(w_norm)
                self.sampler.update(update_idx)
                self.w[update_idx] = 0
                self.w[update_idx+n_components] = 0
            else:
                for idx in range(n_components):
                    w_norm[idx] = np.abs(self.w[idx])
                update_idx = np.argmin(w_norm)
                self.sampler.update(update_idx)
                self.w[update_idx] = 0
        Xrow_til = self.sampler.fit_transform(Xrow)
        score = max(1 - np.dot(Xrow_til,self.w.T)*y,0)
        if score > 0:
            if self.alpha == 0:
                self.w = self.w + y*Xrow_til/np.sqrt(T)
            else:
                self.w = (1-1/T)*self.w + y*Xrow_til/T/self.alpha
        else:
            if self.alpha == 0:
                self.w = self.w
            else:
                self.w = (1-1/T)*self.w
        score = max(1 - np.dot(Xrow_til,self.w.T)*y,0)
        return score

    def predict(self,X):
        output = []
        X_til = self.sampler.fit_transform(X)
        if X_til.dot(self.w.T) > 0:
            output.append(self.classes_[0])
        else:
            output.append(self.classes_[1])
        return np.array(output)

class HRFSVM:
    def __init__(self,n_components=20,gamma=1,p=0,
        alpha=0,max_iter=5,tol=10**(-3),n_jobs=-1):
        self.n_old_features = 0
        self.n_components = 20
        self.gamma = 1
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.estimator = []
        self.classes_ = []

    def _fit_binary(self,X,Y):
        estimator = HRFSVM_binary(n_old_features=self.n_old_features,
            n_components=self.n_components,gamma=self.gamma,
            p=self.p,alpha=self.alpha,max_iter=self.max_iter,
            tol=self.tol)
        estimator.fit(X,Y)
        return estimator

    def fit(self,X,Y):
        self.n_old_features = len(X[0])
        self.classes_ = list(set(Y))
        if len(self.classes_) > 2:
            Ycopy = np.empty((len(self.classes_),len(Y)))
            for idx,val in enumerate(self.classes_):
                for jdx,label in enumerate(Y):
                    if label == val:
                        Ycopy[idx,jdx] = 1
                    else:
                        Ycopy[idx,jdx] = -1
            self.estimator = Parallel(n_jobs=self.n_jobs,backend="threading")(
                delayed(self._fit_binary)(X,Ycopy[idx])
                for idx in range(len(self.classes_)))
            return 1

        elif len(self.classes_) == 2:
            for idx in range(len(Y)):
                if Y[idx] == self.classes_[0]:
                    Y[idx] = 1
                elif Y[idx] == self.classes_[1]:
                    Y[idx] = -1
            self.estimator = self._fit_binary(X,Y)
            return 1

    def predict(self,X):
        if len(self.classes_) > 2:
            output = []
            for idx in range(len(X)):
                score = 0
                label = self.classes_[0]
                for jdx,val in enumerate(self.classes_):
                    X_til = self.estimator[jdx].sampler.fit_transform(X[idx])
                    s = X_til.dot(self.estimator[jdx].w.T)
                    if score < s:
                        score = s
                        label = val
                output.append(label)
            return output
        elif len(self.classes_) == 2:
            X_til = self.estimator.sampler.fit_transform(X)
            score = X_til.dot(self.estimator.w.T)
            output = []
            for idx in range(len(X)):
                if score[idx] > 0:
                    output.append(self.classes_[0])
                else:
                    output.append(self.classes_[1])
            return output

    def get_params(self,deep=False):
        return {'n_components': self.n_components,
                'gamma': self.gamma,
                'p': self.p,
                'alpha': self.alpha,
                'max_iter': self.max_iter,
                'tol': self.tol,
                'n_jobs': self.n_jobs}
                
def dim_modifier(X,dim,method='const'):
    if method == 'const':
        Tail = np.ones((X.shape[0],dim))
        Xtil = np.concatenate((X,Tail),axis=-1)
        return Xtil
    else:
        Tail = np.random.randn(X.shape[0],dim)
        Xtil = np.concatenate((X,Tail),axis=-1)
        return Xtil
