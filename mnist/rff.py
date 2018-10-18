import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class myRBFSampler:
    """
    The random nodes have the form
    cos(sqrt(gamma)*w dot x), sin(sqrt(gamma)*w dot x)
    """
    def __init__(self,n_old_features,gamma=1,n_components=20):
        self.name = 'rbf'
        self.sampler = np.random.randn(n_old_features,n_components)*np.sqrt(gamma)
        self.gamma = gamma
        self.n_components = n_components

    def update(self, idx):
        self.sampler[:,idx] = np.random.randn(self.sampler.shape[0])*np.sqrt(self.gamma)
        return 1

    def fit_transform(self, X):
        X_tilc = np.cos(X.dot(self.sampler))
        X_tils = np.sin(X.dot(self.sampler))
        X_til = np.concatenate((X_tilc,X_tils),axis=-1)
        return X_til / np.sqrt(self.n_components)

    def weight_estimate(self, X, X_pool_fraction, Lambda):
        m = len(X)
        X_pool_size = min(int(m * X_pool_fraction),500)
        n_components = self.n_components
        T = np.empty((X_pool_size,n_components*2))
        k = np.random.randint(m,size=X_pool_size)
        X_pool = X[k,:]
        A = X_pool.dot(self.sampler)
        T[:,:n_components] = np.cos(A)
        T[:,n_components:] = np.sin(A)
        U,s,V = np.linalg.svd(T, full_matrices=False)
        Trace = s**2 / (s**2 + Lambda * X_pool_size * n_components)
        Weight = np.empty(n_components*2)
        for idx in range(n_components*2):
            Weight[idx] = V[:,idx].dot(Trace * V[:,idx])
        Weight = Weight[:n_components] + Weight[n_components:]
        return Weight

class optRBFSampler:
    """
    The random nodes have the form
    (1/sqrt(q(w)))cos(sqrt(gamma)*w dot x), (1/sqrt(q(w)))sin(sqrt(gamma)*w dot x).
    q(w) is the optimized density of features with respect to the initial
    feature distribution determined only by the RBF kernel.
    """
    def __init__(self,
                 n_old_features,
                 feature_pool_size,
                 gamma=1,
                 n_components=20):
        self.name = 'opt_rbf'
        self.pool = (np.random.randn(n_old_features,
                                     feature_pool_size)
                    * np.sqrt(gamma))
        self.feature_pool_size = feature_pool_size
        self.gamma = gamma
        self.n_components = n_components
        Weight = np.ones(feature_pool_size)
        self.Weight = Weight
        self.Prob = Weight / np.sum(Weight)
        self.feature_list = np.random.choice(feature_pool_size,
                                size=n_components,
                                p=self.Prob)
        self.sampler = self.pool[:,self.feature_list]

    def reweight(self, X, X_pool_fraction, Lambda=1):
        ### calculate weight and resample the features from pool
        m = len(X)
        feature_pool_size = self.feature_pool_size
        X_pool_size = min(int(m * X_pool_fraction),500)
        T = np.empty((X_pool_size,feature_pool_size*2))
        k = np.random.randint(m,size=X_pool_size)
        X_pool = X[k,:]
        A = X_pool.dot(self.pool)
        T[:,:feature_pool_size] = np.cos(A)
        T[:,feature_pool_size:] = np.sin(A)
        U,s,V = np.linalg.svd(T, full_matrices=False)
        Trace = s**2 / (s**2 + Lambda * X_pool_size * feature_pool_size)
        Weight = np.empty(feature_pool_size*2)
        for idx in range(feature_pool_size*2):
            Weight[idx] = V[:,idx].dot(Trace * V[:,idx])
        Weight = Weight[:feature_pool_size] + Weight[feature_pool_size:]
        self.Weight = Weight
        self.Prob = Weight / np.sum(Weight)
        self.feature_list = np.random.choice(feature_pool_size,
                                             size=self.n_components,
                                             p=self.Prob)
        self.sampler = self.pool[:,self.feature_list]

    def update(self, idx):
        n = np.random.choice(self.pool.shape[1],p=self.Prob)
        self.sampler[:,idx] = self.pool[:,n]
        return 1

    def fit_transform(self, X):
        X_tilc = np.cos(X.dot(self.sampler))
        X_tils = np.sin(X.dot(self.sampler))
        X_til = np.concatenate((X_tilc,X_tils),axis=-1)
        return X_til / np.sqrt(self.n_components)

class myReLUSampler:
    """
    The random nodes have the form
    max(sqrt(gamma)*w dot x, 0)
    """
    def __init__(self,n_old_features,gamma=1,n_components=20):
        self.name = 'ReLU'
        self.sampler = np.random.randn(n_old_features,n_components)*np.sqrt(gamma)
        self.gamma = gamma
        self.n_components = n_components

    def update(self, idx):
        self.sampler[:,idx] = np.random.randn(self.sampler.shape[0])*np.sqrt(self.gamma)
        return 1

    def fit_transform(self, X):
        """
        It transforms one data vector a time
        """
        X_til = np.empty(self.n_components)
        for idx in range(self.n_components):
                X_til[idx] = max(X.dot(self.sampler[:,idx]),0)
        return X_til / np.sqrt(self.n_components)

class tfRF2L:
    """
    This is a class constructing a 2-layer net with cos and sin nodes
    in the hidden layer. The weights in the first layer is
    initialized using random Gaussian features.
    Layerwise training can be applied.
    """
    def __init__(self,n_old_features,
        n_components,Lambda,Gamma,classes,
        loss_fn='log loss',log=False):
        import tensorflow as tf
        self._d = n_old_features
        self._N = n_components
        self._Lambda = Lambda
        self._Gamma = Gamma
        self._classes = classes
        self._loss_fn = loss_fn
        self.log = log
        self._total_iter = 0
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        if self._model_fn() == 0:
            raise ValueError

    @property
    def d(self):
        return self._d
    @property
    def N(self):
        return self._N
    @property
    def Lambda(self):
        return self._Lambda
    @property
    def Gamma(self):
        return self._Gamma
    @property
    def classes(self):
        return self._classes
    @property
    def loss_fn(self):
        return self._loss_fn
    @property
    def total_iter(self):
        return self._total_iter

    def _model_fn(self):
        d = self._d
        N = self._N
        Lambda = self._Lambda
        Gamma = self._Gamma
        n_classes = len(self._classes)
        loss_fn = self._loss_fn

        with self._graph.as_default():
            global_step_1 = tf.Variable(0,trainable=False,name='global1')
            global_step_2 = tf.Variable(0,trainable=False,name='global2')
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,d],name='features')
            y = tf.placeholder(dtype=tf.uint8,
                shape=[None],name='labels')

            with tf.name_scope('RF_layer'):
                initializer = tf.random_normal_initializer(
                    stddev=tf.sqrt(Gamma))

                trans_layer = tf.layers.dense(inputs=x,units=N,
                    use_bias=False,
                    kernel_initializer=initializer,
                    name='Gaussian')

                cos_layer = tf.cos(trans_layer)
                sin_layer = tf.sin(trans_layer)
                concated = tf.concat([cos_layer,sin_layer],axis=1)
                RF_layer = tf.div(concated,tf.sqrt(N*1.0))
                tf.summary.histogram('inner weights',
                    self._graph.get_tensor_by_name('Gaussian/kernel:0'))

            if self._loss_fn == 'hinge loss':
                if n_classes == 2:
                    logits = tf.layers.dense(inputs=RF_layer,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Lambda),
                        units=1,name='Logits')
                else:
                    print("hinge loss only works for binary classificaiton.")
                    return 0
            elif self._loss_fn == 'log loss':
                logits = tf.layers.dense(inputs=RF_layer,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Lambda),
                    units=n_classes,name='Logits')
            tf.add_to_collection("Probab",logits)
            tf.summary.histogram('outer weights',
                self._graph.get_tensor_by_name('Logits/kernel:0'))

            probab = tf.nn.softmax(logits, name="softmax")
            tf.add_to_collection("Probab",probab)

            # hinge loss only works for binary classification.
            regularizer = tf.losses.get_regularization_loss(scope='Logits')
            if self._loss_fn == 'hinge loss':
                reg_loss = tf.losses.hinge_loss(labels=y,
                    logits=logits) + regularizer
            elif self._loss_fn == 'log loss':
                onehot_labels = tf.one_hot(indices=y, depth=n_classes)
                loss_log = tf.losses.softmax_cross_entropy(
                    onehot_labels=onehot_labels, logits=logits)
                reg_loss = tf.add(tf.reduce_mean(loss_log),regularizer)
            tf.add_to_collection('Loss',reg_loss)

            merged = tf.summary.merge_all()
            tf.add_to_collection('Summary',merged)
            self._sess.run(tf.global_variables_initializer())

        if self.log:
            summary = self._sess.run(merged)
            self._train_writer.add_summary(summary)
        return 1

    def predict(self,data):
        with self._graph.as_default():
            feed_dict = {'features:0':data}
            logits,probab = tf.get_collection('Probab')
            if self._loss_fn == 'hinge loss':
                if logits > 0:
                    index = 1
                else:
                    index = 0
                predictions = {
                    # Generate predictions (for PREDICT and EVAL mode)
                    "indices": index,
                    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                    # `logging_hook`.
                    "probabilities": probab}
            elif self._loss_fn == 'log loss':
                predictions = {
                    # Generate predictions (for PREDICT and EVAL mode)
                    "indices": tf.argmax(input=logits,axis=1),
                    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                    # `logging_hook`.
                    "probabilities": probab}

        results = self._sess.run(predictions,feed_dict=feed_dict)
        classes = [self._classes[index] for index in results['indices']]
        probabilities = results['probabilities']
        return classes,probabilities

    def score(self,data,labels):
        predictions,_ = self.predict(data)
        s = 0.
        for idx in range(len(data)):
            s += predictions[idx]==labels[idx]
        accuracy = s / len(data)
        return accuracy

    def fit(self,data,labels,mode='layer 2',
        batch_size=1,n_iter=1000):
        indices = [self._classes.index(label) for label in labels]
        indices = np.array(indices)
        with self._graph.as_default():
            in_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                'Gaussian')
            out_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                'Logits')
            loss = tf.get_collection('Loss')[0]
            global_step_1 = self._graph.get_tensor_by_name('global1:0')
            global_step_2 = self._graph.get_tensor_by_name('global2:0')
            merged = tf.get_collection('Summary')[0]
            if mode == 'layer 2':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=10)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50.,
                 #   l2_regularization_strength=0.)
                # optimizer = tf.train.AdamOptimizer(learning_rate=10.)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_2,
                    var_list=out_weights
                )
                # self._sess.run(tf.global_variables_initializer())
            if mode == 'layer 1':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=10)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50.,
                #    l2_regularization_strength=0.)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50,
                #     l2_regularization_strength=0.)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_1,
                    var_list=in_weights
                )
                # self._sess.run(tf.global_variables_initializer())
            if mode == 'over all':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=10)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50.,
                 #   l2_regularization_strength=0.)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50,
                #     l2_regularization_strength=0.)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_1,
                    )
                # self._sess.run(tf.global_variables_initializer())
            if self.log:
                self._train_writer = tf.summary.FileWriter('tmp',
                    tf.get_default_graph())

        for idx in range(n_iter):
            rand_list = np.random.randint(len(data),size=batch_size)
            feed_dict = {'features:0':data[rand_list,:],
                         'labels:0':indices[rand_list]}
            if idx % 10 == 1:
                if self.log:
                    print('iter: {0:d}, loss: {1:.4f}'.format(
                        idx, self._sess.run(loss,feed_dict)))
                    summary = self._sess.run(merged)
                    self._train_writer.add_summary(summary,self._total_iter)
            self._sess.run(train_op,feed_dict)
            self._total_iter += 1

    def get_params(self,deep=False):
        params = {
            'n_old_features': self._d,
            'n_components': self._N,
            'Lambda': self._Lambda,
            'Gamma': self._Gamma,
            'classes': self._classes,
            'loss_fn': self._loss_fn
        }
        return params

    def __del__(self):
        self._sess.close()
        print('Session is closed.')

def unit_interval(leftend,rightend,samplesize):
    if min(leftend,rightend)<0 or max(leftend,rightend)>1:
        print("The endpoints must be between 0 and 1!")
        return False
    X = list()
    Y = list()
    for idx in range(samplesize):
        x = np.random.random()
        X.append(x)
        if leftend>rightend:
            if x>rightend and x<leftend:
                Y.append(-1)
            else:
                Y.append(1)
        else:
            if x>leftend and x<rightend:
                Y.append(1)
            else:
                Y.append(-1)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def unit_circle(datarange,overlap,samplesize):
    if min(datarange,overlap)<0 or max(datarange,overlap)>1:
        print("The datarange and overlap values must be between 0 and 1!")
        return False
    X = list()
    Y = list()
    rad1upper = 1+datarange*overlap/2
    rad1lower = rad1upper-datarange
    rad2lower = 1-datarange*overlap/2
    rad2upper = rad2lower+datarange
    for idx in range(samplesize):
        if np.random.random()<0.5:
            Y.append(-1)
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(rad1lower,rad1upper)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
        else:
            Y.append(1)
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(rad2lower,rad2upper)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def unit_circle_ideal(gap,label_prob,samplesize):
    X = list()
    Y = list()
    rad1upper = 1 - gap/2
    rad2lower = 1 + gap/2
    for idx in range(samplesize):
        p = np.random.random()
        if p < 0.5:
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(0,rad1upper)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
            if p < 0.5*label_prob:
                Y.append(-1)
            else:
                Y.append(1)
        if p > 0.5:
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(rad1upper,2)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
            if p < 0.5 + 0.5*label_prob:
                Y.append(1)
            else:
                Y.append(-1)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def gamma_est(X,portion = 0.3):
    s = 0
    n = int(X.shape[0]*portion)
    if n > 200:
        n = 200
    for idx in range(n):
        for jdx in range(n):
            s = s+np.linalg.norm(X[idx,:]-X[jdx,:])**2
    return n**2/s

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
    A = np.array(X[0:m])
    Y = Y[0:m]
    c = list()
    for idx in range(m):
        if Y[idx]==1:
            c.append('r')
        else:
            c.append('b')
    fig = plt.figure()
    plt.scatter(A[:,0],A[:,1],c=c)
    circle = plt.Circle((0,0),1,fill=False)
    plt.gcf().gca().add_artist(circle)
    plt.axis('equal')
    plt.savefig('image/circle.eps')
    plt.close(fig)
    return 1
