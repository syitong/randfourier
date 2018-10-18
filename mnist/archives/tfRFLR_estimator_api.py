import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import tensorflow as tf

def _RFLM(features,labels,mode,params):
    d = params['n_old_features']
    N = params['n_components']
    Lambda = params['lambda']
    Gamma = params['gamma']
    n_classes = params['n_classes']
    method = params['method']
    initializer = tf.random_normal_initializer(
        stddev=tf.sqrt(Gamma.astype(np.float32)))

    trans_layer = tf.layers.dense(inputs=features,units=N,
        use_bias=False,
        kernel_initializer=initializer,
        name='Gaussian')

    cos_layer = tf.cos(trans_layer)
    sin_layer = tf.sin(trans_layer)
    RF_layer = tf.div(tf.concat([cos_layer,sin_layer],axis=1), tf.sqrt(N*1.0))
    # in_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
    #    'RF_Layer')
    logits = tf.layers.dense(inputs=RF_layer,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Lambda),
        units=n_classes,name='logits')
    out_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        'logits')

    predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=logits, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    if method == 'svm':
        loss = tf.losses.hinge_loss(
            labels=labels,
            logits=logits,
        )
    else:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8),
            depth=n_classes)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.Variable(0,trainable=False)
        learning_rate = tf.train.inverse_time_decay(
            learning_rate=1.,
            decay_steps=1,
            global_step=global_step,
            decay_rate=1.)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.FtrlOptimizer(learning_rate=50,
            l2_regularization_strength=0.)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
            var_list=out_weights
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        
class tfRFLM(tf.estimator.Estimator):
    """
    This class implements the RFSVM with softmax + cross entropy
    as loss function by using Tensor Flow library
    to solve multi-class problems natively.
    """
    def __init__(self,parameters):
        super().__init__(
            model_fn=_RFLM,
            model_dir=None,
            params=parameters
        )
        self.parameters = parameters

    def fit(self,X,Y,n_iter=1000):
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X.astype(np.float32)},
            y=Y,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
        self.train(
            input_fn=train_input_fn,
            steps=n_iter)
        return 1

    def infer(self,X,predict_keys):
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":X.astype(np.float32)},
            num_epochs=1,
            shuffle=False
        )
        return self.predict(
            input_fn=pred_input_fn,
            predict_keys=predict_keys,
        )

    def score(self,X,Y):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x':X.astype(np.float32)},
            y=Y,
            num_epochs=1,
            shuffle=False
        )
        return self.evaluate(input_fn=eval_input_fn)['accuracy']

    def get_params(self,deep=False):
        return {"parameters": self.parameters}
