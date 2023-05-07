import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.losses import categorical_crossentropy


def mixup_data(x, y, alpha):
    """Mix data
    x: input numpy array.
    y: target numpy array.
    alpha: float.
    Paper url: https://arxiv.org/pdf/1710.09412.pdf
    Return
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, len(x))
    else:
        lam = np.array([1.0] * len(x))

    indexes = np.random.permutation(range(len(x)))
    mixed_x = []
    for i in range(len(x)):
        mixed_x.append(x[i] * lam[i] + (1 - lam[i]) * x[indexes[i]])
    y_a = np.reshape(y, (-1, 1))
    y_b = np.reshape(y[indexes], (-1, 1))
    lam = np.reshape(lam, (-1, 1))
    return np.array(mixed_x).reshape(x.shape), y_a, y_b, lam, indexes


def mixup_data_variant(x, y, alpha):
    """Mix data
    x: input numpy array.
    y: target numpy array.
    alpha: float.
    Paper url: https://arxiv.org/pdf/1710.09412.pdf
    Return
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    indexes = np.random.permutation(range(len(x)))
    mixed_x = lam * x + (1 - lam) * x[indexes, :]
    y_a = np.reshape(y, (-1, 1))
    y_b = np.reshape(y[indexes], (-1, 1))
    lam = np.reshape(np.array([lam]*len(x)), (-1, 1))
    return mixed_x, y_a, y_b, lam, indexes


def mixup_data_one_sample(x1, y1, x2, y2, alpha):
    """Mix data on one sample.
    x1: input numpy array.
    y1: target numpy array.
    lam: lambda. [0, 1]
    Paper url: https://arxiv.org/pdf/1710.09412.pdf
    Return
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = x1 * lam + x2 * (1.0 - lam)
    return mixed_x, y1, y2, lam


# loss function
def compute_loss(y_true, y_pred):
    try:
        y_a, y_b, lam = tf.split(y_true, 3, 1)
        y_a = tf.cast(y_a, dtype=tf.int32)
        y_b = tf.cast(y_b, dtype=tf.int32)
        y_a = tf.reshape(y_a, (-1,))
        y_b = tf.reshape(y_b, (-1,))
        lam = tf.reshape(lam, (-1,))
        y_a = K.one_hot(y_a, 10)
        y_b = K.one_hot(y_b, 10)
        return tf.add(tf.math.multiply(lam, categorical_crossentropy(y_a, y_pred)), tf.math.multiply((1 - lam), categorical_crossentropy(y_b, y_pred)))
    except:
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.reshape(y_true, (-1,))
        y_true = K.one_hot(y_true, 10)
        return categorical_crossentropy(y_true, y_pred)


def compute_loss_100(y_true, y_pred):
    try:
        y_a, y_b, lam = tf.split(y_true, 3, 1)
        y_a = tf.cast(y_a, dtype=tf.int32)
        y_b = tf.cast(y_b, dtype=tf.int32)
        y_a = tf.reshape(y_a, (-1,))
        y_b = tf.reshape(y_b, (-1,))
        lam = tf.reshape(lam, (-1,))
        y_a = K.one_hot(y_a, 100)
        y_b = K.one_hot(y_b, 100)
        return tf.add(tf.math.multiply(lam, categorical_crossentropy(y_a, y_pred)), tf.math.multiply((1 - lam), categorical_crossentropy(y_b, y_pred)))
    except:
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.reshape(y_true, (-1,))
        y_true = K.one_hot(y_true, 100)
        return categorical_crossentropy(y_true, y_pred)


def base_compute_loss(y_true, y_pred, num_classes):
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_true = K.one_hot(y_true, num_classes)
    return categorical_crossentropy(y_true, y_pred)


def compute_acc(y_true, y_pred):
    try:
        y_a, y_b, lam = tf.split(y_true, 3, 1)
        y_a = tf.cast(y_a, tf.int64)
        y_b = tf.cast(y_b, tf.int64)
        y_a = tf.reshape(y_a, (-1,))
        y_b = tf.reshape(y_b, (-1,))
        y_pred = tf.argmax(y_pred, axis=-1)
        return lam * accuracy(y_a, y_pred) + (1 - lam) * accuracy(y_b, y_pred)
    except:
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.argmax(y_pred, axis=1)
        return accuracy(y_true, y_pred)

