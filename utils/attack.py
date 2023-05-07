import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.context import Context


class FGSM:
    """
    We use FGSM to generate a batch of adversarial examples. 
    """
    def __init__(self, model, ep=0.01, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        self.time_start = time.time()

    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        ground_truths = tf.constant(y, dtype=float)

        original_xs = x.copy()
        original_prediction_labels = np.argmax(self.model(original_xs), axis=1)
        original_prediction_labels_one_hot = tf.constant(keras.utils.to_categorical(original_prediction_labels, len(y[0])), dtype=float)

        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0.0, 1.0)

        x = tf.Variable(x, dtype=float)
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(original_prediction_labels_one_hot, self.model(x))
            grads = tape.gradient(loss, x)
        delta = tf.sign(grads)

        x_aes = x + self.ep * delta
        x_aes = tf.clip_by_value(x_aes, clip_value_min=original_xs-self.ep, clip_value_max=original_xs+self.ep)
        x_aes = tf.clip_by_value(x_aes, clip_value_min=0, clip_value_max=1)

        ae_prediction_labels = np.argmax(self.model(x_aes), axis=1)
        idxs = np.where(ae_prediction_labels != original_prediction_labels)[0]
        print(f"The number of successful ae is {len(idxs)}, time {time.time()-self.time_start}")

        selected_original_xs, selected_x_aes, selected_ground_truths = original_xs[idxs], x_aes.numpy()[idxs], ground_truths.numpy()[idxs]
        time_list = []
        for _ in selected_x_aes:
            time_list.append(time.time() - self.time_start)
        return np.array(idxs), selected_x_aes, selected_ground_truths, np.array(time_list)


class PGD:
    """
    We use PGD to generate a batch of adversarial examples. PGD could be seen as iterative version of FGSM.
    """
    def __init__(self, model, ep=0.01, step=None, epochs=10, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        if step is None:
            self.step = ep/6
        self.epochs = epochs
        self.time_start = time.time()
        
    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        ground_truths = tf.constant(y, dtype=float)

        original_xs = x.copy()
        original_prediction_labels = np.argmax(self.model(original_xs), axis=1)
        original_prediction_labels_one_hot = tf.constant(keras.utils.to_categorical(original_prediction_labels, len(y[0])), dtype=float)

        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0.0, 1.0)

        x_aes = tf.Variable(x, dtype=float)
        for i in range(self.epochs):
            with tf.GradientTape() as tape:
                loss = keras.losses.categorical_crossentropy(original_prediction_labels_one_hot, self.model(x_aes))
                grads = tape.gradient(loss, x_aes)
            delta = tf.sign(grads)
            x_aes.assign_add(self.step * delta)
            x_aes = tf.clip_by_value(x_aes, clip_value_min=original_xs-self.ep, clip_value_max=original_xs+self.ep)
            x_aes = tf.clip_by_value(x_aes, clip_value_min=0.0, clip_value_max=1.0)
            x_aes = tf.Variable(x_aes)

        ae_prediction_labels = np.argmax(self.model(x_aes), axis=1)
        idxs = np.where(ae_prediction_labels != original_prediction_labels)[0]
        print(f"The number of successful ae is {len(idxs)}, time {time.time()-self.time_start}")

        selected_original_xs, selected_x_aes, selected_ground_truths = original_xs[idxs], x_aes.numpy()[idxs], ground_truths.numpy()[idxs]
        time_list = []
        for _ in selected_x_aes:
            time_list.append(time.time() - self.time_start)
        return np.array(idxs), selected_x_aes, selected_ground_truths, np.array(time_list)

