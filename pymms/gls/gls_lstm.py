import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, CuDNNLSTM, Bidirectional, TimeDistributed

import keras.backend.tensorflow_backend as tfb
from keras import backend as K

num_features = 129
layer_size = 128

def weighted_binary_crossentropy(target, output):
    """
    Helper function for calculating the f1 score needed for importing the TF Keras model.

    Weighted binary crossentropy between an output tensor
    and a target tensor. POS_WEIGHT is used as a multiplier
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=7.395491055129609)
    return tf.reduce_mean(loss, axis=-1)


def f1(y_true, y_pred):
    """
    Helper function for calculating the f1 score needed for importing the TF Keras model.

    Args:
        y_true: A tensor with ground truth values.
        y_pred: A tensor with predicted truth values.

    Returns:
        A float with the f1 score of the two tensors.
    """

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def cpu_lstm():
    """
    Defines the CPU version of the mp-dl-unh model.
    """
    model = Sequential()

    model.add(
        Bidirectional(LSTM(layer_size, return_sequences=True), input_shape=(None, num_features)))

    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    opt = tf.keras.optimizers.Adam()

    model.compile(loss=weighted_binary_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy', f1, tf.keras.metrics.Precision()])

    return model


def gpu_lstm():
    """
    Defines the GPU version of the mp-dl-unh model.
    """
    model = Sequential()

    model.add(
        Bidirectional(CuDNNLSTM(layer_size, return_sequences=True), input_shape=(None, num_features)))

    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    opt = tf.keras.optimizers.Adam()

    model.compile(loss=weighted_binary_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy', f1, tf.keras.metrics.Precision()])

    return model