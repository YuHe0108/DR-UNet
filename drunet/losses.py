import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf


def dice_loss(y_true, y_pred):
    def dice_coeff():
        smooth = 1
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_mean(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_mean(y_true_f) + tf.reduce_mean(y_pred_f) + smooth)
        return score

    return 1 - dice_coeff()


def bce_dice_loss(y_true, y_pred):
    losses = keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return losses


def focal_loss(y_true, y_pred, p_weight=0.75, n_weight=0.25, gamma=2., epsilon=1e-6):
    """
    focal loss used for train positive/negative samples rate out of balance, improve train performance
    """

    def _focal_loss():
        positive = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        negative = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -p_weight * K.pow(1. - positive, gamma) * K.log(
            positive + epsilon) - n_weight * K.pow(negative, gamma) * K.log(1. - negative + epsilon)

    return keras.backend.mean(_focal_loss())


def focal_dice_loss(y_true, y_pred):
    losses = 10 * dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)
    return losses


def loss_fun(model_name):
    model_name = str(model_name).lower()
    if model_name == 'my_model':
        return focal_dice_loss
    else:
        return tf.keras.losses.binary_crossentropy
