import os

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

from tensorflow.keras import backend as K

thresHold = 0.5

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return abs(dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred))


def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def f1scores_append(y_true, predictions):
    f1Scores = []
    for prediction in predictions:
        prediction = np.where(prediction > thresHold, 1, 0)
        prediction = np.ndarray.flatten(prediction)
        f1Scores.append(f1_score(y_true, prediction))

    return f1Scores


def precision_append(y_true, predictions):
    precision_storage = []
    for prediction in predictions:
        prediction = np.where(prediction > thresHold, 1, 0)
        prediction = np.ndarray.flatten(prediction)
        precision_storage.append(precision_score(y_true, prediction))

    return precision_storage


def recall_append(y_true, predictions):
    recall_storage = []
    for prediction in predictions:
        prediction = np.where(prediction > thresHold, 1, 0)
        prediction = np.ndarray.flatten(prediction)
        recall_storage.append(recall_score(y_true, prediction))

    return recall_storage


def ioU_score_append(y_true, predictions):
    iou = []
    for prediction in predictions:
        prediction = np.where(prediction > thresHold, 1, 0)
        prediction = np.ndarray.flatten(prediction)
        iou.append(jaccard_score(y_true, prediction))

    return iou
