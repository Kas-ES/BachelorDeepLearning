import os
from glob import glob

import sklearn
from natsort import natsorted

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
from tensorflow.python.client import device_lib

import tensorflow as tf
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from ttictoc import tic, toc

print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices("GPU")

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 256, 3]

# Training
df_merged_files = []
df_merged_mask_files = natsorted(glob('Dataset_CCE/Merged/*_mask*'))

for i in df_merged_mask_files:
    df_merged_files.append(i.replace('_mask', ''))

df_merged = pd.DataFrame(data={"filename": df_merged_files, 'mask': df_merged_mask_files})


# Testing
df_test_files = []
df_test_labels = []
df_test_mask_files = natsorted(glob('Dataset_CCE/SortedDataset/df_test/*_mask*'))

for i in df_test_mask_files:
    df_test_files.append(i.replace('_mask', ''))

index = 0
nopolyp = 'Dataset_CCE/SortedDataset/df_test\\NoPolyp'
_mask = '_mask'
_png = '.png'
comparator = nopolyp + str(index) + _mask + _png
for i in df_test_mask_files:
    comparator = nopolyp + str(index) + _mask + _png
    if i == comparator or i == nopolyp + str(nopolyp) + _png:
        df_test_labels.append("No Polyp")
    else:
        df_test_labels.append("Polyp")
    index += 1

df_test = pd.DataFrame(data={"filename": df_test_files, 'mask': df_test_mask_files})

print(df_test.describe())
print(df_test.head())
print(df_test.tail())
print(df_test.shape)
print(df_merged.shape)

inputs_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


def readAndsaveGroundTruth(amount, dataframe):
    array = np.array([])
    for i in range(amount):
        mask = cv2.imread(dataframe['mask'].iloc[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        if array.size == 0:
            array = np.array([mask])
        else:
            array = np.append(array, np.array([mask]), axis=0)
    return array

def convertToBinary_Flatten(array):
    array = np.where(array > 0.5, 1, 0)
    array = np.ndarray.flatten(array)
    return array

# ground_truthsMerged = readAndsaveGroundTruth(int(df_merged.size/2), df_merged)
# ground_truthsMerged = convertToBinary_Flatten(ground_truthsMerged)
# ground_truthMergedofOnes = np.count_nonzero(ground_truthsMerged == 1)
# print('Percentage of 1s in the overall ground truth masks is::: ', (ground_truthMergedofOnes / ground_truthsMerged.size) * 100)

ground_truthTest = readAndsaveGroundTruth(int(df_test.size/2), df_test)
ground_truthTest = convertToBinary_Flatten(ground_truthTest)
ground_truthTestofOnes = np.count_nonzero(ground_truthTest == 1)
print('Percentage of 1s in the overall ground truth masks is::: ', (ground_truthTestofOnes / ground_truthTest.size) * 100)

arrayExample = np.array([0,1,1,1,1,1,1,1,1,1])
arrayExampleTruth = np.count_nonzero(arrayExample == 1)
print('Percentage of 1s in the overall array Example is::: ', (arrayExampleTruth / arrayExample.size) * 100)

# 512 -- 256
def train_generator(data_frame, batch_size, aug_dict,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    class_mode=None,
                    save_to_dir=None,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col="filename",
        class_mode=class_mode,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask",
        class_mode=class_mode,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img, mask)


def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return (img, mask)


def imageVis(model):
    for i in range(10):
        index = np.random.randint(1, len(df_test.index))
        img = cv2.imread(df_test['filename'].iloc[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img = img / 255
        img = img[np.newaxis, :, :, :]
        pred = model.predict(img)

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(np.squeeze(img))
        plt.title('Original Image')
        plt.subplot(1, 3, 2)
        plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))
        plt.title('Original Mask')
        plt.subplot(1, 3, 3)
        plt.imshow(np.squeeze(pred) > .5)
        plt.title('Prediction')
        plt.show()


def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []

    for threshold in thresholds:
        ##y_pred = [True if score >= threshold else False for score in pred_scores]
        y_pred = [True if score >= threshold else False for score in pred_scores]
        ##y_pred1 = np.array(y_pred)
        ##y_pred1 = np.expand_dims(y_pred1,1)
        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='binary', pos_label=True)
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='binary', pos_label=True)

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def plotPrecision_Recall_Curve(recalls, precisions):
    plt.plot(recalls, precisions, linewidth=4, color="blue", zorder=0, label='Conventional')
    plt.scatter(recalls[8], precisions[8], zorder=1, linewidth=6)

    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.legend()
    plt.show()
    AP = np.sum((np.subtract(recalls[:-1], recalls[1:])) * precisions[:-1])
    print(AP)

def confusionMatrixAndClasificaitonReport(Y_pred, ground_truth, modelname):
    cmat = confusion_matrix(ground_truth, Y_pred)

    print(cmat)
    print(classification_report(ground_truth, Y_pred, target_names=['No Polyp', 'Polyp']))

    plt.figure(figsize=(6, 6))
    sns.heatmap(cmat, cmap="Reds", annot=True, square=1, linewidth=2.)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.title(modelname)
    plt.show()


smooth = 1


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
    return dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)


def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


batchSIZE = 1
learning_rate = 1e-4
epochs = 100

decay_rate = learning_rate / epochs
test_gen = train_generator(df_test, batchSIZE,
                           dict(),
                           target_size=(IMG_HEIGHT, IMG_WIDTH))

'''MODELS'''
# VGG19 = keras.models.load_model('100NoPolyp/VGG19net_100NoPolyp.hdf5', custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})
# ResNextUnet= keras.models.load_model('100NoPolyp/ResNextUnet_100NoPolyp.hdf5', custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})
# Resnet = keras.models.load_model('100NoPolyp/ResUnet_100NoPolyp.hdf5', custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})
# InceptionResnetV2 = keras.models.load_model('100NoPolyp/inceptionresnetv2_100NoPOLYP.hdf5', custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})
InceptionV3 = keras.models.load_model('100NoPolyp/Inceptionv3_100NoPOLYP.hdf5',custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou,'dice_coef': dice_coef})
#EUnet = keras.models.load_model('100NoPolyp/EUnet_100NoPolyp.hdf5',custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou,'dice_coef': dice_coef})


tic()
Y_pred = InceptionV3.predict(test_gen, steps=321, verbose=1)
timeElapsed = toc()
print(timeElapsed)

Y_pred = convertToBinary_Flatten(Y_pred)

# Thresholds
thresholds = np.arange(start=0.05, stop=0.95, step=0.05)

confusionMatrixAndClasificaitonReport(Y_pred, ground_truthTest, 'Inception_V3')
precisions, recalls = precision_recall_curve(ground_truthTest, Y_pred, thresholds)
plotPrecision_Recall_Curve(recalls, precisions)
