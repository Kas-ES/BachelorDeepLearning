import os

from skimage.transform import resize

import LoadData
import LoadModels
import Plotting
import ScoreCalculation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2
import numpy as np
from tensorflow.python.client import device_lib

import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from ttictoc import tic, toc

print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices("GPU")

testingPath = "Dataset_CCE/New_DataSet/df_test/*.png"
df_test = LoadData.load_data(testingPath)

print(df_test.describe())
print(df_test.shape)

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 256, 3]


def readAndsaveGroundTruth(amount, dataframe):
    array = np.array([])
    for i in range(amount):
        mask = cv2.imread(dataframe['mask'].iloc[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        mask = np.expand_dims(resize(mask,
                                     (IMG_HEIGHT, IMG_WIDTH),
                                     mode='constant',
                                     anti_aliasing=True,
                                     preserve_range=True), axis=-1)
        mask = mask / 255
        mask = np.squeeze(mask) > .5
        # mask[mask > 0.5] = 1
        # mask[mask <= 0.5] = 0
        if array.size == 0:
            array = np.array([mask])
        else:
            array = np.append(array, np.array([mask]), axis=0)
    return array


batchSIZE = 1

timeInferences = []
predictions = []
neural_networks = []


def model_prediction(type):
    if (type == "EUnets"):
        neural_networks = LoadModels.EUnets
    elif (type == "All"):
        neural_networks = LoadModels.AllNueralNetworks

    for neural_network in neural_networks:
        model = LoadModels.load_model(neural_network, type)

        test_gen = LoadData.train_generator(df_test, batchSIZE,
                                            dict(),
                                            target_size=(IMG_HEIGHT, IMG_WIDTH), shuffle=False)

        tic()
        Y_pred = model.predict(test_gen, steps=int(df_test.size / 2), verbose=1, batch_size=1)
        timeElapsed = toc()
        print(neural_network + " Predict Duration: " + str(timeElapsed))
        timeInferences.append(timeElapsed)
        predictions.append(Y_pred)


colors = ['red', 'black', 'orange', 'purple', 'pink', 'blue', 'green', 'cyan', 'grey', 'midnightblue', 'olive', 'lime',
          'yellow']

ground_truthTest = readAndsaveGroundTruth(int(df_test.size / 2), df_test)
ground_truthTest1D = np.ndarray.flatten(ground_truthTest)

'''All predicitons'''
labels = ['VGG19', 'ResNeXt50', 'ResNet50', 'InceptionV3', 'InceptionResnetV2', 'EU-Net', 'U-Net++', 'U-Net+++',
          'U-Net', 'EU-Net VGG19 5-4-1', 'EU-Net VGG19 5-3-2']
#model_prediction("All")

'''EUNET predicitons'''
# labels = ['EU-Net 5-5-5','EU-Net 5-5-4', 'EU-Net 5-4-4', 'EU-Net 5-5-3',
#            'EU-Net 5-5-2', 'EU-Net 5-4-3','EU-Net 4-5-5' ,'EU-Net 4-4-4',
#            'EU-Net 4-4-2', 'EU-Net 4-4-1','EU-Net 4-3-2'  ,'EU-Net 3-3-3']
# model_prediction("EUnets")


'''Image Visualization'''

def imageProcess (type):
    if (type == "EUnets"):
        neural_networks = LoadModels.EUnets
    elif (type == "All"):
        neural_networks = LoadModels.AllNueralNetworks

    for i in range(len(neural_networks)):
        model = LoadModels.load_model(neural_networks[i], type)
        Plotting.imageVisualiaziton(model, df_test, labels[i], IMG_HEIGHT, IMG_WIDTH)


imageProcess("All")

# Plotting.imageVisualiaziton(VGG19,df_test, labels[0], IMG_HEIGHT, IMG_WIDTH)
#
#
# Plotting.imageVisualiaziton(ResNextUnet,df_test, labels[1], IMG_HEIGHT, IMG_WIDTH)
# Plotting.imageVisualiaziton(Resnet,df_test, labels[2], IMG_HEIGHT, IMG_WIDTH)
# Plotting.imageVisualiaziton(InceptionV3,df_test, labels[3], IMG_HEIGHT, IMG_WIDTH)
# Plotting.imageVisualiaziton(InceptionResnetV2,df_test, labels[4], IMG_HEIGHT, IMG_WIDTH)
# Plotting.imageVisualiaziton(EUnet,df_test, labels[5], IMG_HEIGHT, IMG_WIDTH)
# Plotting.imageVisualiaziton(Unet2plus,df_test, labels[6], IMG_HEIGHT, IMG_WIDTH)
# Plotting.imageVisualiaziton(Unet3plus,df_test, labels[7], IMG_HEIGHT, IMG_WIDTH)
# Plotting.imageVisualiaziton(Unet,df_test, labels[8], IMG_HEIGHT, IMG_WIDTH)
# Plotting.imageVisualiaziton(EUnetVGG19_4_1,df_test, labels[9], IMG_HEIGHT, IMG_WIDTH)
# Plotting.imageVisualiaziton(EUnetVGG19_3_2,df_test, labels[10], IMG_HEIGHT, IMG_WIDTH)

'''Draw Precision Recall Curves'''
#Plotting.precision_recall_curve_models(predictions, ground_truthTest1D, labels, colors)

'''Scores'''
f1Scores = ScoreCalculation.f1scores_append(ground_truthTest1D, predictions)
print('F1-Score')
print(f1Scores)

precision_storage = ScoreCalculation.precision_append(ground_truthTest1D, predictions)
recall_storage = ScoreCalculation.recall_append(ground_truthTest1D, predictions)
print('Precision')
print(precision_storage)
print('Recall')
print(recall_storage)

iou = ScoreCalculation.ioU_score_append(ground_truthTest1D, predictions)
print('IoU')
print(iou)

'''Draw Circles'''
#Plotting.drawCircles(f1Scores, timeInferences, labels, colors)
