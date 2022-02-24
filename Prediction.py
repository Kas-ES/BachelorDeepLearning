import os
from glob import glob

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
from skimage.io import imread, imsave
from skimage.transform import resize


print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices("GPU")

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 256, 3]


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
comparator = nopolyp+str(index)+_mask+_png
for i in df_test_mask_files:
    comparator = nopolyp + str(index) + _mask + _png
    if i == comparator or i == nopolyp+str(nopolyp)+_png:
        df_test_labels.append("No Polyp")
    else:
        df_test_labels.append("Polyp")
    index += 1

df_test = pd.DataFrame(data={"filename": df_test_files, 'mask': df_test_mask_files, 'label': df_test_labels}, index=[df_test_labels])

print(df_test.describe())
print(df_test.head())
print(df_test.tail())
print(df_test.shape)

inputs_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

ground_truth = np.array([])

for i in range(321):
  mask = cv2.imread(df_test['mask'].iloc[i])
  mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
  mask = mask / 255
  mask[mask > 0.5] = 1
  mask[mask <= 0.5] = 0
  if ground_truth.size == 0:
    ground_truth = np.array([mask])
  else:
      ground_truth = np.append(ground_truth, np.array([mask]), axis=0)


#print("GROUND TRUTH SIZE:::", ground_truth.size)
#print("GROUND TRUTH::::",ground_truth.shape)
#ground_truth = np.argmax(ground_truth, axis=2)
#print("ARGMAX::::", ground_truth.shape)
#ground_truth = np.reshape(ground_truth, (321,256, 1))
#print("ARGMAX::::", ground_truth.shape)

ground_truth = np.ndarray.flatten(ground_truth)
print(ground_truth.round(2))

ground_truth = np.where(ground_truth > 0.5, 1, 0)

####
aug_dict = None

image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()


image_generator = image_datagen.flow_from_dataframe(
    df_test,
    x_col="filename",
    y_col='label',
    classes=['No Polyp', 'Polyp'],
    class_mode='binary',
    color_mode='rgb',
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=1,
    save_to_dir=None,
    save_prefix='"image"',
    shuffle=False,
    seed=1)

mask_generator = mask_datagen.flow_from_dataframe(
    df_test,
    x_col="mask",
    y_col='label',
    classes=['No Polyp', 'Polyp'],
    class_mode='binary',
    color_mode='grayscale',
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=1,
    save_to_dir=None,
    save_prefix='mask',
    shuffle=False,
    seed=1)

# 512 -- 256
def train_generator(image_generator, mask_generator):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    #image_datagen = ImageDataGenerator(**aug_dict)
    #mask_datagen = ImageDataGenerator(**aug_dict)

    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img, mask)


def adjust_data(img, mask):
    img = img[0]
    mask = mask[0]

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

decay_rate = learning_rate/epochs
test_gen = train_generator(image_generator, mask_generator)



modelincresnet = keras.models.load_model('inceptionresnetv2_POLYP.hdf5', custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})

#modelRES = keras.models.load_model('ResUnet_POLYP.hdf5', custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})

#modelINC = keras.models.load_model('Inceptionv3_POLYP.hdf5',
                      #custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})


Y_pred = modelincresnet.predict(test_gen, steps=321 ,verbose=1)

wtf = mask_generator.classes

print("WWWWW::::",wtf)

print('Y_pred:::',Y_pred.shape)

Y_pred = np.ndarray.flatten(Y_pred)
print(Y_pred.round(2))

y_pred = np.where(Y_pred >0.5, 1, 0)
#
# y_pred = np.argmax(Y_pred, axis=2)
# y_predArray = np.argmax(y_pred, axis=2)
# y_predArray = np.argmax(y_predArray, axis=1)
# y_predList = list(y_predArray)
# #print('y_pred::::', y_pred)

target_names = ['No Polyp', 'Polyp']
print("CONFUSION MATRIX")
print(confusion_matrix(ground_truth , y_pred))
print('Classification Report')
print(classification_report(ground_truth, y_pred, target_names=target_names))
