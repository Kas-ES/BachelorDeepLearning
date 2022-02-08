import os
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Add, Concatenate, MaxPool2D, Dropout

import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.python.client import device_lib
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Activation, BatchNormalization
from tensorflow.python.keras.models import Model

from keras.models import Model
from tensorflow import keras
from tensorflow.python.keras.applications.densenet import layers
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator

import Models

Models.ressUnet()
print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices("GPU")

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [512, 512, 3]


train_files = []
mask_files = glob('Dataset_CCE/Merged/*_mask*')

for i in mask_files:
    train_files.append(i.replace('_mask', ''))

print(len(train_files))
print(len(mask_files))

df = pd.DataFrame(data={"filename": train_files, 'mask': mask_files})
df_train, df_test = train_test_split(df, test_size=0.1)
df_train, df_val = train_test_split(df_train, test_size=0.2)

print(df_train.shape)
print(df_test.shape)
print(df_val.shape)

print(df_train)

print(df_train.iloc[1000]["filename"])

inputs_size = input_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# From: https://github.com/zhixuhao/unet/blob/master/data.py
def train_generator(data_frame, batch_size, aug_dict,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    save_to_dir=None,
                    target_size=(IMG_CHANNELS, IMG_WIDTH),
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
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask",
        class_mode=None,
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
    return dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)


def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

epochs = 10
batchSIZE = 5
learning_rate = 1e-4

train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

decay_rate = learning_rate / epochs
opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate,
                            amsgrad=False)

train_gen = train_generator(df_train, batchSIZE, train_generator_args,
                            target_size=(IMG_HEIGHT, IMG_WIDTH))

valid_generator = train_generator(df_val, batchSIZE,
                                  dict(),
                                  target_size=(IMG_HEIGHT, IMG_WIDTH))

model = Models.ressUnet()
model.summary()

model.compile(loss=bce_dice_loss, optimizer=opt,
              metrics=['binary_accuracy', dice_coef, iou])

history = model.fit(train_gen,
                    steps_per_epoch=len(df_train) / batchSIZE,
                    epochs=epochs,
                    validation_data=valid_generator,
                    validation_steps=len(df_val) / batchSIZE, verbose=2)