import os
from glob import glob

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

print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices("GPU")

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 256, 3]


# Testing
df_test_files = []
df_test_mask_files = glob('Dataset_CCE/SortedDataset/df_test/*_mask*')

for i in df_test_mask_files:
    df_test_files.append(i.replace('_mask', ''))

df_test = pd.DataFrame(data={"filename": df_test_files, 'mask': df_test_mask_files})


print(df_test.shape)

inputs_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


# 512 -- 256
def train_generator(data_frame, batch_size, aug_dict,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
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
test_gen = train_generator(df_test, batchSIZE,
                           dict(),
                           target_size=(IMG_HEIGHT, IMG_WIDTH))



modelVGG = keras.models.load_model('VGGU19net_POLYP.hdf5', custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})

#modelRES = keras.models.load_model('ResUnet_POLYP.hdf5', custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})

#modelINC = keras.models.load_model('Inceptionv3_POLYP.hdf5',
                      #custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})

opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1,
                                amsgrad=False)

modelVGG.compile(loss=bce_dice_loss, optimizer=opt,
              metrics=['binary_accuracy', dice_coef, iou])

#modelVGG.evaluate(test_gen, verbose=1, batch_size=8)

Y_pred = modelVGG.predict(test_gen, verbose=1)

#for i in range(321):




y_pred = np.argmax(Y_pred, axis=1)
print("CONFUSION MATRIX")
print(confusion_matrix(test_gen.classes, y_pred))
print('Classification Report')
target_names = ['No Polyp', 'Polyp']
print(classification_report(test_gen.classes, y_pred, target_names=target_names))
