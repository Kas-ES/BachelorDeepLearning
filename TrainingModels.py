import gc
import os
import time

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.python.keras.utils.vis_utils import plot_model
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from natsort import natsorted
from tensorflow.python.client import device_lib
from tensorflow.python.keras.callbacks import ModelCheckpoint

import Models
from tensorflow.keras import backend as K
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from keras_unet_collection import models
import segmentation_models as sm

print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices("GPU")

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 256, 3]


'''100% NO POLYP DATA'''
# Training
df_train_files = []
df_train_mask_files = natsorted(glob('Dataset_CCE/New_DataSet/df_train/*.png'))

for i in df_train_mask_files:
    df_train_files.append(i[:-3]+'jpg')

df_train = pd.DataFrame(data={"filename": df_train_files, 'mask': df_train_mask_files})

# Validaiton
df_valid_files = []
df_valid_mask_files = natsorted(glob('Dataset_CCE/New_DataSet/df_valid/*.png'))

for i in df_valid_mask_files:
    df_valid_files.append(i[:-3]+'jpg')

df_val = pd.DataFrame(data={"filename": df_valid_files, 'mask': df_valid_mask_files})

print(df_train.shape)
print(df_val.shape)


inputs_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# From: https://github.com/zhixuhao/unet/blob/master/data.py
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


def plotAcuracy_Loss(history, modelname):
    plt.plot(history.history['iou'])
    plt.plot(history.history['val_iou'])
    plt.title(modelname+'—Intersection over union')
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title(modelname+'—Dice coefficient')
    plt.ylabel('Dice Coef')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(modelname+'—Dice coefficient loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()


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

'''Training Configuration'''
epochs = 260
batchSIZE = 4
learning_rate = 1e-4

train_generator_args1 = dict(rotation_range=45,
                            #width_shift_range=0.1,
                            #height_shift_range=0.1,
                            #shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            vertical_flip=True,
                            brightness_range=(0.85, 1.15),
                            fill_mode="constant")

# train_generator_args = dict(rotation_range=0.2,
#                             width_shift_range=0.05,
#                             height_shift_range=0.05,
#                             shear_range=0.05,
#                             zoom_range=0.05,
#                             horizontal_flip=True,
#                             vertical_flip=True,
#                             brightness_range=(0.85, 1.15),
#                             fill_mode='constant')

print(train_generator_args1.items())

decay_rate = learning_rate / epochs

callbackEarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=90)
# callbackReduceROnPlateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6)

train_gen = train_generator(df_train, batchSIZE, train_generator_args1,
                              target_size=(IMG_HEIGHT, IMG_WIDTH))

valid_generator = train_generator(df_val, batchSIZE,
                                      dict(),
                                      target_size=(IMG_HEIGHT, IMG_WIDTH))

def trainingConfiguration(model, callbackModelCheckPoint, df_train, df_val, modelName):
    K.clear_session()
    gc.collect()

    polyp = [next(train_gen) for i in range(0,5)]
    fig, ax = plt.subplots(1, 5, figsize=(16, 6))
    l = [ax[i].imshow(polyp[i][0][0]) for i in range(0, 5)]
    plt.show()

    opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate,
                                amsgrad=False)

    model.compile(loss=bce_dice_loss, optimizer=opt,
                  metrics=['binary_accuracy', dice_coef, iou])

    history = model.fit(train_gen,
                        steps_per_epoch=len(df_train) / batchSIZE,
                        epochs=epochs,
                        validation_data=valid_generator,
                        validation_steps=len(df_val) / batchSIZE, verbose=2,
                        callbacks=[callbackModelCheckPoint, callbackEarlyStopping])

    plotAcuracy_Loss(history, modelName)

    time.sleep(90)


'''Models'''
###VGG19###
modelvgg19 = Models.build_vgg19_unet()
callbackModelCheckPointvgg19_100NoPolyp = [
    ModelCheckpoint('New_modelsWithoutDropout/VGGU19net.hdf5', verbose=2, save_best_only=True)]


###ResNeXt50###
modelresnext50 = Models.build_resnext50()
callbackModelCheckPointresnext50_100NoPolyp = [
    ModelCheckpoint('New_modelsWithoutDropout/ResNextUnet.hdf5', verbose=2, save_best_only=True)]

###ResNet50###
modelresnet50 = Models.build_resnet50()
callbackModelCheckPointresnet50_100NoPolyp = [
    ModelCheckpoint('New_modelsWithoutDropout/ResUnet.hdf5', verbose=2, save_best_only=True)]

###Inceptionv3###
modelinceptionv3 = Models.build_inceptionV3()
callbackModelCheckPointinceptionv3_100NoPolyp = [
    ModelCheckpoint('New_modelsWithoutDropout/Inceptionv3.hdf5', verbose=2, save_best_only=True)]

###InceptionResNetv2###
modelinceptionresnetv2 = Models.build_inceptionresnetV2()
callbackModelCheckPointinceptionresnetv2_100NoPolyp = [
    ModelCheckpoint('New_modelsWithoutDropout/inceptionresnetv2.hdf5', verbose=2, save_best_only=True)]

###EU-Net###
##dd can never be higher than KK
NN = 5
KK = 4
dd = 3
modelEUnet = Models.EU_Net_Segmentation(NN, KK, dd, 'sigmoid')
callbackModelCheckpointEUnet100NoPolyp = [
    ModelCheckpoint('New_modelsWithoutDropout/EUnet.hdf5', verbose=2, save_best_only=True)]

##Unet++###
modelUnet2plus = Models.UNetPP(inputs_size)
callbackModelCheckPointUnet2plus_100NoPolyp = [
    ModelCheckpoint('New_modelsWithoutDropout/Unet2plus.hdf5', verbose=2, save_best_only=True)]


'''train_generator_args'''
###trainingConfiguration(modelvgg19, callbackModelCheckPointvgg19_100NoPolyp, df_train, df_val, "VGG19Unet")
###trainingConfiguration(modelresnext50, callbackModelCheckPointresnext50_100NoPolyp,df_train, df_val,  "ResNeXt50")
###trainingConfiguration(modelresnet50, callbackModelCheckPointresnet50_100NoPolyp,df_train, df_val, "ResNet50")
####trainingConfiguration(modelinceptionv3, callbackModelCheckPointinceptionv3_100NoPolyp,df_train, df_val,  "InceptionV3")
###trainingConfiguration(modelinceptionresnetv2, callbackModelCheckPointinceptionresnetv2_100NoPolyp, df_train, df_val, "InceptionResnetV2")
###trainingConfiguration(modelEUnet, callbackModelCheckpointEUnet100NoPolyp, df_train, df_val,  "EU-Net")
###trainingConfiguration(modelUnet2plus, callbackModelCheckPointUnet2plus_100NoPolyp, df_train, df_val, "U-Net++ ")

'''Plot The Models'''
# plot_model(modelresnext50, "ModelArchitecturesPlot/modelresnext50.png", show_shapes=False,dpi=1536)
# plot_model(modelvgg19, "ModelArchitecturesPlot/modelvgg19.png", show_shapes=True, dpi=1536, show_layer_names=True)
# plot_model(modelresnet50, "ModelArchitecturesPlot/modelresnet50.png", show_shapes=True, dpi=1536, show_layer_names=True)
# plot_model(modelinceptionv3, "ModelArchitecturesPlot/modelinceptionv3.png", show_shapes=True, dpi=1536,
#            show_layer_names=True)
# plot_model(modelinceptionresnetv2, "ModelArchitecturesPlot/modelinceptionresnetv2.png", show_shapes=True, dpi=3072,
#            show_layer_names=True)
# plot_model(modelEUnet, "ModelArchitecturesPlot/EUnet.png", show_shapes=True, dpi=1536, show_layer_names=True)
# plot_model(modelUnet2plus,"ModelArchitecturesPlot/Unet++.png", show_shapes=True, dpi=1536, show_layer_names=False )

'''Model summaries'''
#modelvgg19.summary()
#modelresnext50.summary()
#modelresnet50.summary()
#modelinceptionv3.summary()
#modelinceptionresnetv2.summary()
#modelEUnet.summary()
modelUnet2plus.summary()








