import gc
import math
import os
import time

import Plotting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import Models
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.python.keras.callbacks import ModelCheckpoint

import LoadData
import ScoreCalculation
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from random import randint

print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices("GPU")

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 256, 3]

inputs_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

training_path = 'Dataset_CCE/New_DataSet/df_train/*.png'
validaiton_path = 'Dataset_CCE/New_DataSet/df_valid/*.png'

df_train = LoadData.load_data(training_path)
df_val = LoadData.load_data(validaiton_path)

print(df_train.shape)
print(df_val.shape)


def preprocessfunction(filename):
    x = randint(0, 50)

    if (x > 49):
        filename = tfa.image.rotate(filename, 180 * math.pi / 180, interpolation='bilinear')
    return filename


train_generator_args1 = dict(rotation_range=45,
                             # width_shift_range=0.1,
                             # height_shift_range=0.1,
                             shear_range=0.05,
                             zoom_range=0.05,
                             horizontal_flip=True,
                             vertical_flip=True,
                             brightness_range=(0.85, 1.15),
                             # preprocessing_function=preprocessfunction,
                             fill_mode="constant")

print(train_generator_args1.items())

'''Training Configuration'''
epochs = 260
batchSIZE = 6
learning_rate = 1e-4

decay_rate = learning_rate / epochs
callbackEarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=90)

train_gen = LoadData.train_generator(df_train, batchSIZE, train_generator_args1,
                                     target_size=(IMG_HEIGHT, IMG_WIDTH))

valid_generator = LoadData.train_generator(df_val, batchSIZE,
                                           dict(),
                                           target_size=(IMG_HEIGHT, IMG_WIDTH))


def trainingConfiguration(model, callbackModelCheckPoint, df_train, df_val, modelName):
    K.clear_session()
    gc.collect()

    number_of_images = 5
    polyp = [next(train_gen) for i in range(0, 5)]
    fig, ax = plt.subplots(1, number_of_images, figsize=(16, 6))

    for x in range(number_of_images):
        ax[x].axis('off')

    l = [ax[i].imshow(polyp[i][0][0]) for i in range(0, 5)]
    plt.show()

    opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate,
                                amsgrad=False)

    model.compile(loss=ScoreCalculation.bce_dice_loss, optimizer=opt,
                  metrics=['binary_accuracy', ScoreCalculation.dice_coef, ScoreCalculation.iou])

    history = model.fit(train_gen,
                        steps_per_epoch=len(df_train) / batchSIZE,
                        epochs=epochs,
                        validation_data=valid_generator,
                        validation_steps=len(df_val) / batchSIZE, verbose=2,
                        callbacks=[callbackModelCheckPoint, callbackEarlyStopping])

    Plotting.plotTraining_Data(history, modelName)

    time.sleep(90)


'''Models'''
###VGG19###
# modelvgg19 = Models.build_vgg19_unet()
# callbackModelCheckPointvgg19_100NoPolyp = [
#     ModelCheckpoint('180DegreeConstant/RANDOMTEST.hdf5', verbose=2, save_best_only=True)]

#
# ###ResNeXt50###
# modelresnext50 = Models.build_resnext50()
# callbackModelCheckPointresnext50_100NoPolyp = [
#     ModelCheckpoint('New_modelsWithoutDropout/ResNextUnet.hdf5', verbose=2, save_best_only=True)]
#
# ###ResNet50###
# modelresnet50 = Models.build_resnet50()
# callbackModelCheckPointresnet50_100NoPolyp = [
#     ModelCheckpoint('New_modelsWithoutDropout/ResUnet.hdf5', verbose=2, save_best_only=True)]
#
# ###Inceptionv3###
# modelinceptionv3 = Models.build_inceptionV3()
# callbackModelCheckPointinceptionv3_100NoPolyp = [
#     ModelCheckpoint('New_modelsWithoutDropout/Inceptionv3.hdf5', verbose=2, save_best_only=True)]
#
# ###InceptionResNetv2###
# modelinceptionresnetv2 = Models.build_inceptionresnetV2()
# callbackModelCheckPointinceptionresnetv2_100NoPolyp = [
#     ModelCheckpoint('New_modelsWithoutDropout/inceptionresnetv2.hdf5', verbose=2, save_best_only=True)]
#
###EU-Net###
##dd can never be higher than KK
# NN = 2
# KK = 2
# dd = 2
# modelEUnet = Models.EU_Net_Segmentation(NN, KK, dd, 'sigmoid')
# callbackModelCheckpointEUnet100NoPolyp = [
#     ModelCheckpoint('EUnets/Useless.hdf5', verbose=2, save_best_only=True)]
#
###U-Net###
##dd can never be higher than KK
# NN = 5
# KK = 5
# dd = 0
# modelUnet = Models.EU_Net_Segmentation(NN, KK, dd, 'sigmoid')
# callbackModelCheckpointUnet100NoPolyp = [
#     ModelCheckpoint('New_modelsWithoutDropout/Unet.hdf5', verbose=2, save_best_only=True)]
#

# ##Unet++###
# modelUnet2plus = Models.UNetPP(inputs_size)
# callbackModelCheckPointUnet2plus_100NoPolyp = [
#     ModelCheckpoint('New_modelsWithoutDropout/Unet2plus_WithShear.hdf5', verbose=2, save_best_only=True)]
#
##Unet+++###
# modelUnet3plus = Models.build_Unet3p()
# callbackModelCheckPointUnet3plus = [
#     ModelCheckpoint('New_modelsWithoutDropout/Unet3plus_WithShear.hdf5', verbose=2, save_best_only=True)]
# modelUnet3plus.summary()

# EU-Net VGG19##
NN = 5
KK = 4
dd = 1
EUnetVGG19 = Models.EU_Net_Segmentation(NN, KK, dd, 'sigmoid', 'vgg19')
callbackModelCheckPointEUnetVGG19 = [
    ModelCheckpoint('EUnets/TEST.hdf5', verbose=2, save_best_only=True)]
EUnetVGG19.summary()

'''Training'''
# trainingConfiguration(modelvgg19, callbackModelCheckPointvgg19_100NoPolyp, df_train, df_val, "VGG19Unet")
# trainingConfiguration(modelresnext50, callbackModelCheckPointresnext50_100NoPolyp,df_train, df_val,  "ResNeXt50")
# trainingConfiguration(modelresnet50, callbackModelCheckPointresnet50_100NoPolyp,df_train, df_val, "ResNet50")
# trainingConfiguration(modelinceptionv3, callbackModelCheckPointinceptionv3_100NoPolyp,df_train, df_val,  "InceptionV3")
# trainingConfiguration(modelinceptionresnetv2, callbackModelCheckPointinceptionresnetv2_100NoPolyp, df_train, df_val, "InceptionResnetV2")
# trainingConfiguration(modelEUnet, callbackModelCheckpointEUnet100NoPolyp, df_train, df_val, "EU-Net")
# trainingConfiguration(modelUnet, callbackModelCheckpointUnet100NoPolyp, df_train, df_val, "U-Net")
# trainingConfiguration(modelUnet2plus, callbackModelCheckPointUnet2plus_100NoPolyp, df_train, df_val, "U-Net++ ")
# trainingConfiguration(modelUnet3plus, callbackModelCheckPointUnet3plus, df_train, df_val, "U-Net+++ ")
# trainingConfiguration(EUnetVGG19, callbackModelCheckPointEUnetVGG19, df_train, df_val, "EU-Net")


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
# modelvgg19.summary()
# modelresnext50.summary()
# modelresnet50.summary()
# modelinceptionv3.summary()
# modelinceptionresnetv2.summary()
# modelEUnet.summary()
# modelUnet2plus.summary()
