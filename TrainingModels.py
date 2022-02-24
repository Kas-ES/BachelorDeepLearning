import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import segmentation_models as sm

print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices("GPU")

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 256, 3]

#Training
df_train_files = []
df_train_mask_files = glob('Dataset_CCE/SortedDataset/df_train/*_mask*')

for i in df_train_mask_files:
    df_train_files.append(i.replace('_mask', ''))

df_train = pd.DataFrame(data={"filename": df_train_files, 'mask': df_train_mask_files})

#Validaiton

df_valid_files = []
df_valid_mask_files = glob('Dataset_CCE/SortedDataset/df_valid/*_mask*')

for i in df_valid_mask_files:
    df_valid_files.append(i.replace('_mask', ''))

df_val = pd.DataFrame(data={"filename": df_valid_files, 'mask': df_valid_mask_files})

#Testing

df_test_files = []
df_test_mask_files = glob('Dataset_CCE/SortedDataset/df_test/*_mask*')

for i in df_test_mask_files:
    df_test_files.append(i.replace('_mask', ''))

df_test = pd.DataFrame(data={"filename": df_test_files, 'mask': df_test_mask_files})

print(df_train.shape)
print(df_test.shape)
print(df_val.shape)

inputs_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


# From: https://github.com/zhixuhao/unet/blob/master/data.py
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


def plotAcuracy_Loss(history):
    plt.plot(history.history['iou'])
    plt.plot(history.history['val_iou'])
    plt.title('Intersection over union')
    plt.ylabel('IoU')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Dice coefficient')
    plt.ylabel('dice_coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Dice coefficient loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
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
    return dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)


def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


'''Training Configuration'''
epochs = 150
batchSIZE = 8
learning_rate = 1e-4

# Change shift range from 0.05 to 0.15
# Change shear and zoom range from 0.05 to 0.15
# train_generator_args = dict(rotation_range=0.25,
#                             width_shift_range=0.15,
#                             height_shift_range=0.15,
#                             shear_range=45,
#                             channel_shift_range=150.0,
#                             zoom_range=[0.5, 1.5],
#                             horizontal_flip=True,
#                             brightness_range=(0.1, 0.9),
#                             fill_mode="nearest")

train_generator_args = dict(rotation_range=0.1,
                            width_shift_range=0.09,
                            height_shift_range=0.09,
                            shear_range=0.09,
                            #channel_shift_range=50.0,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            #brightness_range=(0.3, 0.9),
                            fill_mode="nearest")

decay_rate = learning_rate / epochs
#opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, decay=decay_rate,
#                            amsgrad=False)

train_gen = train_generator(df_train, batchSIZE, train_generator_args,
                            target_size=(IMG_HEIGHT, IMG_WIDTH))

valid_generator = train_generator(df_val, batchSIZE,
                                  dict(),
                                  target_size=(IMG_HEIGHT, IMG_WIDTH))
test_gen = train_generator(df_test, batchSIZE,
                                  dict(),
                                  target_size=(IMG_HEIGHT, IMG_WIDTH))


callbackEarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
# callbackReduceROnPlateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6)

def trainingConfiguration (model, callbackModelCheckPoint):
    if model == modelvgg19 or model == modelinceptionresnetv2:
        opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1 ,decay=decay_rate,
                                    amsgrad=False)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, decay=decay_rate,
                                    amsgrad=False)

    model.compile(loss=bce_dice_loss, optimizer=opt,
                  metrics=['binary_accuracy', dice_coef, iou])

    history = model.fit(train_gen,
                        steps_per_epoch=len(df_train) / batchSIZE,
                        epochs=epochs,
                        validation_data=valid_generator,
                        validation_steps=len(df_val) / batchSIZE, verbose=2,
                        callbacks=[callbackModelCheckPoint, callbackEarlyStopping])

    plotAcuracy_Loss(history)

'''Models'''
###VGG19###
## model = Models.build_vgg19_unet()
modelvgg19 = sm.Unet('vgg19', classes=1, activation='sigmoid', encoder_weights='imagenet', encoder_freeze=False,input_shape=inputs_size)
callbackModelCheckPointvgg19 = [ModelCheckpoint('VGGU19net_POLYP.hdf5', verbose=2, save_best_only=True)]

###ResNeXt###
modelresnext50 = sm.Unet('resnext50', classes=1, activation='sigmoid', encoder_weights='imagenet', encoder_freeze=False, input_shape=inputs_size)
callbackModelCheckPointresnext50 = [ModelCheckpoint('ResNextUnet_POLYP.hdf5', verbose=2, save_best_only=True)]

###ResNet###
modelresnet50 = sm.Unet('resnet50', activation='sigmoid', classes=1, encoder_weights='imagenet', encoder_freeze=False, input_shape=inputs_size)
callbackModelCheckPointresnet50 = [ModelCheckpoint('ResUnet_POLYP.hdf5', verbose=2, save_best_only=True)]

###Inceptionv3###
modelinceptionv3 = sm.Unet('inceptionv3', classes=1, activation='sigmoid', encoder_weights='imagenet', encoder_freeze=False, input_shape=inputs_size)
callbackModelCheckPointinceptionv3 = [ModelCheckpoint('Inceptionv3_POLYP.hdf5', verbose=2, save_best_only=True)]

###InceptionResNetv2###
modelinceptionresnetv2 = sm.Unet('inceptionresnetv2', classes=1, activation='sigmoid', encoder_weights='imagenet', encoder_freeze=False, input_shape=inputs_size)
callbackModelCheckPointinceptionresnetv2 = [ModelCheckpoint('inceptionresnetv2_POLYP.hdf5', verbose=2, save_best_only=True)]

#model.summary()

'''Model compile and training'''
# model.compile(loss=bce_dice_loss, optimizer=opt,
#               metrics=['binary_accuracy', dice_coef, iou])
#
# history = model.fit(train_gen,
#                     steps_per_epoch=len(df_train) / batchSIZE,
#                     epochs=epochs,
#                     validation_data=valid_generator,
#                     validation_steps=len(df_val) / batchSIZE, verbose=2,
#                     callbacks=[callbackModelCheckPoint, callbackEarlyStopping])

trainingConfiguration(modelvgg19, callbackModelCheckPointvgg19)
K.clear_session()
trainingConfiguration(modelresnext50, callbackModelCheckPointresnext50)
K.clear_session()
trainingConfiguration(modelresnet50, callbackModelCheckPointresnet50)
K.clear_session()
trainingConfiguration(modelinceptionv3, callbackModelCheckPointinceptionv3)
K.clear_session()
trainingConfiguration(modelinceptionresnetv2, callbackModelCheckPointinceptionresnetv2)
