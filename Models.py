
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


IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [512, 512, 3]
inputs_size = input_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

##ResUnet
# lets create model now
def resblock(X, f):
    '''
    function for creating res block
    '''
    X_copy = X  # copy of input

    # main path
    X = Conv2D(f, kernel_size=(1, 1), kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(f, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)

    # shortcut path
    X_copy = Conv2D(f, kernel_size=(1, 1), kernel_initializer='he_normal')(X_copy)
    X_copy = BatchNormalization()(X_copy)

    # Adding the output from main path and short path together
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    return X


def upsample_concat(x, skip):
    '''
    funtion for upsampling image
    '''
    X = UpSampling2D((2, 2))(x)
    merge = Concatenate()([X, skip])

    return merge


def ressUnet():
    input = Input(inputs_size)

    # Stage 1
    conv_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    pool_1 = MaxPool2D((2, 2))(conv_1)
    pool_1 = Dropout(0.2)(pool_1)

    # stage 2
    conv_2 = resblock(pool_1, 32)
    pool_2 = MaxPool2D((2, 2))(conv_2)

    # Stage 3
    conv_3 = resblock(pool_2, 64)
    pool_3 = MaxPool2D((2, 2))(conv_3)

    # Stage 4
    conv_4 = resblock(pool_3, 128)
    pool_4 = MaxPool2D((2, 2))(conv_4)
    pool_4 = Dropout(0.2)(pool_4)

    # Stage 5 (bottle neck)
    conv_5 = resblock(pool_4, 256)

    # Upsample Stage 1
    up_1 = upsample_concat(conv_5, conv_4)
    up_1 = resblock(up_1, 128)

    # Upsample Stage 2
    up_2 = upsample_concat(up_1, conv_3)
    up_2 = resblock(up_2, 64)

    # Upsample Stage 3
    up_3 = upsample_concat(up_2, conv_2)
    up_3 = resblock(up_3, 32)

    # Upsample Stage 4
    up_4 = upsample_concat(up_3, conv_1)
    up_4 = resblock(up_4, 16)

    drop1 = Dropout(0.7)(up_4)
    # final output
    output = Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same', activation='sigmoid')(drop1)
    return Model(inputs=[input], outputs=[output])

##VGGNET
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_vgg19_unet():
    """ Input """
    inputs = Input(inputs_size)

    """ Pre-trained VGG19 Model """
    vgg19 = keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
    # before this i tried with trainable layer but the accuracy was less as compared

    """ Encoder """
    s1 = vgg19.get_layer("block1_conv2").output
    s2 = vgg19.get_layer("block2_conv2").output
    s3 = vgg19.get_layer("block3_conv4").output
    s4 = vgg19.get_layer("block4_conv4").output

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    dropout = Dropout(0.8)(d4)
    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(dropout)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model