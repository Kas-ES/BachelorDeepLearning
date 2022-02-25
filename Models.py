# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 08:59:56 2022

@author: asta
"""
import numpy as np
from keras import Model, Input
from keras.layers import Conv2D, Activation, Add, UpSampling2D, MaxPool2D, Dropout, Conv2DTranspose, Concatenate, \
    Lambda, MaxPooling2D, concatenate, BatchNormalization
from tensorflow import keras

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 256, 3]
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


## EU-Net
# KK = 4  # THE NUMBER OF CONTRACTING/EXPANSIVE STAGES IN PROPOSED EW-Net
# dd = 3
# NN = 5
# Optimizer = 'adam'
# Final_Act = 'sigmoid'#'tanh'# 'sigmoid'

def Con_Exp_Path(X, Y, n):
    L = Conv2D(2 ** X, (3, 3), kernel_initializer='he_normal',  # 'glorot_uniform', 'he_normal'
                padding='same')(n)
    L = BatchNormalization()(L)
    L = Activation('relu')(L)
    L = Conv2D(2 ** X, (3, 3), kernel_initializer='he_normal',  # 'glorot_uniform', 'he_normal'
                padding='same')(n)
    L = BatchNormalization()(L)
    L = Activation('relu')(L)
    L = Dropout(Y)(L)
    L = Conv2D(2 ** X, (3, 3), activation='relu', kernel_initializer='he_normal',  # 'glorot_uniform', 'he_normal'
                padding='same')(L)
    return L

def EU_Net_Segmentation(NN, KK, dd, Final_Act):
    NL = 9 + (KK + dd - 1) * 5  # No. of layers for EU-Net architecture
    MD = [None] * (NL)  # EU_Net Layers combination

    MD[0] = Input(input_size)
    MD[1] = Lambda(lambda x: x)(MD[0])

    # MD[1] = Conv2D(NN, (1,1), padding='valid')(MD[1])
    for tt in np.arange(0, KK):
        MD[2 * (tt + 1)] = Con_Exp_Path(NN + tt, 0.1 * (1 + np.fix(tt / 2)), MD[2 * (tt) + 1])
        MD[2 * (tt + 1) + 1] = MaxPooling2D((2, 2))(MD[2 * (tt + 1)])

    gg = 2 * (KK + 1)  # e.g. for KK=3 & dd=1==>gg=8 or for KK=2 & dd=2==>gg=6
    MD[gg] = Con_Exp_Path(NN + KK, 0.1 * (1 + np.fix((tt + 1) / 2)), MD[gg - 1])

    for tt in np.arange(dd - 1, -1, -1):
        MD[gg + 3 * (dd - tt - 1) + 1] = Conv2DTranspose(2 ** (NN + KK - (dd - tt)), \
                                                          (2, 2), strides=(2, 2), \
                                                          padding='same')(MD[gg + 3 * (dd - tt - 1)])
        MD[gg + 3 * (dd - tt - 1) + 2] = concatenate([MD[gg + 3 * (dd - tt - 1) + 1], MD[gg - 2 * (dd - tt)]])
        MD[gg + 3 * (dd - tt - 1) + 3] = Con_Exp_Path(NN + KK - (dd - tt), 0.1 * (1 + np.fix((KK - dd + tt) / 2)), \
                                                      MD[gg + 3 * (dd - tt - 1) + 2])

    gg += 3 * dd  # e.g. for KK=3 & dd=1 ==> gg=11 or for KK=2 & dd=2==>gg=12

    for tt in np.arange(0, dd):
        MD[gg + 2 * tt + 1] = MaxPooling2D((2, 2))(MD[gg + 2 * tt])
        MD[gg + 2 * tt + 2] = Con_Exp_Path(NN + KK + (tt + 1 - dd), 0.1 * (1 + np.fix((KK - dd + tt + 1) / 2)), \
                                            MD[gg + 2 * tt + 1])

    gg += 2 * dd  # e.g. for KK=3 & dd=1 ==> gg=13 or for KK=2 & dd=2==>gg=16

    for tt in np.arange(dd - 1, -1, -1):
        MD[gg + 3 * (dd - tt - 1) + 1] = Conv2DTranspose(2 ** (NN + KK - (dd - tt)), (2, 2), strides=(2, 2), \
                                                          padding='same')(MD[gg + 3 * (dd - tt - 1)])
        MD[gg + 3 * (dd - tt - 1) + 2] = concatenate([MD[gg + 3 * (dd - tt - 1) + 1], MD[gg - 2 * (dd - tt)]])
        MD[gg + 3 * (dd - tt - 1) + 3] = Con_Exp_Path(NN + KK - (dd - tt), 0.1 * (1 + np.fix((KK - dd + tt) / 2)), \
                                                      MD[gg + 3 * (dd - tt - 1) + 2])

    gg += 3 * dd  # e.g. for KK=3 & dd=1 ==> gg=16 or for KK=2 & dd=2==>gg=22

    for tt in np.arange(KK, dd, -1):
        MD[gg + 3 * (KK - tt) + 1] = Conv2DTranspose(2 ** (NN + tt - (dd + 1)), (2, 2), strides=(2, 2), \
                                                      padding='same')(MD[gg + 3 * (KK - tt)])
        MD[gg + 3 * (KK - tt) + 2] = concatenate([MD[gg + 3 * (KK - tt) + 1], MD[2 * (tt - dd)]])
        MD[gg + 3 * (KK - tt) + 3] = Con_Exp_Path(NN + tt - (dd + 1), 0.1 * (1 + np.fix((tt - dd - 1) / 2)), \
                                                  MD[gg + 3 * (KK - tt) + 2])
    if gg != NL - 1:
        gg += 3 * (KK - dd)

    # MD[gg+1] = Conv2D(1,(1,1), activation='sigmoid')(MD[gg])
    MD[gg + 1] = Conv2D(1, (1, 1))(MD[gg])
    MD[gg + 1] = Activation(Final_Act)(MD[gg + 1])

    model = Model(inputs=[MD[0]], outputs=[MD[gg + 1]])
    return model

# if Final_Act == 'tanh':
#     Pred_thresh = 0.01
# else:
#     Pred_thresh = 0.5