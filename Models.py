# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 08:59:56 2022

@author: asta
"""
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import numpy as np
from tensorflow import keras
from keras_unet_collection._backbone_zoo import bach_norm_checker, backbone_zoo
from keras_unet_collection._model_unet_2d import UNET_left, UNET_right
from keras_unet_collection.layer_utils import CONV_output, decode_layer, CONV_stack, encode_layer
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Activation, Dropout, Conv2DTranspose, Concatenate, \
    Lambda, MaxPooling2D, concatenate, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
import segmentation_models as sm

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 256, 3]
inputs_size = input_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

inputs = keras.Input(shape=input_size)


##ResUnet
def build_resnet50():
    modelresnet50 = sm.Unet('resnet50', activation='sigmoid', classes=1, encoder_weights='imagenet',
                            encoder_freeze=False,
                            input_shape=inputs_size)
    model_input = modelresnet50.input
    model_output = modelresnet50.layers[-3].output
    dropout = Dropout(0.2)(model_output)
    output = Conv2D(1, 1, padding="same", activation="sigmoid")(dropout)
    model_dp = Model(model_input, output)
    return model_dp


def build_resnext50():
    modelresnext50 = sm.Unet('resnext50', classes=1, activation='sigmoid', encoder_weights='imagenet',
                             encoder_freeze=False,
                             input_shape=inputs_size)
    model_input = modelresnext50.input
    model_output = modelresnext50.layers[-3].output
    dropout = Dropout(0.2)(model_output)
    output = Conv2D(1, 1, padding="same", activation="sigmoid")(dropout)
    model_dp = Model(model_input, output)
    return model_dp


def build_inceptionV3():
    modelinceptionv3 = sm.Unet('inceptionv3', classes=1, activation='sigmoid', encoder_weights='imagenet',
                               encoder_freeze=False, input_shape=inputs_size)
    model_input = modelinceptionv3.input
    model_output = modelinceptionv3.layers[-3].output
    dropout = Dropout(0.2)(model_output)
    output = Conv2D(1, 1, padding="same", activation="sigmoid")(dropout)
    model_dp = Model(model_input, output)
    return model_dp


def build_inceptionresnetV2():
    modelinceptionresnetv2 = sm.Unet('inceptionresnetv2', classes=1, activation='sigmoid', encoder_weights='imagenet',
                                     encoder_freeze=False, input_shape=inputs_size)
    model_input = modelinceptionresnetv2.input
    model_output = modelinceptionresnetv2.layers[-3].output
    dropout = Dropout(0.2)(model_output)
    output = Conv2D(1, 1, padding="same", activation="sigmoid")(dropout)
    model_dp = Model(model_input, output)
    return model_dp


def build_Unet3p():
    base_Unet3p = unet_3plus_2d(inputs_size, 1, [8, 16, 32, 64, 128, 256], weights='None', batch_norm=True, deep_supervision=True)

    model_input = base_Unet3p.input
    model_output = base_Unet3p.layers[-3].output
    dropout = Dropout(0.2)(model_output)
    output = Conv2D(1, 1, padding="same", activation="sigmoid")(dropout)
    model_dp = Model(model_input, output)
    return model_dp


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

    # dropout = Dropout(0.5)(d4)
    dropout = Dropout(0.5)(d4)
    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(dropout)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model


##U-Net++
# https://github.com/longpollehn/UnetPlusPlus/blob/master/unetpp.py


def conv2d(filters: int):
    return Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.),
                  bias_regularizer=l2(0.))


def conv2dtranspose(filters: int):
    return Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same')


def UNetPP(input, number_of_filters=2):
    model_input = Input(input)
    x00 = conv2d(filters=int(16 * number_of_filters))(model_input)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    # x00 = Dropout(0.2)(x00)
    x00 = conv2d(filters=int(16 * number_of_filters))(x00)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    # x00 = Dropout(0.2)(x00)
    p0 = MaxPooling2D(pool_size=(2, 2))(x00)

    x10 = conv2d(filters=int(32 * number_of_filters))(p0)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    x10 = Dropout(0.2)(x10)
    x10 = conv2d(filters=int(32 * number_of_filters))(x10)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    # x10 = Dropout(0.2)(x10)
    p1 = MaxPooling2D(pool_size=(2, 2))(x10)

    x01 = conv2dtranspose(int(16 * number_of_filters))(x10)
    x01 = concatenate([x00, x01])
    x01 = conv2d(filters=int(16 * number_of_filters))(x01)
    x01 = BatchNormalization()(x01)
    x01 = LeakyReLU(0.01)(x01)
    x01 = conv2d(filters=int(16 * number_of_filters))(x01)
    x01 = BatchNormalization()(x01)
    x01 = LeakyReLU(0.01)(x01)
    # x01 = Dropout(0.2)(x01)

    x20 = conv2d(filters=int(64 * number_of_filters))(p1)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    # x20 = Dropout(0.2)(x20)
    x20 = conv2d(filters=int(64 * number_of_filters))(x20)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    # x20 = Dropout(0.2)(x20)
    p2 = MaxPooling2D(pool_size=(2, 2))(x20)

    x11 = conv2dtranspose(int(16 * number_of_filters))(x20)
    x11 = concatenate([x10, x11])
    x11 = conv2d(filters=int(16 * number_of_filters))(x11)
    x11 = BatchNormalization()(x11)
    x11 = LeakyReLU(0.01)(x11)
    x11 = conv2d(filters=int(16 * number_of_filters))(x11)
    x11 = BatchNormalization()(x11)
    x11 = LeakyReLU(0.01)(x11)
    # x11 = Dropout(0.2)(x11)

    x02 = conv2dtranspose(int(16 * number_of_filters))(x11)
    x02 = concatenate([x00, x01, x02])
    x02 = conv2d(filters=int(16 * number_of_filters))(x02)
    x02 = BatchNormalization()(x02)
    x02 = LeakyReLU(0.01)(x02)
    x02 = conv2d(filters=int(16 * number_of_filters))(x02)
    x02 = BatchNormalization()(x02)
    x02 = LeakyReLU(0.01)(x02)
    # x02 = Dropout(0.2)(x02)

    x30 = conv2d(filters=int(128 * number_of_filters))(p2)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    x30 = Dropout(0.2)(x30)
    x30 = conv2d(filters=int(128 * number_of_filters))(x30)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    # x30 = Dropout(0.2)(x30)
    p3 = MaxPooling2D(pool_size=(2, 2))(x30)

    x21 = conv2dtranspose(int(16 * number_of_filters))(x30)
    x21 = concatenate([x20, x21])
    x21 = conv2d(filters=int(16 * number_of_filters))(x21)
    x21 = BatchNormalization()(x21)
    x21 = LeakyReLU(0.01)(x21)
    x21 = conv2d(filters=int(16 * number_of_filters))(x21)
    x21 = BatchNormalization()(x21)
    x21 = LeakyReLU(0.01)(x21)
    # x21 = Dropout(0.2)(x21)

    x12 = conv2dtranspose(int(16 * number_of_filters))(x21)
    x12 = concatenate([x10, x11, x12])
    x12 = conv2d(filters=int(16 * number_of_filters))(x12)
    x12 = BatchNormalization()(x12)
    x12 = LeakyReLU(0.01)(x12)
    x12 = conv2d(filters=int(16 * number_of_filters))(x12)
    x12 = BatchNormalization()(x12)
    x12 = LeakyReLU(0.01)(x12)
    # x12 = Dropout(0.2)(x12)

    x03 = conv2dtranspose(int(16 * number_of_filters))(x12)
    x03 = concatenate([x00, x01, x02, x03])
    x03 = conv2d(filters=int(16 * number_of_filters))(x03)
    x03 = BatchNormalization()(x03)
    x03 = LeakyReLU(0.01)(x03)
    x03 = conv2d(filters=int(16 * number_of_filters))(x03)
    x03 = BatchNormalization()(x03)
    x03 = LeakyReLU(0.01)(x03)
    # x03 = Dropout(0.2)(x03)

    m = conv2d(filters=int(256 * number_of_filters))(p3)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    m = conv2d(filters=int(256 * number_of_filters))(m)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    # m = Dropout(0.2)(m)

    x31 = conv2dtranspose(int(128 * number_of_filters))(m)
    x31 = concatenate([x31, x30])
    x31 = conv2d(filters=int(128 * number_of_filters))(x31)
    x31 = BatchNormalization()(x31)
    x31 = LeakyReLU(0.01)(x31)
    x31 = conv2d(filters=int(128 * number_of_filters))(x31)
    x31 = BatchNormalization()(x31)
    x31 = LeakyReLU(0.01)(x31)
    # x31 = Dropout(0.2)(x31)

    x22 = conv2dtranspose(int(64 * number_of_filters))(x31)
    x22 = concatenate([x22, x20, x21])
    x22 = conv2d(filters=int(64 * number_of_filters))(x22)
    x22 = BatchNormalization()(x22)
    x22 = LeakyReLU(0.01)(x22)
    x22 = conv2d(filters=int(64 * number_of_filters))(x22)
    x22 = BatchNormalization()(x22)
    x22 = LeakyReLU(0.01)(x22)
    # x22 = Dropout(0.2)(x22)

    x13 = conv2dtranspose(int(32 * number_of_filters))(x22)
    x13 = concatenate([x13, x10, x11, x12])
    x13 = conv2d(filters=int(32 * number_of_filters))(x13)
    x13 = BatchNormalization()(x13)
    x13 = LeakyReLU(0.01)(x13)
    x13 = conv2d(filters=int(32 * number_of_filters))(x13)
    x13 = BatchNormalization()(x13)
    x13 = LeakyReLU(0.01)(x13)
    # x13 = Dropout(0.2)(x13)

    x04 = conv2dtranspose(int(16 * number_of_filters))(x13)
    x04 = concatenate([x04, x00, x01, x02, x03], axis=3)
    x04 = conv2d(filters=int(16 * number_of_filters))(x04)
    x04 = BatchNormalization()(x04)
    x04 = LeakyReLU(0.01)(x04)
    x04 = conv2d(filters=int(16 * number_of_filters))(x04)
    x04 = BatchNormalization()(x04)
    x04 = LeakyReLU(0.01)(x04)
    x04 = Dropout(0.2)(x04)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(x04)
    model = Model(inputs=[model_input], outputs=[output], name="UNetPP")
    return model


def unet_3plus_2d_base(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate,
                       stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True,
                       backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True,
                       name='unet3plus'):
    '''
    The base of UNET 3+ with an optional ImagNet-trained backbone.

    unet_3plus_2d_base(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate,
                       stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True,
                       backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')

    ----------
    Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020.
    UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation.
    In ICASSP 2020-2020 IEEE International Conference on Acoustics,
    Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.

    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num_down: a list that defines the number of filters for each
                         downsampling level. e.g., `[64, 128, 256, 512, 1024]`.
                         the network depth is expected as `len(filter_num_down)`
        filter_num_skip: a list that defines the number of filters after each
                         full-scale skip connection. Number of elements is expected to be `depth-1`.
                         i.e., the bottom level is not included.
                         * Huang et al. (2020) applied the same numbers for all levels.
                           e.g., `[64, 64, 64, 64]`.
        filter_num_aggregate: an int that defines the number of channels of full-scale aggregations.
        stack_num_down: number of convolutional layers per downsampling level/block.
        stack_num_up: number of convolutional layers (after full-scale concat) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        name: prefix of the created keras model and its layers.

        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.
    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers here.

    Output
    ----------
        A list of tensors with the first/second/third tensor obtained from
        the deepest/second deepest/third deepest upsampling block, etc.
        * The feature map sizes of these tensors are different,
          with the first tensor has the smallest size.

    '''

    depth_ = len(filter_num_down)

    X_encoder = []
    X_decoder = []

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # stacked conv2d before downsampling
        X = CONV_stack(X, filter_num_down[0], kernel_size=3, stack_num=stack_num_down,
                       activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
        X_encoder.append(X)

        # downsampling levels
        for i, f in enumerate(filter_num_down[1:]):
            # UNET-like downsampling
            X = UNET_left(X, f, kernel_size=3, stack_num=stack_num_down, activation=activation,
                          pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i + 1))
            X_encoder.append(X)

    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder)

        # for other backbones
        else:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_ - 1, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder) + 1

        # extra conv2d blocks are applied
        # if downsampling levels of a backbone < user-specified downsampling levels
        if depth_encode < depth_:

            # begins at the deepest available tensor
            X = X_encoder[-1]

            # extra downsamplings
            for i in range(depth_ - depth_encode):
                i_real = i + depth_encode

                X = UNET_left(X, filter_num_down[i_real], stack_num=stack_num_down, activation=activation, pool=pool,
                              batch_norm=batch_norm, name='{}_down{}'.format(name, i_real + 1))
                X_encoder.append(X)

    # treat the last encoded tensor as the first decoded tensor
    X_decoder.append(X_encoder[-1])

    # upsampling levels
    X_encoder = X_encoder[::-1]

    depth_decode = len(X_encoder) - 1

    # loop over upsampling levels
    for i in range(depth_decode):

        f = filter_num_skip[i]

        # collecting tensors for layer fusion
        X_fscale = []

        # for each upsampling level, loop over all available downsampling levels (similar to the unet++)
        for lev in range(depth_decode):

            # counting scale difference between the current down- and upsampling levels
            pool_scale = lev - i - 1  # -1 for python indexing

            # deeper tensors are obtained from **decoder** outputs
            if pool_scale < 0:
                pool_size = 2 ** (-1 * pool_scale)

                X = decode_layer(X_decoder[lev], f, pool_size, unpool,
                                 activation=activation, batch_norm=batch_norm,
                                 name='{}_up_{}_en{}'.format(name, i, lev))

            # unet skip connection (identity mapping)
            elif pool_scale == 0:

                X = X_encoder[lev]

            # shallower tensors are obtained from **encoder** outputs
            else:
                pool_size = 2 ** (pool_scale)

                X = encode_layer(X_encoder[lev], f, pool_size, pool, activation=activation,
                                 batch_norm=batch_norm, name='{}_down_{}_en{}'.format(name, i, lev))

            # a conv layer after feature map scale change
            X = CONV_stack(X, f, kernel_size=3, stack_num=1,
                           activation=activation, batch_norm=batch_norm,
                           name='{}_down_from{}_to{}'.format(name, i, lev))

            X_fscale.append(X)

            # layer fusion at the end of each level
        # stacked conv layers after concat. BatchNormalization is fixed to True

        X = concatenate(X_fscale, axis=-1, name='{}_concat_{}'.format(name, i))
        X = CONV_stack(X, filter_num_aggregate, kernel_size=3, stack_num=stack_num_up,
                       activation=activation, batch_norm=True, name='{}_fusion_conv_{}'.format(name, i))
        X_decoder.append(X)

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation
    if depth_decode < depth_ - 1:
        for i in range(depth_ - depth_decode - 1):
            i_real = i + depth_decode
            X = UNET_right(X, None, filter_num_aggregate, stack_num=stack_num_up, activation=activation,
                           unpool=unpool, batch_norm=batch_norm, concat=False,
                           name='{}_plain_up{}'.format(name, i_real))
            X_decoder.append(X)

    # return decoder outputs
    return X_decoder


def unet_3plus_2d(input_size, n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto',
                  stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                  batch_norm=False, pool=True, unpool=True, deep_supervision=False,
                  backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus'):
    '''
    UNET 3+ with an optional ImageNet-trained backbone.

    unet_3plus_2d(input_size, n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto',
                  stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                  batch_norm=False, pool=True, unpool=True, deep_supervision=False,
                  backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')

    ----------
    Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020.
    UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation.
    In ICASSP 2020-2020 IEEE International Conference on Acoustics,
    Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.

    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num_down: a list that defines the number of filters for each
                         downsampling level. e.g., `[64, 128, 256, 512, 1024]`.
                         the network depth is expected as `len(filter_num_down)`
        filter_num_skip: a list that defines the number of filters after each
                         full-scale skip connection. Number of elements is expected to be `depth-1`.
                         i.e., the bottom level is not included.
                         * Huang et al. (2020) applied the same numbers for all levels.
                           e.g., `[64, 64, 64, 64]`.
        filter_num_aggregate: an int that defines the number of channels of full-scale aggregations.
        stack_num_down: number of convolutional layers per downsampling level/block.
        stack_num_up: number of convolutional layers (after full-scale concat) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        deep_supervision: True for a model that supports deep supervision. Details see Huang et al. (2020).
        name: prefix of the created keras model and its layers.

        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.

    * The Classification-guided Module (CGM) is not implemented.
      See https://github.com/yingkaisha/keras-unet-collection/tree/main/examples for a relevant example.
    * Automated mode is applied for determining `filter_num_skip`, `filter_num_aggregate`.
    * The default output activation is sigmoid, consistent with Huang et al. (2020).
    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers here.

    Output
    ----------
        model: a keras model.

    '''

    depth_ = len(filter_num_down)

    verbose = False

    if filter_num_skip == 'auto':
        verbose = True
        filter_num_skip = [filter_num_down[0] for num in range(depth_ - 1)]

    if filter_num_aggregate == 'auto':
        verbose = True
        filter_num_aggregate = int(depth_ * filter_num_down[0])

    if verbose:
        print('Automated hyper-parameter determination is applied with the following details:\n----------')
        print('\tNumber of convolution filters after each full-scale skip connection: filter_num_skip = {}'.format(
            filter_num_skip))
        print('\tNumber of channels of full-scale aggregated feature maps: filter_num_aggregate = {}'.format(
            filter_num_aggregate))

    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)

    X_encoder = []
    X_decoder = []

    IN = Input(input_size)

    X_decoder = unet_3plus_2d_base(IN, filter_num_down, filter_num_skip, filter_num_aggregate,
                                   stack_num_down=stack_num_down, stack_num_up=stack_num_up, activation=activation,
                                   batch_norm=batch_norm, pool=pool, unpool=unpool,
                                   backbone=backbone, weights=weights, freeze_backbone=freeze_backbone,
                                   freeze_batch_norm=freeze_batch_norm, name=name)
    X_decoder = X_decoder[::-1]

    if deep_supervision:

        # ----- frozen backbone issue checker ----- #
        if ('{}_backbone_'.format(backbone) in X_decoder[0].name) and freeze_backbone:
            backbone_warn = '\n\nThe deepest UNET 3+ deep supervision branch directly connects to a frozen backbone.\nTesting your configurations on `keras_unet_collection.base.unet_plus_2d_base` is recommended.'
            warnings.warn(backbone_warn);
        # ----------------------------------------- #

        OUT_stack = []
        L_out = len(X_decoder)

        print(
            '----------\ndeep_supervision = True\nnames of output tensors are listed as follows ("sup0" is the shallowest supervision layer;\n"final" is the final output layer):\n')

        # conv2d --> upsampling --> output activation.
        # index 0 is final output
        for i in range(1, L_out):

            pool_size = 2 ** (i)

            X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv_{}'.format(name, i - 1))(X_decoder[i])

            X = decode_layer(X, n_labels, pool_size, unpool,
                             activation=None, batch_norm=False, name='{}_output_sup{}'.format(name, i - 1))

            if output_activation:
                print('\t{}_output_sup{}_activation'.format(name, i - 1))

                if output_activation == 'Sigmoid':
                    X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i - 1))(X)
                else:
                    activation_func = eval(output_activation)
                    X = activation_func(name='{}_output_sup{}_activation'.format(name, i - 1))(X)
            else:
                if unpool is False:
                    print('\t{}_output_sup{}_trans_conv'.format(name, i - 1))
                else:
                    print('\t{}_output_sup{}_unpool'.format(name, i - 1))

            OUT_stack.append(X)

        X = CONV_output(X_decoder[0], n_labels, kernel_size=3,
                        activation=output_activation, name='{}_output_final'.format(name))
        OUT_stack.append(X)

        if output_activation:
            print('\t{}_output_final_activation'.format(name))
        else:
            print('\t{}_output_final'.format(name))

        model = Model([IN, ], OUT_stack)

    else:

        OUT = CONV_output(X_decoder[0], n_labels, kernel_size=3,
                          activation=output_activation, name='{}_output_final'.format(name))

        model = Model([IN, ], [OUT, ])

    return model



