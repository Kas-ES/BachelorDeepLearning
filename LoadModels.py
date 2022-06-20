import os

import ScoreCalculation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras

import tensorflow as tf

EUnets = ["EUnet5-5-5", "EUnet5-5-4", "EUnet5-4-4", "EUnet5-5-3", "EUnet5-5-2", "EUnet5-4-3", "EUnet4-5-5",
          "EUnet4-4-4", "EUnet4-4-2", "EUnet4-4-1", "EUnet4-3-2", "EUnet3-3-3"]
AllNueralNetworks = ["VGGU19net", "ResNextUnet", "ResUnet", "Inceptionv3", "inceptionresnetv2", "EUnet5-5-3",
                     "Unet2plus", "Unet3plus", "Unet", "EUnetVGG19_5-4-1_FinalVersion_Shear_Dropout0.1_FinalDropout0.5",
                     "EUnetVGG19_5-3-2(3Dropout)"]


def load_model(name_path, type):
    if (type == "EUnets"):
        model = keras.models.load_model("EUnets/" + name_path + '.hdf5',
                                        custom_objects={'bce_dice_loss': ScoreCalculation.bce_dice_loss,
                                                        'iou': ScoreCalculation.iou,
                                                        'dice_coef': ScoreCalculation.dice_coef, 'tf': tf})
        return model
    elif (type == "All"):
        model = keras.models.load_model("New_modelsWithoutDropout/" + name_path + '.hdf5',
                                        custom_objects={'bce_dice_loss': ScoreCalculation.bce_dice_loss,
                                                        'iou': ScoreCalculation.iou,
                                                        'dice_coef': ScoreCalculation.dice_coef, 'tf': tf})
        return model
