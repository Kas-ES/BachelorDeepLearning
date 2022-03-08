from glob import glob

import numpy as np


# Testing
import pandas as pd
from matplotlib import pyplot as plt
from natsort import natsorted
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras import backend as K
import Models

df_test_files = []
df_test_labels = []
df_test_mask_files = natsorted(glob('Dataset_CCE/New_DataSet/df_test/*.png'))

for i in df_test_mask_files:
    df_test_files.append(i[:-3]+'jpg')

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 256, 3]
df_test = pd.DataFrame(data={"filename": df_test_files, 'mask': df_test_mask_files})



from skimage.io import imread#, imshow
test_Inputs = []
test_Masks = []
for ii in range(len(df_test)):
    temp = imread(df_test.iloc[ii]['filename'])
    temp = np.uint8(resize(temp,
                    (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                    mode='constant',
                    anti_aliasing=True,
                    preserve_range=True))
    test_Inputs.append(temp)
    temp = imread(df_test.iloc[ii]['mask'])
    temp = np.expand_dims(resize(temp,
                                  (IMG_HEIGHT, IMG_WIDTH),
                                  mode='constant',
                                  anti_aliasing=True,
                                  preserve_range=True), axis=-1)
    test_Masks.append(temp>0)

test_Masks = np.array(1*test_Masks)
test_Inputs = np.array(test_Inputs)

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
    return abs(dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred))


def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

'''MODELS'''
VGG19 = tf.keras.models.load_model('New_modelsWithoutDropout/VGGU19net.hdf5',
                                custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})
ResNextUnet = tf.keras.models.load_model('New_modelsWithoutDropout/ResNextUnet.hdf5',
                                       custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou,
                                                       'dice_coef': dice_coef})
Resnet = tf.keras.models.load_model('New_modelsWithoutDropout/ResUnet.hdf5',
                                  custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef})
InceptionV3 = tf.keras.models.load_model('New_modelsWithoutDropout/Inceptionv3.hdf5',
                                       custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou,
                                                    'dice_coef': dice_coef})
InceptionResnetV2 = tf.keras.models.load_model('New_modelsWithoutDropout/inceptionresnetv2.hdf5',
                                             custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou,
                                                            'dice_coef': dice_coef,'tf': tf})
EUnet = tf.keras.models.load_model('New_modelsWithoutDropout/EUnet.hdf5',
                                custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef,'tf': tf})

Unet2plus = tf.keras.models.load_model('New_modelsWithoutDropout/Unet2plus.hdf5',
                                 custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'dice_coef': dice_coef,'tf': tf})


preds_test = Unet2plus.predict(test_Inputs, verbose=2, batch_size=1)

SS1=test_Masks.shape
y_true = test_Masks.reshape(np.prod(SS1),1)
pred_scores = preds_test.reshape(np.prod(SS1),1)

# test_Inputs1 = test_Inputs
# test_Inputs = (lambda x: x/(255**2))(test_Inputs)
test_Inputs = (lambda x: x/(255))(test_Inputs)


def get_confusion_matrix_elements(groundtruth_list, predicted_list):
    """returns confusion matrix elements i.e TN, FP, FN, TP as floats
    """
    A = np.uint8(np.squeeze(groundtruth_list))  # np.uint8(np.squeeze(Test_Masks[iix]))
    B = np.squeeze(predicted_list)  # np.squeeze(preds_test_t[iix])

    tp = np.count_nonzero(A * B)
    fn = np.count_nonzero(A) - tp
    fp = np.count_nonzero(B) - tp
    tn = (A.shape[0] * A.shape[1]) - (tp + fp + fn)

    return tn, fp, fn, tp

import random
iix = random.randint(0, len(test_Inputs) - 1)

TN, FP, FN, TP = get_confusion_matrix_elements(test_Masks[iix], preds_test[iix]>0.5)

# 13. Pretty Confusion Matrix Drawing

# imports
from pandas import DataFrame
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh


def get_new_fig(fn, figsize=[9, 9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = [];
    text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if (cell_val != 0):
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%' % (per_ok), '100%'][per_ok == 100]

        # text to DEL
        text_del.append(oText)

        # text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy();
        dic['color'] = 'g';
        lis_kwa.append(dic);
        dic = text_kwargs.copy();
        dic['color'] = 'r';
        lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y), (oText._x, oText._y + 0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        # print '\n'

        # set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if (per > 0):
            txt = '%s\n%.2f%%' % (cell_val, per)
        else:
            if (show_null_values == 0):
                txt = ''
            elif (show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        # main diagonal
        if (col == lin):
            # set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del


#
import seaborn as sn

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    # print ('\ndf_cm:\n', df_cm, '\n\b\n')

#'twilight_shifted'#'twilight'#'hsv'#'plasma'#'viridis'#'Greys'#'cividis'#'Greens' #'gist_rainbow' #'Oranges'  # 'oranges'
cmap = 'twilight_shifted'

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap=cmap, fmt='.2f', fz=20,
                                 lw=1.5, cbar=False, figsize=[8, 8], show_null_values=0, pred_val_axis='x'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if (pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    # this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    # thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    # set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, fontsize=15, \
                       weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=35, fontsize=15, \
                       weight='bold')

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = [];
    text_del = [];
    posi = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1]);
        col = int(pos[0]);
        posi += 1
        # print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        # set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    plt.figure(figsize=(10, 7))

    sn.set(font_scale=1.4)  # for label size

    # titles and legends
    ax.set_title('Confusion matrix', weight='bold', fontsize=30)
    ax.set_xlabel(xlbl, weight='bold', fontsize=25)
    ax.set_ylabel(ylbl, weight='bold', fontsize=25)
    plt.tight_layout()  # set layout slim
    plt.show()


#

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap=cmap,
                                    fmt='.2f', fz=20, lw=1.5, cbar=False, fig_size=[8, 8], show_null_values=0,
                                    pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        without a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    # data
    if not columns:
        # labels axis integer:
        # columns = range(1, len(np.unique(y_test))+1)
        # labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' % i for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    conf_matrix = confusion_matrix(y_test, predictions)
    fz = 20
    fig_size = [9, 9]
    show_null_values = 2
    dataframe = DataFrame(conf_matrix, index=columns, columns=columns)
    pretty_plot_confusion_matrix(dataframe, fz=fz, cmap=cmap, figsize=fig_size,
                                 show_null_values=show_null_values,
                                 pred_val_axis=pred_val_axis)


array = np.array([[TP, FN],
                  [FP, TN]])

df_cm = DataFrame(array, index=["TP", "FP"], columns=["FN", "TN"])
cmap = 'twilight'#'twilight_shifted'#'hsv'#'plasma'#'viridis'#'Greys'#'cividis'#'Greens' #'gist_rainbow' #'Oranges'  # 'oranges'
pretty_plot_confusion_matrix(df_cm, cmap=cmap)

# 14. Schematic of proposed
import cv2


def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """ Returns dict of 4 boolean numpy arrays with True at TP, FP, FN, TN
    """

    confusion_matrix_arrs = {}

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs['tp'] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs['tn'] = np.logical_and(groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs['fp'] = np.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs['fn'] = np.logical_and(groundtruth, predicted_inverse)

    return confusion_matrix_arrs


def get_confusion_matrix_overlaid_mask(image, groundtruth, predicted, alpha, colors):
    """
    Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
    each a color given by the 'colors' dictionary
    """
    #    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masks = get_confusion_matrix_intersection_mats(groundtruth, predicted)
    color_mask = np.zeros_like(image)
    for label1, mask in masks.items():
        color = colors[label1]
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask != 0] = color
        color_mask += mask_rgb
    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)


alpha = 0.5
confusion_matrix_colors = {
    'tp': (0, 255, 255),  # cyan
    'fp': (255, 0, 255),  # magenta
    'fn': (255, 255, 0),  # yellow
    'tn': (0, 0, 0)  # black
}
A = np.uint8(np.squeeze(test_Masks[iix]))
B = (np.squeeze(preds_test[iix]>0.5))
validation_mask = get_confusion_matrix_overlaid_mask(test_Inputs[iix], A,
                                                     B, alpha,
                                                     confusion_matrix_colors)
print('Cyan - TP')
print('Magenta - FP')
print('Yellow - FN')
print('Black - TN')
plt.imshow(validation_mask)
plt.axis('off')
plt.title('confusion matrix overlay mask')

# 15. Confusion Matrix for total of Test_Inputs

TP = 0
TN = 0
FN = 0
FP = 0

for t in np.arange(0, len(test_Inputs)):
    TN1, FP1, FN1, TP1 = get_confusion_matrix_elements(test_Masks[t], preds_test[t]>0.5)
    TP += TP1
    FP += FP1
    TN += TN1
    FN += FN1

array = np.array([[TP, FN],
                  [FP, TN]])

df_cm = pd.DataFrame(array, index=["TP", "FP"], columns=["FN", "TN"])

# df_cm = df_cm.style.set_properties({'font-size': '20pt'})
pretty_plot_confusion_matrix(df_cm, cmap=cmap)
plt.figure(figsize=(10, 7))

sn.set(font_scale=2.4)  # for label size

F_score = np.round(10000 * (2 * TP) / (2 * TP + FP + FN)) / 100
Accuracy = np.round(10000 * (TP + TN) / (TP + FP + FN + TN)) / 100
print('TP=', TP)
print('FP=', FP)
print('FN=', FN)
print('TN=', TN)
print('Accuracy= ', Accuracy)
print('F_Score=', F_score)