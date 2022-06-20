import os

import numpy as np
from cv2 import cv2
from sklearn.metrics import precision_recall_curve, average_precision_score, ConfusionMatrixDisplay, \
    classification_report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["SM_FRAMEWORK"] = "tf.keras"

import matplotlib.pyplot as plt


def plotTraining_Data(history, modelname):
    plt.plot(history.history['iou'])
    plt.plot(history.history['val_iou'])
    plt.title(modelname + '—Intersection over union')
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('IOU.png')
    plt.show()

    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title(modelname + '—Dice coefficient')
    plt.ylabel('Dice Coef')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('DICE.png')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(modelname + '—Dice coefficient loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('LOSS.png')
    plt.show()


'''Precision recall Curve of all models'''


def precision_recall_curve_models(predictions, y_true, labels, colors):
    fig, ax = plt.subplots(figsize=(7, 8))

    # f_scores = np.linspace(0.2, 0.8, num=4)
    # lines, labels = [], []
    # for f_score in f_scores:
    #     x = np.linspace(0.01, 1)
    #     y = f_score * x / (2 * x - f_score)
    #     (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
    #     plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    new_labels = []

    for i in range(len(predictions)):
        y_pred = np.ndarray.flatten(predictions[i])
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        AP = average_precision_score(y_true, y_pred)

        amountAP = labels[i] + ' (AP = ' + str(round(AP, 2)) + ')'
        new_labels.append(amountAP)

        ax.plot(recall, precision, color=colors[i], label=new_labels[i])

    # add the legend for the iso-f1 curves
    # handles, labels = display.ax_.get_legend_handles_labels()
    # handles.extend([l])
    # labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(labels=new_labels, loc="best")
    # ax.set_title("Models' Precision-Recall Curves")
    ax.set_title("EU-Net Configurations Precision-Recall Curve")

    plt.ylabel('Precision')
    plt.xlabel('Recall')

    # x = [0, 1]
    # y = [0, 1]
    # plt.plot(x, y, '--')
    # plt.annotate('AP ', (0.50,0.46))

    plt.grid()
    plt.show()


'''Time Inference Plot'''


def drawCircles(f1scores, time_inferances, labels, colors):
    # 0.01=1M , 0.1=10M, 0.15=15M, 0.2=20M, 0.25=25M, 0.3=30M, 0,6=60M
    model_parameter_number = [31, 32, 32, 30, 62, 15, 8, 1, 31, 39, 16]
    model_parameters_size = [0.31 / 30, 0.32 / 30, 0.32 / 30, 0.3 / 30, 0.62 / 30, 0.15 / 30, 0.08 / 30, 0.01 / 30,
                             0.31 / 30, 0.39 / 30, 0.16 / 30]

    fig, ax = plt.subplots()

    labels_with_parameter = []
    for i in range(len(time_inferances)):
        minute = time_inferances[i] / 60
        label = str(labels[i]) + '-' + str(model_parameter_number[i]) + 'M'
        labels_with_parameter.append(label)
        circle = plt.Circle((minute, f1scores[i]), model_parameters_size[i], color=colors[i], label=label)
        ax.add_patch(circle)

    ax.set(xlim=(0, 0.35), ylim=(0.55, 1))
    ax.set_aspect('equal', 'box')

    fig.tight_layout()
    plt.title('Time Inference plot')
    plt.ylabel('F1-Score')
    plt.xlabel('Minute')
    plt.legend(labels_with_parameter, loc='upper right', prop={"size": 6})

    'Real time processborder'
    # seconds = 60/frames_per_second
    # minute = seconds / 60
    # x = [minute, minute]
    # y = [0, 1]
    # plt.plot(x,y,'--')
    # plt.annotate('Real Time Process Border', (minute+0.01,0.65))

    plt.grid()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()
    fig.savefig('timeInferencePlot.png')


def imageMultipleViz(model, dataframe, IMG_HEIGHT, IMG_WIDTH):
    for i in range(10):
        # index = np.random.randint(1, len(dataframe.index))
        index = 4 * i
        img = cv2.imread(dataframe['filename'].iloc[72])
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
        plt.imshow(np.squeeze(cv2.imread(dataframe['mask'].iloc[index])))
        plt.title('Original Mask')
        plt.subplot(1, 3, 3)
        plt.imshow(np.squeeze(pred) > .5)
        plt.title('Prediction')
        plt.show()


def imageVisualiaziton(model, dataframe, label, IMG_HEIGHT, IMG_WIDTH):
    index = 72
    img = cv2.imread(dataframe['filename'].iloc[index])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred = model.predict(img)

    # fig, (ax1, ax2) = plt.subplots(1,2)
    #
    # ax1.imshow(np.squeeze(img))
    # ax1.set_title('Original Image')
    #
    # ax2.imshow(np.squeeze(cv2.imread(dataframe['mask'].iloc[index])))
    # ax2.set_title('Ground Truth')

    plt.imshow(np.squeeze(pred) > .5)
    plt.title(label + ' Prediction')

    plt.show()

    # plt.figure(figsize=(10, 10))
    # # plt.subplot(1, 3, 1)
    # # plt.imshow(np.squeeze(img))
    # # plt.title('Original Image')
    # plt.subplot(1, 1, 1)
    # plt.imshow(np.squeeze(cv2.imread(dataframe['mask'].iloc[index])))
    # plt.title('Original Mask')
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.squeeze(pred) > .5)
    # plt.title(label + ' Prediction')
    # plt.show()

def confusionMatrixAndClasificaitonReport(Y_pred, ground_truth, model_name):
    y_pred = np.where(Y_pred > 0.5, 1, 0)
    y_pred = np.ndarray.flatten(y_pred)

    disp1 = ConfusionMatrixDisplay.from_predictions(ground_truth, y_pred, display_labels=['No Polyp', 'Polyp'],
                                                    values_format='')
    plt.title(model_name + " Confusion Matrix")
    disp1.plot()
    plt.show()

    print(classification_report(ground_truth, y_pred, target_names=['No Polyp', 'Polyp']))

