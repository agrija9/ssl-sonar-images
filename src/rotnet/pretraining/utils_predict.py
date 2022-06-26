import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sn
import pandas as pd

def plot_predicted_rotation(image, true_label, predicted_label):
    """
    Plots predicted label for single image.
    """
    rotation_angles = {0: "0 degrees",
                       1: "90 degrees",
                       2: "180 degrees",
                       3: "270 degrees"}

    fig, ax = plt.subplots(1)
    fig.set_size_inches(12, 8)
    ax.imshow(image)
    ax.text(0, 38, "Predicted rotation:" + rotation_angles[predicted_label], fontsize="xx-large", bbox={'facecolor': 'green', "alpha" : 0.5, 'pad': 5})
    ax.text(0, 35, "True rotation:" + rotation_angles[true_label], fontsize="xx-large", bbox={'facecolor': 'red', "alpha" : 0.5, 'pad': 5})
    plt.show()

def plot_batch_predicted_rotation(images, true_labels, predicted_labels, images_prefix):
    """
    Plots predicted rotations for batch of images.
    Creates a 4x4 image grid.
    """
    rotation_angles = {0: "0$^\circ$",
                       1: "90$^\circ$",
                       2: "180$^\circ$",
                       3: "270$^\circ$"}

    fig = plt.figure(figsize=(12, 12))
    columns = 3
    rows = 4

    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns*rows):
        img = images[i]
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i+1))
        # set title
        ax[-1].set_title("Pred rotation: " + rotation_angles[predicted_label] + "\n" + "True rotation: " + rotation_angles[true_label], fontsize=18)
        # ax[-1].set_facecolor('xkcd:salmon')
        # ax[-1].set_facecolor((1.0, 0.47, 0.42))
        plt.axis('off')
        plt.imshow(img[:,:,0], alpha=0.90, cmap="gray")

    fig.tight_layout()
    # fig.savefig(os.path.join(images_prefix, "rotnet_prediction_" + str(i) + ".png"), dpi=fig.dpi)
    plt.show()
    return fig

def plot_batch_predicted_class(images, true_labels, predicted_labels, images_prefix, dataset):
    """
    Plots class predictions for marine debris sonar dataset.
    """

    if dataset=="sonar1": # marine debris tank
        sonar_classes = {0:'can', 1:'bottle', 2:'drink-carton', 3:'chain',
                         4:'propeller', 5:'tire', 6:'hook', 7:'valve',
                         8:'shampoo-bottle', 9:'standing-bottle', 10:'background'}

    elif dataset=="sonar2": # marine turned table
        sonar_classes = {0:"bottle", 1:"can", 2:"carton", 3:"box", 4:"bidon",
                         5:"pipe", 6:"platform", 7:"propeller", 8:"sachet",
                         9:"tire", 10:"valve", 11:"wrench"}
    else:
        print("Can not predict classes with given sonar dataset name")
        exit()

    fig = plt.figure(figsize=(12, 12))
    columns = 3
    rows = 4

    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns*rows):
        img = images[i]
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i+1))
        # set title
        ax[-1].set_title("Pred class: " + sonar_classes[predicted_label] + "\n" + "True class: " + sonar_classes[true_label], fontsize=18)
        plt.axis('off')
        plt.imshow(img[:,:,0], alpha=0.90, cmap="gray")

    fig.tight_layout()
    plt.show()
    return fig

def plot_confusion_matrix(confusion_matrix, dataset):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes

      Obtained from: https://www.tensorflow.org/tensorboard/image_summaries
      """

    # Names of the integer classes, i.e., 0 -> bottle, 1 -> can, etc.
    if dataset=="sonar1":
        class_names = ["can", 'bottle', 'drink-carton', 'chain',
                           'propeller', 'tire', 'hook', 'valve',
                           'shampoo-bottle', 'standing-bottle', 'background']

    elif dataset=="sonar2":
        class_names = ["bottle", "can", "carton", "box", "bidon",
                         "pipe", "platform", "propeller", "sachet",
                         "tire", "valve", "wrench"]
    else:
        print("Can not compute confusion matrix w given sonar dataset name")
        exit()

    # Normalize confusion matrix
    confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # figure = plt.figure(figsize=(10, 10))
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Confusion matrix")
    # plt.colorbar()
    # tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks, class_names, rotation=45)
    # plt.yticks(tick_marks, class_names)
    #
    # # Compute the labels from the normalized confusion matrix.
    # labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    #
    # # Use white text if squares are dark; otherwise black.
    # threshold = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     color = "white" if cm[i, j] > threshold else "black"
    #     plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)
    #
    # # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

    figure = plt.figure(figsize = (10,10))
    plt.title("Confusion Matrix", fontsize=18)
    df_cm = pd.DataFrame(confusion_matrix_normalized, index = [i for i in class_names], columns = [i for i in class_names])
    sn.heatmap(df_cm, cmap="Greens", annot=True)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=13)
    plt.yticks(tick_marks, class_names, rotation=0, fontsize=13)
    plt.ylabel("True label", fontsize=16)
    plt.xlabel("Predicted label", fontsize=16)

    return figure

def plot_confusion_matrix_rotations(confusion_matrix):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes

      Obtained from: https://www.tensorflow.org/tensorboard/image_summaries
      """

    # Names of the integer classes, i.e., 0 -> bottle, 1 -> can, etc.
    class_names = ["0$^\circ$",
                    "90$^\circ$",
                    "180$^\circ$",
                    "270$^\circ$"]

    # Normalize confusion matrix
    confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # figure = plt.figure(figsize=(10, 10))
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Confusion matrix")
    # plt.colorbar()
    # tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks, class_names, rotation=45)
    # plt.yticks(tick_marks, class_names)
    #
    # # Compute the labels from the normalized confusion matrix.
    # labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    #
    # # Use white text if squares are dark; otherwise black.
    # threshold = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     color = "white" if cm[i, j] > threshold else "black"
    #     plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)
    #
    # # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

    figure = plt.figure(figsize = (10,10))
    plt.title("Confusion Matrix for Rotations", fontsize=18)
    df_cm = pd.DataFrame(confusion_matrix_normalized, index = [i for i in class_names], columns = [i for i in class_names])
    sn.heatmap(df_cm, cmap="Greens", annot=True)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=13)
    plt.yticks(tick_marks, class_names, rotation=0, fontsize=13)
    plt.ylabel("True label", fontsize=16)
    plt.xlabel("Predicted label", fontsize=16)

    return figure

def compute_precision_recall_fscore(y_true, y_pred, average="weighted"):
    # based on: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    return precision_recall_fscore_support(y_true, y_pred, average=average)
