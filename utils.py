import re
import numpy as np
import logging
from skimage import exposure, measure
import tensorflow as tf
from keras import backend as K
from random import gauss
from tensorflow.python import math_ops
import os
import model

def sorted_aphanumeric(data):
    """
    :param data: list d'élément alphanumérique.
    :return: list triée dans l'ordre croissant alphanumérique.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def intersection(lst1, lst2):
    """
    :param lst1: list d'éléments.
    :param lst2: list d'éléments.
    :return: intersection de ses deux listes.
    """
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def contraste_and_reshape(souris):
    """
    :param souris: Ensemble d'image, vérification si ensemble ou image unique avec la condition if.
    :return: Ensemble d'image avec contraste amélioré et shape modifié pour entrer dans le réseaux.
    """
    if len(souris.shape) > 2:
        data = []
        for i in np.arange(souris.shape[0]):
            img_adapteq = exposure.equalize_adapthist(souris[i], clip_limit=0.03)
            data.append(img_adapteq)
        data = np.array(data).reshape(-1, 128, 128, 1)
        return (data)
    else:
        img_adapteq = exposure.equalize_adapthist(souris, clip_limit=0.03)
        img = np.array(img_adapteq).reshape(128, 128, 1)
        return(img)

def calcul_numSouris(path_souris):
    list_souris = sorted_aphanumeric(os.listdir(path_souris))
    numSouris = []
    for k in np.arange(len(list_souris)):
        numSouris.append(int(re.findall('\d+', list_souris[k])[0]))
    return numSouris


def mean_iou(y_true, y_pred):
    """
    :param y_true: array de label annoté.
    :param y_pred: array de label prédit par le modèle.
    :return: valeur de l'IoU.
    """
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def apply_mask(img, mask):
    """
    :param img:
    :param mask:
    :return:
    """
    im = np.zeros((128,128))
    for i in np.arange(128):
        for j in np.arange(128):
            if mask[i,j] == True :
                im[i,j] = img[i,j]*1
            else:
                im[i,j] = 0
    return im

def apply_mask_and_noise(img, mask, noise):
    """
    :param img:
    :param mask:
    :return:
    """
    im = np.zeros((128,128))
    for i in np.arange(128):
        for j in np.arange(128):
            if mask[i,j] == True :
                im[i,j] = img[i,j]*1
            else:
                im[i,j] = noise + gauss(0,10)
    return im


def etale_hist(img):
    new_img = (img - img.min())*255/(img.max()-img.min())
    return new_img


def concat_data(a,b):
    new = np.zeros(((np.shape(a)[0]+np.shape(b)[0],128,128)))
    new[0:np.shape(a)[0]] = a
    new[np.shape(a)[0]:(np.shape(a)[0]+np.shape(b)[0])] = b
    return new


def inverse_binary_mask(msk):
    new_mask = np.ones((128,128)) - msk
    return new_mask


def weight_map(label,a,b):
    """
    :param label: stack label
    :return: stack weight per slice
    """
    weight = np.zeros(((label.shape[0], 128, 128)))

    for k in np.arange(label.shape[0]):

        lab = label[k]
        contour = measure.find_contours(lab, 0.8)
        indx_mask = np.where(lab == 1)[0]
        indy_mask = np.where(lab == 1)[1]

        w = np.ones((128, 128))
        w[indx_mask, indy_mask] = a

        for i in np.arange(len(contour)):
            indx_cont = np.array(contour[i][:, 0], dtype='int')
            indy_cont = np.array(contour[i][:, 1], dtype='int')
            w[indx_cont, indy_cont] = b

        #w = w ** 2
        weight[k] = w

    return(weight)



# def weighted_cross_entropy(beta):
#
#     def convert_to_logits(y_pred):
#         y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
#         return tf.log(y_pred / (1 - y_pred))
#
#     def loss(y_true,y_pred):
#         y_pred = convert_to_logits(y_pred)
#         loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
#         return tf.reduce_mean(loss)
#
#     return loss



def weighted_cross_entropy(y_true, y_pred):
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=3)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    epsilon = tf.convert_to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))

    zeros = tf.zeros_like(y_pred, dtype=y_pred.dtype)  #array_ops
    cond = (y_pred >= zeros)
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(relu_logits - y_pred * seg, math_ops.log1p(math_ops.exp(neg_abs_logits)), name=None)
    return K.mean(math_ops.multiply(weight, entropy), axis=-1)


def stats_pixelbased(y_true, y_pred):
    """Calculates pixel-based statistics
    (Dice, Jaccard, Precision, Recall, F-measure)
    Takes in raw prediction and truth data in order to calculate accuracy
    metrics for pixel based classfication. Statistics were chosen according
    to the guidelines presented in Caicedo et al. (2018) Evaluation of Deep
    Learning Strategies for Nucleus Segmentation in Fluorescence Images.
    BioRxiv 335216.
    Args:
        y_true (3D np.array): Binary ground truth annotations for a single
            feature, (batch,x,y)
        y_pred (3D np.array): Binary predictions for a single feature,
            (batch,x,y)
    Returns:
        dictionary: Containing a set of calculated statistics
    Raises:
        ValueError: Shapes of `y_true` and `y_pred` do not match.
    Warning:
        Comparing labeled to unlabeled data will produce low accuracy scores.
        Make sure to input the same type of data for `y_true` and `y_pred`
    """

    if y_pred.shape != y_true.shape:
        raise ValueError('Shape of inputs need to match. Shape of prediction '
                         'is: {}.  Shape of y_true is: {}'.format(
            y_pred.shape, y_true.shape))

    pred = y_pred
    truth = y_true

    # if pred.sum() == 0 and truth.sum() == 0:
    #     logging.warning('DICE score is technically 1.0, '
    #                     'but prediction and truth arrays are empty. ')

    if truth.sum() == 0:
        pred = inverse_binary_mask(pred)
        truth = inverse_binary_mask(truth)

    # Calculations for IOU
    intersection = np.logical_and(pred, truth)
    union = np.logical_or(pred, truth)

    # Sum gets count of positive pixels
    dice = (2 * intersection.sum() / (pred.sum() + truth.sum()))
    jaccard = intersection.sum() / union.sum()
    precision = intersection.sum() / pred.sum()
    recall = intersection.sum() / truth.sum()
    Fmeasure = (2 * precision * recall) / (precision + recall)

    return {
        'Dice': dice,
        'IoU': jaccard,
        'precision': precision,
        'recall': recall,
        'Fmeasure': Fmeasure
    }




# def reshape_for_lstm(data_3D, shp):
#     a = int(data_3D.shape[1] / shp)
#     b = data_3D.shape[0]*a
#     newData = np.zeros(((((b, shp, 128, 128, 1)))))
#     i = 0
#     for k in range(data_3D.shape[0]):
#         c = shp*1
#         j = 0
#         while c < data_3D.shape[0]:
#             newData[i] = data_3D[k,shp*j:shp*(j+1),:,:,:]
#             i+=1
#     return newData


def moy_geom(a):
    x_t = np.exp(1 / len(a) * np.sum(np.log(a)))
    return x_t

