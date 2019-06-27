import re
import numpy as np
from skimage import exposure
import tensorflow as tf
from keras import backend as K
from random import gauss


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