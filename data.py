import numpy as np
from skimage import io, exposure
import utils
import os


############################################################################################
                ####################### POUMONS #######################
############################################################################################



def create_data_detect_poum(path_img,tab,numSouris):
    """
    -- Chargement des images de souris annotées + amélioration du contraste
       Chargement des masques + complétions par masque vide --

    :param path_img: ensemble des images de souris où les poumons ont été annotés.
    :param tab: tableau résumant les identifiants et annotations pour les souris.
    :param numSouris: numéro de souris annotées (cf tableau deuxième colonne).

    :return: image avec son masque associé et son identifiant
    """

    data_detec = []
    label_detec = []
    ind_detec = []

    for i in np.arange(len(tab)):
         if tab[i, 1] in numSouris:
            im = io.imread(path_img + 'img_' + str(i) + '.tif', plugin='tifffile')
            img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03)
            data_detec.append(img_adapteq)
            label_detec.append(tab[i, 3])
            ind_detec.append(i)

    data_detec = np.array(data_detec)
    label_detec = np.array(label_detec)
    ind_detec = np.array(ind_detec)

    return(data_detec, label_detec, ind_detec)



def create_data_seg_poum(path_img,path_lab,tab):

    """
    -- Chargement des Images et des Masques : que ceux où poumon == 1 sur csv. --
    :param path_img: ensemble des images de souris où les poumons ont été annotés.
    :param path_lab: ensemble des annotations de poumons.
    :param tab: tableau résumant les identifiants et annotations pour les souris.
    :return:
    """

    data_2D = []
    label_2D = []
    ind_2D = []

    for i in np.arange(len(tab)):
        if tab[i,4]==1:
            im = io.imread(path_img + 'img_'+str(i)+'.tif', plugin='tifffile')
            img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03)
            data_2D.append(img_adapteq)
            label_2D.append(io.imread(path_lab + 'm_'+str(i)+'.tif'))
            ind_2D.append(i)

    data_2D = np.array(data_2D)
    label_2D = np.array(label_2D, dtype=np.bool)
    ind_2D = np.array(ind_2D)

    return (data_2D, label_2D, ind_2D)



def recup_new_data():

    path_new = "/home/achauviere/Bureau/Poumon_sain_seg/Nouvelles_Images/"

    newData = []
    newMaskPoum = []
    newMaskMeta = []

    list_new_souris = utils.sorted_aphanumeric(os.listdir(path_new))

    for k in np.arange(len(list_new_souris)):
        path_new_img = path_new + list_new_souris[k] + "/Images/"
        path_new_msk_poum = path_new + list_new_souris[k] + "/Masque_Poum/"
        path_new_msk_meta = path_new + list_new_souris[k] + "/Masque_Meta/"

        for i in np.arange(len(os.listdir(path_new_img))):
            im = io.imread(path_new_img + utils.sorted_aphanumeric(os.listdir(path_new_img))[i])
            img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03)
            newData.append(img_adapteq)
            newMaskPoum.append(io.imread(path_new_msk_poum + utils.sorted_aphanumeric(os.listdir(path_new_msk_poum))[i]))
            newMaskMeta.append(io.imread(path_new_msk_meta + utils.sorted_aphanumeric(os.listdir(path_new_msk_meta))[i]))

    newData = np.array(newData)
    newMaskPoum = np.array(newMaskPoum, dtype=np.bool)
    newMaskMeta = np.array(newMaskMeta, dtype=np.bool)

    return(newData, newMaskPoum, newMaskMeta)



def crate_data_3D(path_img, path_lab, tab, numSouris, etale=False):

    data_3D = []
    label_3D = []
    ind_3D = []

    for i in np.arange(len(tab)):
        if tab[i, 1] in numSouris:
            im = io.imread(path_img + 'img_' + str(i) + '.tif', plugin='tifffile')
            img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03)
            if not etale:
                data_3D.append(img_adapteq)
            else:
                img = utils.etale_hist(img_adapteq)
                data_3D.append(img)
            ind_3D.append(i)
            if tab[i, 4] == 1:
                label_3D.append(io.imread(path_lab + 'm_' + str(i) + '.tif', plugin='tifffile'))
            else:
                label_3D.append(np.zeros((128, 128)))

    data_3D = np.array(data_3D)
    label_3D = np.array(label_3D)
    ind_3D = np.array(ind_3D)

    # Reconstruction en souris
    data_ = np.zeros((((len(numSouris), 128, 128, 128))))
    label_ = np.zeros((((len(numSouris), 128, 128, 128))))

    for i in np.arange(len(numSouris)):
        data_[i] = data_3D[(128 * i):(128 * (i + 1))]
        label_[i] = label_3D[(128 * i):(128 * (i + 1))]

    data_3D = data_
    label_3D = label_

    return(data_3D, label_3D, ind_3D)






############################################################################################
                ####################### METASTASES #######################
############################################################################################




def create_data_meta(path_img, path_lab, tab):
    """
    :param path_img: Toutes les slices appartenant a des souris ou les métas ont été annotées
    :param path_lab: Tous les labels correspondant (vide si pas de métastases)
    :param tab: Tableau csv avec les colonnes pour les métastases (Nbr, Num, Annot)
    :return: Data, Label, ind
    """

    Data = []
    label = []
    ind = []

    for i in np.arange(len(tab)):
        if tab[i, 5] == 1:
            if tab[i, 6] != 0:
                im = io.imread(path_img + 'img_' + str(i) + '.tif', plugin='tifffile')
                img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03)
                Data.append(img_adapteq)
                label.append(io.imread(path_lab + 'm_' + str(i) + '.tif'))
                ind.append(i)

    Data = np.array(Data)
    label = np.array(label, dtype=np.bool)
    ind = np.array(ind)

    return(Data, label, ind)