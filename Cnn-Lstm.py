import pandas as pd
import utils
import numpy as np
import model
import data
from keras.callbacks import EarlyStopping
import keras
import os
import sys

# Choix de path
console = False
if console:
    ROOT_DIR = os.path.abspath("/home/achauviere/Bureau/Projet_Detection_Metastase_Souris/")
    sys.path.append(ROOT_DIR)
    PATH_GIT = os.path.join(ROOT_DIR, "./Antoine_Git/")
    PATH_DATA = os.path.join(ROOT_DIR, "./DATA/")
else:
    PATH_GIT = "./Antoine_Git/"
    PATH_DATA = "./DATA/"

path_result = os.path.join(PATH_GIT, "Poumons/model/")
path_small_Unet = os.path.join(PATH_GIT, "Poumons/model/small_unet.h5")

""" 
L'idée est ici d'évaluer la méthode des k-Unet avec différents valeurs de time. 
L'architecture d'un U-Net est le small U-Net est une couche BDC LSTM au niveau du pont du U-Net.
Une deuxième comparaison est faite en transferant les poids déjà entrainé d'un small U-Net dans ce réseau 
et d'entrainer seulement la partie BDC LSTM
"""

for time in [6, 9, 12, 16, 32, 64, 128]:

    ### Path ###
    path_souris = PATH_DATA + "Poumons/Souris/"
    path_mask = PATH_DATA + "Poumons/Masques/"
    path_img = PATH_DATA + "Poumons/Image/"
    path_lab = PATH_DATA + "Poumons/Label/"
    tab = pd.read_csv(PATH_DATA + "Poumons/Tableau_General.csv").values

    ### Data ###
    numSouris = utils.calcul_numSouris(path_souris)
    data_3D, label_3D, ind_3D = data.crate_data_3D(path_img, path_lab, tab, numSouris)
    data_3D = data_3D.reshape(-1, 128, 128, 128, 1)
    label_3D = label_3D.reshape(-1, 128, 128, 128, 1)

    sample = int(data_3D.shape[0]*data_3D.shape[1]/time)

    # fit de lots de time
    data_3D = data_3D.reshape(sample, time, 128, 128, 1)
    label_3D = label_3D.reshape(sample, time, 128, 128, 1)
    label_3D = np.array(label_3D, dtype='bool')

    ### Modèle BCLSTM ###
    input_shape = (time, 128, 128, 1)
    model_bcsltm = model.bclstm_unet(input_shape)
    model_bcsltm.compile(optimizer='adam', loss="binary_crossentropy", metrics=[utils.mean_iou])
    earlystopper = EarlyStopping(patience=5, verbose=1)
    model_bcsltm.fit(data_3D, label_3D, validation_split=0.2, batch_size=1, epochs=50, callbacks=[earlystopper])
    model_bcsltm.save(path_result + "bclstm_"+str(time)+".h5")

    ### Transfer Learning ###
    input_shape = (time, 128, 128, 1)

    # model small unet (pour un autre modèle, il s'agira de revoir le nbr de couches à freezer par la suite)
    small_Unet = keras.models.load_model(path_small_Unet, custom_objects={'mean_iou': utils.mean_iou})

    # model bclstm
    bclstm_Unet = model.bclstm_unet(input_shape)

    # transfer weight and freeze
    for k in np.arange(12) + 1:
        bclstm_Unet.layers[k].set_weights(small_Unet.layers[k].get_weights())
        bclstm_Unet.layers[k].trainable = False

    for k in (np.arange(16) + 16):
        if k != (17 or 22 or 27):
            bclstm_Unet.layers[k - 1].set_weights(small_Unet.layers[k].get_weights())
            bclstm_Unet.layers[k - 1].trainable = False

    bclstm_Unet.compile(optimizer='adam', loss="binary_crossentropy", metrics=[utils.mean_iou])
    earlystopper = EarlyStopping(patience=5, verbose=1)
    bclstm_Unet.fit(data_3D, label_3D, validation_split=0.2, batch_size=8, epochs=50, callbacks=[earlystopper])
    bclstm_Unet.save(path_result + "bclstm_"+str(time)+"_tl.h5")

    # Il serait peut être pertinent d'ensuite réentrainer le modèle en entier pour une bonne continuité entre la partie
    # freezer et la partie LSTM entrainée.
