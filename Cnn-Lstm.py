import pandas as pd
import os
import utils
import numpy as np
import re
import model
import data
from keras.callbacks import EarlyStopping
import keras
import matplotlib.pyplot as plt

path_data = "/home/achauviere/Bureau/DATA/"
path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/"
path_small_Unet = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/small_unet.h5"

for time in [6, 9, 12, 16, 32, 64, 128]:

    ### Path ###
    path_souris = path_data + "Souris/"
    path_mask = path_data + "Masques/"
    path_img = path_data + "Image/"
    path_lab = path_data + "Label/"
    tab = pd.read_csv(path_data + "Tableau_General.csv").values


    ### Data ###
    numSouris = utils.calcul_numSouris(path_souris)
    data_3D, label_3D, ind_3D = data.crate_data_3D(path_img, path_lab, tab, numSouris)
    data_3D = data_3D.reshape(-1,128,128,128,1)
    label_3D = label_3D.reshape(-1,128,128,128,1)
    # label_3D = np.array(label_3D, dtype='bool')

    sample = int(3456/time)

    # fit de lots de time
    data_3D = data_3D.reshape(sample,time,128,128,1)
    label_3D = label_3D.reshape(sample,time,128,128,1)
    label_3D = np.array(label_3D, dtype='bool')

    ### Modèle BCLSTM ###
    input_shape = (time,128,128,1)
    model_bcsltm = model.bclstm_unet(input_shape)
    model_bcsltm.compile(optimizer='adam', loss="binary_crossentropy", metrics=[utils.mean_iou])
    earlystopper = EarlyStopping(patience=5, verbose=1)
    model_bcsltm.fit(data_3D, label_3D, validation_split=0.2, batch_size=1, epochs=50, callbacks=[earlystopper])
    model_bcsltm.save(path_result + "bclstm_"+str(time)+".h5")





    ### Transfer Learning ###
    input_shape = (time,128,128,1)

    # model small unet
    #path_small_Unet = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/small_unet.h5"
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



