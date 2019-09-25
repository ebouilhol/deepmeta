########################################################################################################################
########################################################################################################################
#                                              SEGMENTATION DES POUMONS                                                #
########################################################################################################################
########################################################################################################################

import os
import numpy as np
import pandas as pd
from random import sample
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import model
import data
import utils
import sys


ROOT_DIR = os.path.abspath("/home/achauviere/Bureau/Projet_Detection_Metastase_Souris/")
sys.path.append(ROOT_DIR)

# PATH_GIT = "./Antoine_Git/"
PATH_GIT = os.path.join(ROOT_DIR, "./Antoine_Git/")

# PATH_DATA = "./DATA/"
PATH_DATA = os.path.join(ROOT_DIR, "./DATA/")

# PATH_Synth = "./Data_Synthetique/"
PATH_Synth = os.path.join(ROOT_DIR, "./Data_Synthetique/")

path_souris = os.path.join(PATH_DATA, "Poumons/Souris/")
path_mask = os.path.join(PATH_DATA, "Poumons/Masques/")
path_img = os.path.join(PATH_DATA, "Poumons/Image/")
path_lab = os.path.join(PATH_DATA, "Poumons/Label/")
tab = pd.read_csv(os.path.join(PATH_DATA, "Poumons/Tableau_General.csv")).values

# Ensemble de Souris présente pour l'entrainement :
numSouris = utils.calcul_numSouris(path_souris)



####################################################################################
  ##################### MODELE 2D DETECTION + SEGMENTATION #####################
####################################################################################

"""
Detection Poumons :
"""

data_seg, label_seg, ind_seg = data.create_data_detect_poum(path_img, tab, numSouris)

no = np.arange(len(ind_seg)) ; no_sample = sample(list(no), len(no))

data_seg = data_seg.reshape(-1, 128, 128, 1)[no_sample].astype('float32')
label_seg = to_categorical(label_seg[no_sample])

model_detect = model.model_detection()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
model_detect.fit(data_seg, label_seg, validation_split=0.2, batch_size=264, epochs=20, callbacks=[es])
model_detect.save(os.path.join(PATH_GIT, "Poumons/model/model_detect.h5"))


"""
Segmentation Poumons :
"""

data_2D, label_2D, ind_2D = data.create_data_seg_poum(path_img, path_lab, tab)

N = np.arange(len(ind_2D)) ; N_sample = sample(list(N), len(N))

data_2D = data_2D.reshape(-1, 128, 128, 1)[N_sample]
label_2D = label_2D.reshape(-1, 128, 128,1)[N_sample]

input_shape = (128,128,1)

model_seg = model.model_unet_2D(input_shape)
earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg.fit(data_2D, label_2D, validation_split=0.2, batch_size=32, epochs=50,callbacks=[earlystopper])
model_seg.save(os.path.join(PATH_GIT, "Poumons/model/model_seg.h5"))




####################################################################################
        ##################### MODELE 2D MULTI-AXES #####################
####################################################################################

"""
Multi-Axes
    Chargement des images de souris annotées + amélioration du contraste
    Chargement des masques + complétions par masque vide 
"""

data_3D, label_3D, ind_3D = data.crate_data_3D(path_img, path_lab, tab, numSouris)

#Data selon axe
data_ax = []; data_sag = []; data_cor = []
label_ax = []; label_sag = []; label_cor = []

for k in np.arange(len(numSouris)):

    for i in np.arange(128):
        data_ax.append(data_3D[k][i, :, :])
        label_ax.append(label_3D[k][i, :, :])

        data_sag.append(data_3D[k][:, i, :])
        label_sag.append(label_3D[k][:, i, :])

        data_cor.append(data_3D[k][:, :, i])
        label_cor.append(label_3D[k][:, :, i])


n = np.arange(len(ind_3D)) ; n_sample = sample(list(n), len(n))

data_ax = np.array(data_ax).reshape(-1, 128, 128, 1)[n_sample]
label_ax = np.array(label_ax, dtype=np.bool).reshape(-1, 128, 128, 1)[n_sample]

data_sag = np.array(data_sag).reshape(-1, 128, 128, 1)[n_sample]
label_sag = np.array(label_sag, dtype=np.bool).reshape(-1, 128, 128, 1)[n_sample]

data_cor = np.array(data_cor).reshape(-1, 128, 128, 1)[n_sample]
label_cor = np.array(label_cor, dtype=np.bool).reshape(-1, 128, 128, 1)[n_sample]

input_shape = (128, 128, 1)


## Modele axial
earlystopper = EarlyStopping(patience=5, verbose=1)
model_axial = model.model_unet_2D(input_shape)
model_axial.fit(data_ax, label_ax, validation_split=0.2, batch_size=32, epochs=50,
                                callbacks=[earlystopper])
model_axial.save(os.path.join(PATH_GIT, "Poumons/model/model_axial.h5"))


## Modele sagital
earlystopper = EarlyStopping(patience=5, verbose=1)
model_sagital = model.model_unet_2D(input_shape)
model_sagital.fit(data_sag, label_sag, validation_split=0.2, batch_size=32, epochs=50,
                  callbacks=[earlystopper])
model_sagital.save(os.path.join(PATH_GIT, "Poumons/model/model_sagital.h5"))


## Modele corronal
earlystopper = EarlyStopping(patience=5, verbose=1)
model_corronal = model.model_unet_2D(input_shape)
model_corronal.fit(data_cor, label_cor, validation_split=0.2, batch_size=32, epochs=50,
                     callbacks=[earlystopper])
model_corronal.save(os.path.join(PATH_GIT, "Poumons/model/model_corronal.h5"))



####################################################################################
        ##################### MODELE RESNET #####################
####################################################################################

resnet = model.ResNet50(input_shape=(128, 128, 1))
resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=[utils.mean_iou])
# En faite, il est possible de construire un U-Net avec encodeur resNet avec la librairie segmentation_models
# Cela peut être utile à essayer.


####################################################################################
        ##################### UNET PLUS PLUS #####################
####################################################################################

data_2D, label_2D, ind_2D = data.create_data_seg_poum(path_img, path_lab, tab)

N = np.arange(len(ind_2D)) ; N_sample = sample(list(N), len(N))

data_2D = data_2D.reshape(-1, 128, 128, 1)[N_sample]
label_2D = label_2D.reshape(-1, 128, 128, 1)[N_sample]

input_shape = (128, 128, 1)

model_seg = model.unet_plusplus(input_shape)
earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg.fit(data_2D, label_2D, validation_split=0.2, batch_size=32, epochs=50,callbacks=[earlystopper])
model_seg.save(os.path.join(PATH_GIT, "model/model_unet_plusplus.h5"))


####################################################################################
        ##################### Small U-Net #####################
####################################################################################

data_2D, label_2D, ind_2D = data.create_data_seg_poum(path_img, path_lab, tab)

N = np.arange(len(ind_2D)) ; N_sample = sample(list(N), len(N))

data_2D = data_2D.reshape(-1, 128, 128, 1)[N_sample]
label_2D = label_2D.reshape(-1, 128, 128, 1)[N_sample]

input_shape = (128,128,1)

# model_seg = model.small_unet(input_shape)
model_seg = model.unetCoupe2Max(input_shape)

earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg.fit(data_2D, label_2D, validation_split=0.2, batch_size=32, epochs=50,callbacks=[earlystopper])

# model_seg.save(os.path.join(PATH_GIT, "Poumons/model/small_unet.h5"))
model_seg.save(os.path.join(PATH_GIT, "Poumons/model/unetCoupe2Max.h5"))



####################################################################################
    ##################### AMELIORATION AVEC CREATIVE DATA #####################
####################################################################################

"""
Segmentation Poumons :
"""

# Charge les anciennes images
data_2D, label_2D, ind_2D = data.create_data_seg_poum(path_img, path_lab, tab)

# Charge les nouvelles images
path_new = os.path.join(PATH_Synth, "Nouvelles_Images/")
newData, newPoum, newMeta = data.recup_new_data(path_new)

# On concatene le tout
Data = utils.concat_data(data_2D, newData)
MaskPoum = utils.concat_data(label_2D, newPoum)

# Puis on procède comme précedemment
N = np.arange(Data.shape[0]) ; N_sample = sample(list(N), len(N))

Data = Data.reshape(-1, 128, 128, 1)[N_sample]
MaskPoum = MaskPoum.reshape(-1, 128, 128, 1)[N_sample]

input_shape = (128, 128, 1)

model_seg = model.unet_plusplus(input_shape)
earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg.fit(Data, MaskPoum, validation_split=0.2, batch_size=32, epochs=50,callbacks=[earlystopper])
model_seg.save(os.path.join(PATH_GIT, "Poumons/model/model_seg_poum_creat.h5"))


