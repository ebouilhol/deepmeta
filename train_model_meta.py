########################################################################################################################
########################################################################################################################
#                                              SEGMENTATION DES METASTASES                                             #
########################################################################################################################
########################################################################################################################

"""
Ce script présente mes prémiers essais pour la segmentation des métastases.
La méthode finale utilisée est présentée sur le notebook "Segmentation Métastase"
"""

import pandas as pd
import numpy as np
from skimage import io, exposure
from random import sample
import model
from keras.callbacks import EarlyStopping
import os
import sys
import utils
import data

ROOT_DIR = os.path.abspath("/home/achauviere/Bureau/Projet_Detection_Metastase_Souris/")
sys.path.append(ROOT_DIR)

# PATH_GIT = "./Antoine_Git/"
PATH_GIT = os.path.join(ROOT_DIR, "./Antoine_Git/")

# PATH_DATA = "./DATA/"
PATH_DATA = os.path.join(ROOT_DIR, "./DATA/")

path_souris = os.path.join(PATH_DATA, "Metastases/Souris/")
path_mask = os.path.join(PATH_DATA, "Metastases/Masques/")
path_img = os.path.join(PATH_DATA, "Metastases/Image/")
path_lab = os.path.join(PATH_DATA, "Metastases/Label/")
tab = pd.read_csv(os.path.join(PATH_DATA, "Metastases/Tableau.csv")).values


########################################################################################################################
    ######################## Segmentation des Meta - UNET - Image originale ####################################
########################################################################################################################

Data, Label, ind = data.create_data_meta(path_img, path_lab, tab)

N = np.arange(len(ind)) ; N_sample = sample(list(N), len(N))

Data = Data.reshape(-1, 128, 128, 1)[N_sample]
label = Label.reshape(-1, 128, 128, 1)[N_sample]

input_shape = (128,128,1)

model_seg = model.model_unet_2D(input_shape)

earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg.fit(Data, label, validation_split=0.2, batch_size=32, epochs=70, callbacks=[earlystopper])

model_seg.save(os.path.join(PATH_GIT, "Metastases/model/unet_test.h5"))



########################################################################################################################
   ######################## Segmentation des Meta - UNET - Image Poumon Segmenté ####################################
########################################################################################################################

path_lab_poum = os.path.join(PATH_DATA, "Poumons/Label/")

data = []
label_meta = []
label_poum = []
ind = []

for i in np.arange(len(tab)):
    if tab[i, 5] == 1:
        if tab[i, 6] != 0:
            im = io.imread(path_img + 'img_'+str(i)+'.tif', plugin='tifffile')
            img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03)
            data.append(img_adapteq)
            label_meta.append(io.imread(path_lab + 'm_'+str(i)+'.tif'))
            label_poum.append(io.imread(path_lab_poum + 'm_'+str(i)+'.tif'))
            ind.append(i)

data = np.array(data)
label_meta = np.array(label_meta, dtype=np.bool)
label_poum = np.array(label_poum, dtype=np.bool)
ind = np.array(ind)

data = (data - data.min())*255/(data.max()-data.min())

N = np.arange(len(ind)) ; N_sample = sample(list(N), len(N))

DATA = []
LABEL = []

for i in np.arange(data.shape[0]):
    DATA.append(utils.apply_mask_and_noise(data[i], label_poum[i], 180))
    LABEL.append(utils.apply_mask(label_meta[i], label_poum[i]))

DATA = np.array(DATA).reshape(-1, 128, 128, 1)[N_sample]
LABEL = np.array(LABEL, dtype='bool').reshape(-1, 128, 128, 1)[N_sample]

input_shape = (128, 128, 1)
model_seg2 = model.model_unet_2D(input_shape)
earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg2.fit(DATA, LABEL, validation_split=0.2, batch_size=32, epochs=70, callbacks=[earlystopper])

model_seg2.save(os.path.join(PATH_GIT, "Metastases/model/unet_test2.h5"))

# le modèle ne s'optimise pas, je ne l'ai pas conserver (il est possible de relancer l'entrainement).




####################################################################################
    ##################### AMELIORATION AVEC CREATIVE DATA #####################
####################################################################################


Data, Label, ind = data.create_data_meta(path_img, path_lab, tab)

newData, newPoum, newMeta = data.recup_new_data()

Data = utils.concat_data(Data, newData)
MaskMeta = utils.concat_data(Label, newMeta)

N = np.arange(Data.shape[0]) ; N_sample = sample(list(N), len(N))

Data = Data.reshape(-1, 128, 128, 1)[N_sample]
MaskMeta = MaskMeta.reshape(-1, 128, 128, 1)[N_sample]

input_shape = (128, 128, 1)

model_seg = model.model_unet_2D(input_shape)

earlystopper = EarlyStopping(patience=5, verbose=1)

model_seg.fit(Data, MaskMeta, validation_split=0.2, batch_size=32, epochs=70,callbacks=[earlystopper])

# model_seg.save(os.path.join(PATH_GIT, Metastases/model/unet_creative_diceloss.h5'))
# J'ai supprimer ce modèle car meilleure approche sur notebook.


