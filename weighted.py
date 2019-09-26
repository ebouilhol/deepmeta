import numpy as np
import data
from random import sample
from keras.callbacks import EarlyStopping
import pandas as pd
import model
import os
import utils
import sys

# Choix de path
console = False
if console:
    ROOT_DIR = os.path.abspath("/home/achauviere/Bureau/Projet_Detection_Metastase_Souris/deepmeta-master/")
    sys.path.append(ROOT_DIR)
    PATH_GIT = os.path.join(ROOT_DIR, "../deepmeta-master/")
    PATH_DATA = os.path.join(ROOT_DIR, "../DATA/")
    PATH_Synth = os.path.join(ROOT_DIR, "../Data_Synthetique/")
else:
    PATH_GIT = "../deepmeta-master/"
    PATH_DATA = "../DATA/"
    PATH_Synth = "../Data_Synthetique/"


########################################################################
    ###### Augmenter poids des contours + interieur (poumons) ######
########################################################################

path_souris = os.path.join(PATH_DATA, "Poumons/Souris/")
path_mask = os.path.join(PATH_DATA, "Poumons/Masques/")
path_img = os.path.join(PATH_DATA, "Poumons/Image/")
path_lab = os.path.join(PATH_DATA, "Poumons/Label/")
tab = pd.read_csv(os.path.join(PATH_DATA, "Tableau_General.csv")).values

numSouris = utils.calcul_numSouris(path_souris)

# Data réelle
data_2D, label_2D, ind_2D = data.create_data_seg_poum(path_img, path_lab, tab)

# Data Créative
path_new = os.path.join(PATH_Synth, "Nouvelles_Images/")
newData, newPoum, newMeta = data.recup_new_data(path_new)

data_2D = utils.concat_data(data_2D, newData)
label_2D = utils.concat_data(label_2D, newPoum)

# weight_2D = utils.weight_map(label_2D, 4, 9)
weight_2D = utils.weight_map(label_2D, 2, 4)

N = np.arange(len(ind_2D)) ; N_sample = sample(list(N), len(N))

data_2D = data_2D.reshape(-1, 128, 128, 1)[N_sample]
label_2D = label_2D.reshape(-1, 128, 128, 1)[N_sample]
weight_2D = weight_2D.reshape(-1, 128,128, 1)[N_sample]

y = np.zeros((data_2D.shape[0], 128, 128, 2))
y[:, :, :, 0] = label_2D[:, :, :, 0]
y[:, :, :, 1] = weight_2D[:, :, :, 0]

input_shape = (128, 128, 1)

model_seg = model.model_unet_2D(input_shape, wei=True)

earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg.fit(data_2D, y, validation_split=0.2, batch_size=32, epochs=50, callbacks=[earlystopper])

# model_seg.save(os.path.join(PATH_GIT, "Poumons/model/weight_map149.h5"))
# model_seg.save(os.path.join(PATH_GIT, "Poumons/model/weight_map124.h5"))
model_seg.save(os.path.join(PATH_GIT, "Poumons/model/weight_map_creative149.h5"))


########################################################################
    ###### Augmenter poids des contours + interieur (métas) ######
########################################################################

path_souris = os.path.join(PATH_DATA, "Metastases/Souris/")
path_mask = os.path.join(PATH_DATA, "Metastases/Masques/")
path_img = os.path.join(PATH_DATA, "Metastases/Image/")
path_lab = os.path.join(PATH_DATA, "Metastases/Label/")
tab = pd.read_csv(os.path.join(PATH_DATA, "Tableau.csv")).values

numSouris = utils.calcul_numSouris(path_souris)

# Data réelle
data_2D, label_2D, ind_2D = data.create_data_meta(path_img, path_lab, tab)

# Data Créative
path_new = os.path.join(PATH_Synth, "Nouvelles_Images/")
newData, newPoum, newMeta = data.recup_new_data(path_new)

data_2D = utils.concat_data(data_2D, newData)
label_2D = utils.concat_data(label_2D, newMeta)

weight_2D = utils.weight_map(label_2D, 4, 9)
# weight_2D = utils.weight_map(label_2D, 10, 50)

N = np.arange(len(ind_2D)) ; N_sample = sample(list(N), len(N))

data_2D = data_2D.reshape(-1, 128, 128, 1)[N_sample]
label_2D = label_2D.reshape(-1, 128, 128, 1)[N_sample]
weight_2D = weight_2D.reshape(-1, 128, 128, 1)[N_sample]

y = np.zeros((data_2D.shape[0], 128, 128, 2))
y[:, :, :, 0] = label_2D[:, :, :, 0]
y[:, :, :, 1] = weight_2D[:, :, :, 0]

input_shape = (128, 128, 1)

model_seg = model.model_unet_2D(input_shape, wei=True)

earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg.fit(data_2D, y, validation_split=0.1, batch_size=16, epochs=70, callbacks=[earlystopper])

# model_seg.save(os.path.join(PATH_GIT, "Metastases/model/weight_map149.h5"))
# model_seg.save(os.path.join(PATH_GIT, "Metastases/model/weight_map11050.h5"))
model_seg.save(os.path.join(PATH_GIT, "Metastases/model/weight_map_creative149.h5"))
# model_seg.save(os.path.join(PATH_GIT, "Metastases/model/weight_map_creative11050.h5"))