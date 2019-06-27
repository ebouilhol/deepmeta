########################################################################################################################
########################################################################################################################
#                                              SEGMENTATION DES POUMONS                                                #
########################################################################################################################
########################################################################################################################

import utils
import os
import numpy as np
import pandas as pd
import re
from skimage import exposure, io
from random import sample
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import model


#### Path ####

#path_souris = "/home/achauviere/Bureau/DATA/Souris/"
path_souris = "./DATA/Souris/"
#path_mask = "/home/achauviere/Bureau/DATA/Masques/"
path_mask = "./DATA/Masques/"
#path_img = "/home/achauviere/Bureau/DATA/Image/"
path_img = "./DATA/Image/"
#path_lab = "/home/achauviere/Bureau/DATA/Label/"
path_lab = "./DATA/Label/"
#tab = pd.read_csv("/home/achauviere/Bureau/DATA/Tableau_General.csv").values
tab = pd.read_csv("./DATA/Tableau_General.csv").values




#### Ensemble de Souris présente pour l'entrainement ####
list_souris = utils.sorted_aphanumeric(os.listdir(path_souris))
numSouris = []
for k in np.arange(len(list_souris)):
    numSouris.append(int(re.findall('\d+',list_souris[k])[0]))





####################################################################################
  ##################### MODELE 2D DETECTION + SEGMENTATION #####################
####################################################################################

"""
Detection Poumons :
    Chargement des images de souris annotées + amélioration du contraste
    Chargement des masques + complétions par masque vide
"""


list_img = utils.sorted_aphanumeric(os.listdir(path_img))

data_seg = []
label_seg = []
ind_seg = []

for i in np.arange(len(tab)):

    if tab[i, 1] in numSouris:
        im = io.imread(path_img + 'img_' + str(i) + '.tif', plugin='tifffile')
        img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03)
        data_seg.append(img_adapteq)
        label_seg.append(tab[i, 3])
        ind_seg.append(i)

data_seg = np.array(data_seg)
label_seg = np.array(label_seg)
ind_seg = np.array(ind_seg)


no = np.arange(len(ind_seg)) ; no_sample = sample(list(no), len(no))

data_seg = data_seg.reshape(-1, 128,128, 1)[no_sample].astype('float32')
label_seg = to_categorical(label_seg[no_sample])

model_detect = model.model_detection()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
model_detect.fit(data_seg, label_seg, validation_split=0.2, batch_size=264, epochs=20, callbacks=[es])
#model_detect.save('/home/achauviere/Bureau/2D_and_Multi2D/Detect_Seg/model_detect.h5')
model_detect.save('./2D_and_Multi2D/Detect_Seg/model_detect.h5')



"""
Segmentation Poumons :
    Chargement des Images et des Masques : que ceux où poumon == 1 sur csv.
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

N = np.arange(len(ind_2D)) ; N_sample = sample(list(N), len(N))

data_2D = data_2D.reshape(-1, 128,128, 1)[N_sample]
label_2D = label_2D.reshape(-1,128,128,1)[N_sample]

input_shape = (128,128,1)

model_seg = model.model_unet_2D(input_shape)
earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg.fit(data_2D, label_2D, validation_split=0.2, batch_size=32, epochs=50,callbacks=[earlystopper])
#model_seg.save('/home/achauviere/Bureau/2D_and_Multi2D/Detect_Seg/model_seg.h5')
model_seg.save('./2D_and_Multi2D/Detect_Seg/model_seg.h5')






####################################################################################
        ##################### MODELE 2D MULTI-AXES #####################
####################################################################################

"""
Multi-Axes
    Chargement des images de souris annotées + amélioration du contraste
    Chargement des masques + complétions par masque vide 
"""

list_img = utils.sorted_aphanumeric(os.listdir(path_img))

data_3D = []
label_3D = []
ind_3D = []

for i in np.arange(len(tab)):
    if tab[i, 1] in numSouris:
        im = io.imread(path_img + 'img_' + str(i) + '.tif', plugin='tifffile')
        img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03)
        data_3D.append(img_adapteq)
        ind_3D.append(i)
        if tab[i, 4] == 1:
            label_3D.append(io.imread(path_lab + 'm_' + str(i) + '.tif', plugin='tifffile'))
        else:
            label_3D.append(np.zeros((128, 128)))

data_3D = np.array(data_3D)
label_3D = np.array(label_3D)
ind_3D = np.array(ind_3D)



#Reconstruction en souris
data_ = np.zeros((((len(numSouris),128,128,128))))
label_ = np.zeros((((len(numSouris),128,128,128))))

for i in np.arange(len(numSouris)):
    data_[i] = data_3D[(128*i):(128*(i+1))]
    label_[i] = label_3D[(128*i):(128*(i+1))]

data_3D = data_
label_3D = label_



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

data_ax = np.array(data_ax).reshape(-1, 128,128, 1)[n_sample]
label_ax = np.array(label_ax, dtype=np.bool).reshape(-1, 128,128, 1)[n_sample]

data_sag = np.array(data_sag).reshape(-1, 128,128, 1)[n_sample]
label_sag = np.array(label_sag, dtype=np.bool).reshape(-1, 128,128, 1)[n_sample]

data_cor = np.array(data_cor).reshape(-1, 128,128, 1)[n_sample]
label_cor = np.array(label_cor, dtype=np.bool).reshape(-1, 128,128, 1)[n_sample]

input_shape = (128,128,1)


## Modele axial
earlystopper = EarlyStopping(patience=5, verbose=1)
model_axial = model.model_unet_2D(input_shape)
model_axial.fit(data_ax, label_ax, validation_split=0.2, batch_size=32, epochs=50,
                                callbacks=[earlystopper])
#model_axial.save('/home/achauviere/Bureau/2D_and_Multi2D/Multi_Axes_Seg/model_axial.h5')
model_axial.save('./2D_and_Multi2D/Multi_Axes_Seg/model_axial.h5')


## Modele sagital
earlystopper = EarlyStopping(patience=5, verbose=1)
model_sagital = model.model_unet_2D(input_shape)
model_sagital.fit(data_sag, label_sag, validation_split=0.2, batch_size=32, epochs=50,
                  callbacks=[earlystopper])
#model_sagital.save('/home/achauviere/Bureau/2D_and_Multi2D/Multi_Axes_Seg/model_sagital.h5')
model_sagital.save('./2D_and_Multi2D/Multi_Axes_Seg/model_sagital.h5')


## Modele corronal
earlystopper = EarlyStopping(patience=5, verbose=1)
model_corronal = model.model_unet_2D(input_shape)
model_corronal.fit(data_cor, label_cor, validation_split=0.2, batch_size=32, epochs=50,
                     callbacks=[earlystopper])
#model_corronal.save('/home/achauviere/Bureau/2D_and_Multi2D/Multi_Axes_Seg/model_corronal.h5')
model_corronal.save('./2D_and_Multi2D/Multi_Axes_Seg/model_corronal.h5')