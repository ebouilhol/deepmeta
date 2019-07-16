import numpy as np
import data
from random import sample
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd
import model
import os
import re
import utils


########################################################################
   ############### Comparaison beta pour WCE (méta) ###############
########################################################################

path_souris = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Souris/"
path_mask = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Masques/"
path_img = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Image/"
path_lab = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Label/"
tab = pd.read_csv("/home/achauviere/Bureau/Annotation_Meta/Tableau.csv").values


Result = np.zeros((7, 10))
Beta = [0.5, 0.7, 1, 1.5, 2, 3, 5]


for k in np.arange(10):

    Data, Label, ind = data.create_data_meta(path_img, path_lab, tab)

    N = np.arange(len(ind))
    N_sample = sample(list(N), len(N))

    Data = Data.reshape(-1, 128, 128, 1)[N_sample]
    label = Label.reshape(-1, 128, 128, 1)[N_sample]

    X_train, X_test, y_train, y_test = train_test_split(Data, label, test_size=0.1)

    i = 0
    for b in Beta:

        input_shape = (128, 128, 1)
        model_seg = model.model_unet_2D(input_shape, beta=b)

        earlystopper = EarlyStopping(patience=5, verbose=1)
        model_seg.fit(X_train, y_train, validation_split=0.2, batch_size=16, epochs=1, callbacks=[earlystopper])

        seg = (model_seg.predict(X_test) > 0.5).astype(np.uint8).reshape(X_test.shape[0], 128, 128)

        res = np.zeros(X_test.shape[0])
        for j in np.arange(X_test.shape[0]):
            intersection = np.logical_and(seg[j], y_test[j])
            union = np.logical_or(seg[j], y_test[j])
            res[j] = intersection.sum() / union.sum()

        Result[i, k] = np.nanmean(res)
        i += 1



########################################################################
    ###### Augmenter poids des contours + interieur (poumons) ######
########################################################################

path_souris = "/home/achauviere/Bureau/DATA/Souris/"
path_mask = "/home/achauviere/Bureau/DATA/Masques/"
path_img = "/home/achauviere/Bureau/DATA/Image/"
path_lab = "/home/achauviere/Bureau/DATA/Label/"
tab = pd.read_csv("/home/achauviere/Bureau/DATA/Tableau_General.csv").values


list_souris = utils.sorted_aphanumeric(os.listdir(path_souris))
numSouris = []
for k in np.arange(len(list_souris)):
    numSouris.append(int(re.findall('\d+',list_souris[k])[0]))

# Data réelle
data_2D, label_2D, ind_2D = data.create_data_seg_poum(path_img,path_lab,tab)

# Data Créative
newData, newPoum, newMeta = data.recup_new_data()

data_2D = utils.concat_data(data_2D,newData)
label_2D = utils.concat_data(label_2D,newPoum)

#weight_2D = utils.weight_map(label_2D,4,9)
weight_2D = utils.weight_map(label_2D,2,4)

N = np.arange(len(ind_2D)) ; N_sample = sample(list(N), len(N))

data_2D = data_2D.reshape(-1, 128,128, 1)[N_sample]
label_2D = label_2D.reshape(-1,128,128,1)[N_sample]
weight_2D = weight_2D.reshape(-1,128,128,1)[N_sample]

y = np.zeros((((data_2D.shape[0], 128, 128, 2))))
y[:,:,:,0] = label_2D[:,:,:,0]
y[:,:,:,1] = weight_2D[:,:,:,0]

input_shape = (128,128,1)

model_seg = model.model_unet_2D(input_shape, wei=True)

earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg.fit(data_2D, y, validation_split=0.2, batch_size=32, epochs=50, callbacks=[earlystopper])

#model_seg.save('/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/weight_map149.h5')
#model_seg.save('/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/weight_map124.h5')
model_seg.save('/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/weight_map_creative149.h5')




########################################################################
    ###### Augmenter poids des contours + interieur (métas) ######
########################################################################

path_souris = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Souris/"
path_mask = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Masques/"
path_img = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Image/"
path_lab = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Label/"
tab = pd.read_csv("/home/achauviere/Bureau/Annotation_Meta/Tableau.csv").values

list_souris = utils.sorted_aphanumeric(os.listdir(path_souris))
numSouris = []
for k in np.arange(len(list_souris)):
    numSouris.append(int(re.findall('\d+',list_souris[k])[0]))


# Data réelle
data_2D, label_2D, ind_2D = data.create_data_meta(path_img,path_lab,tab)

# Data Créative
newData, newPoum, newMeta = data.recup_new_data()

data_2D = utils.concat_data(data_2D,newData)
label_2D = utils.concat_data(label_2D,newMeta)

weight_2D = utils.weight_map(label_2D,4,9)
#weight_2D = utils.weight_map(label_2D,10,50)

N = np.arange(len(ind_2D)) ; N_sample = sample(list(N), len(N))

data_2D = data_2D.reshape(-1, 128,128, 1)[N_sample]
label_2D = label_2D.reshape(-1,128,128,1)[N_sample]
weight_2D = weight_2D.reshape(-1,128,128,1)[N_sample]

y = np.zeros((((data_2D.shape[0], 128, 128, 2))))
y[:,:,:,0] = label_2D[:,:,:,0]
y[:,:,:,1] = weight_2D[:,:,:,0]

input_shape = (128,128,1)

model_seg = model.model_unet_2D(input_shape,wei=True)

earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg.fit(data_2D, y, validation_split=0.1, batch_size=16, epochs=70, callbacks=[earlystopper])

#model_seg.save('/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/weight_map149.h5')
#model_seg.save('/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/weight_map11050.h5')
model_seg.save('/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/weight_map_creative149.h5')
#model_seg.save('/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/weight_map_creative11050.h5')
















path_souris = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Souris/"
path_mask = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Masques/"
path_img = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Image/"
path_lab = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Label/"
tab = pd.read_csv("/home/achauviere/Bureau/Annotation_Meta/Tableau.csv").values

list_souris = utils.sorted_aphanumeric(os.listdir(path_souris))
numSouris = []
for k in np.arange(len(list_souris)):
    numSouris.append(int(re.findall('\d+',list_souris[k])[0]))


# Data réelle
data_2D, label_2D, ind_2D = data.create_data_meta(path_img,path_lab,tab)

# Data Créative
newData, newPoum, newMeta = data.recup_new_data()

data_2D = utils.concat_data(data_2D,newData)
label_2D = utils.concat_data(label_2D,newMeta)

#weight_2D = utils.weight_map(label_2D,4,9)
weight_2D = utils.weight_map(label_2D,10,50)

N = np.arange(len(ind_2D)) ; N_sample = sample(list(N), len(N))

data_2D = data_2D.reshape(-1, 128,128, 1)[N_sample]
label_2D = label_2D.reshape(-1,128,128,1)[N_sample]
weight_2D = weight_2D.reshape(-1,128,128,1)[N_sample]

y = np.zeros((((data_2D.shape[0], 128, 128, 2))))
y[:,:,:,0] = label_2D[:,:,:,0]
y[:,:,:,1] = weight_2D[:,:,:,0]

input_shape = (128,128,1)

model_seg = model.model_unet_2D(input_shape,wei=True)

earlystopper = EarlyStopping(patience=5, verbose=1)
model_seg.fit(data_2D, y, validation_split=0.1, batch_size=16, epochs=70, callbacks=[earlystopper])

model_seg.save('/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/weight_map_creative11050.h5')