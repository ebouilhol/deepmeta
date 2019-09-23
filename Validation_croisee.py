import pandas as pd
import utils
import random
import numpy as np
from skimage import io, exposure
import model
from keras.callbacks import EarlyStopping
import os
import scipy
import sys

ROOT_DIR = os.path.abspath("/home/achauviere/Bureau/Projet_Detection_Metastase_Souris/")
sys.path.append(ROOT_DIR)

# PATH_GIT = "./Antoine_Git/"
PATH_GIT = os.path.join(ROOT_DIR, "./Antoine_Git/")

# PATH_DATA = "./DATA/"
PATH_DATA = os.path.join(ROOT_DIR, "./DATA/")


### Optimisation du batch size à utiliser pour le U-Net ###

path_result = os.path.join(PATH_GIT, "Poumons/stats/Batch_size/")
Result = np.zeros((10, 4, 6))  # 10 essais, 4 mesure, 6 batchsize

for k in range(10):

    path_souris = PATH_DATA + "Souris/"
    path_mask = PATH_DATA + "Masques/"
    path_img = PATH_DATA + "Image/"
    path_lab = PATH_DATA + "Label/"
    tab = pd.read_csv(PATH_DATA + "Tableau_General.csv").values

    numSouris = utils.calcul_numSouris(path_souris)
    numSouris_sample = random.sample(list(numSouris), 24)

    X_train = [];  X_test = [];  y_train = []; y_test = []
    ind_train = []; ind_test = []

    for i in np.arange(len(tab)):
        if tab[i, 4] == 1:
            im = io.imread(path_img + 'img_' + str(i) + '.tif', plugin='tifffile')
            img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03)

            if tab[i, 1] in numSouris_sample:
                X_train.append(img_adapteq)
                y_train.append(io.imread(path_lab + 'm_' + str(i) + '.tif'))
                ind_train.append(i)
            else:
                X_test.append(img_adapteq)
                y_test.append(io.imread(path_lab + 'm_' + str(i) + '.tif'))
                ind_test.append(i)

    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=np.bool)
    X_test = np.array(X_test)
    y_test = np.array(y_test, dtype=np.bool)

    N = np.arange(len(ind_train)) ; N_sample = random.sample(list(N), len(N))

    X_train = X_train.reshape(-1, 128, 128, 1)[N_sample]
    y_train = y_train.reshape(-1, 128, 128, 1)[N_sample]

    batch_size = [2, 4, 8, 16, 32, 64]
    for j in range(6):

        #fit model
        input_shape = (128, 128, 1)
        model_seg = model.model_unet_2D(input_shape)
        earlystopper = EarlyStopping(patience=5, verbose=1)
        model_seg.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size[j], epochs=70,
                      callbacks=[earlystopper])

        #pred
        X_test = X_test.reshape(-1, 128, 128, 1)
        pred = (model_seg.predict(X_test) > 0.5).astype(np.uint8).reshape(X_test.shape[0], 128, 128)
        IoU = [utils.stats_pixelbased(y_test[j], pred[j]).get('IoU') for j in np.arange(X_test.shape[0])]

        Result[k, 0, j] = np.mean(IoU)
        Result[k, 1, j] = np.median(IoU)
        Result[k, 2, j] = np.var(IoU)
        Result[k, 3, j] = scipy.stats.mstats.gmean(IoU)

label = ["Mean", "Median", "Var", "Moy_Geom"]

if not os.path.exists(path_result):
    os.makedirs(path_result)

for j in range(6):
    df = pd.DataFrame(Result[:, :, j], columns=label)
    df.to_csv(path_result + "/batch_"+str(batch_size[j])+".csv")