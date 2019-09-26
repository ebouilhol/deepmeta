import pandas as pd
import os
import numpy as np
import utils
import data
import sys
import cv2

# Choix de path
console = False
if console:
    ROOT_DIR = os.path.abspath("/home/achauviere/Bureau/Projet_Detection_Metastase_Souris/deepmeta-master/")
    sys.path.append(ROOT_DIR)
    PATH_Data_contr = os.path.join(ROOT_DIR, "../Data_contraste/")
    PATH_DATA = os.path.join(ROOT_DIR, "../DATA/")
else:
    PATH_Data_contr = "../Data_contraste/"
    PATH_DATA = "../DATA/"

path_souris = os.path.join(PATH_DATA, "Poumons/Souris/")
path_mask = os.path.join(PATH_DATA, "Poumons/Masques/")
path_img = os.path.join(PATH_DATA, "Poumons/Image/")
path_lab = os.path.join(PATH_DATA, "Poumons/Label/")
tab = pd.read_csv(os.path.join(PATH_DATA, "Poumons/Tableau_General.csv")).values

path_med = os.path.join(PATH_Data_contr, "Filtre_debruiteur/Median/")
path_moy = os.path.join(PATH_Data_contr,  "Filtre_debruiteur/Mean/")
path_max = os.path.join(PATH_Data_contr,  "Filtre_debruiteur/Max/")

numSouris = utils.calcul_numSouris(path_souris)

data_3D, label_3D, ind_3D = data.crate_data_3D(path_img, path_lab, tab, numSouris, etale=True)

for n in np.arange(len(numSouris)):

    os.mkdir(path_med + "Souris_" + str(numSouris[n]))
    os.mkdir(path_moy + "Souris_" + str(numSouris[n]))
    os.mkdir(path_max + "Souris_" + str(numSouris[n]))

    test_im = data_3D[n]

    new_im_med = np.zeros((((len(numSouris), 128, 128, 128))))
    new_im_moy = np.zeros((((len(numSouris), 128, 128, 128))))
    new_im_max = np.zeros((((len(numSouris), 128, 128, 128))))

    cv2.imwrite(path_med + "Souris_" + str(n) + "/im_0.tif", test_im[0])
    cv2.imwrite(path_moy + "Souris_" + str(n) + "/im_0.tif", test_im[0])
    cv2.imwrite(path_max + "Souris_" + str(n) + "/im_0.tif", test_im[0])

    for k in (np.arange(126) + 1):
        for i in np.arange(128):
            for j in np.arange(128):
                new_im_med[n][k][i, j] = int(np.median([test_im[k - 1][i, j], test_im[k][i, j], test_im[k + 1][i, j]]))
                new_im_moy[n][k][i, j] = int(np.mean([test_im[k - 1][i, j], test_im[k][i, j], test_im[k + 1][i, j]]))
                new_im_max[n][k][i, j] = int(np.max([test_im[k - 1][i, j], test_im[k][i, j], test_im[k + 1][i, j]]))

        cv2.imwrite(path_med + "Souris_" + str(numSouris[n]) + "/im_" + str(k) + ".tif", new_im_med[n][k])
        cv2.imwrite(path_moy + "Souris_" + str(numSouris[n]) + "/im_" + str(k) + ".tif", new_im_moy[n][k])
        cv2.imwrite(path_max + "Souris_" + str(numSouris[n]) + "/im_" + str(k) + ".tif", new_im_max[n][k])

    cv2.imwrite(path_med + "Souris_" + str(numSouris[n]) + "/im_127.tif", test_im[127])
    cv2.imwrite(path_moy + "Souris_" + str(numSouris[n]) + "/im_127.tif", test_im[127])
    cv2.imwrite(path_max + "Souris_" + str(numSouris[n]) + "/im_127.tif", test_im[127])