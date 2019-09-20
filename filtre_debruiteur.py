import pandas as pd
import os
import re
import numpy as np
import utils
import data

path_souris = "/home/achauviere/Bureau/DATA/Souris/"
path_mask = "/home/achauviere/Bureau/DATA/Masques/"
path_img = "/home/achauviere/Bureau/DATA/Image/"
path_lab = "/home/achauviere/Bureau/DATA/Label/"
tab = pd.read_csv("/home/achauviere/Bureau/DATA/Tableau_General.csv").values

path_med = "/home/achauviere/Bureau/Filtre_debruiteur/Median/"
path_moy = "/home/achauviere/Bureau/Filtre_debruiteur/Mean/"
path_max = "/home/achauviere/Bureau/Filtre_debruiteur/Max/"



list_souris = utils.sorted_aphanumeric(os.listdir(path_souris))
numSouris = []
for k in np.arange(len(list_souris)):
    numSouris.append(int(re.findall('\d+',list_souris[k])[0]))


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