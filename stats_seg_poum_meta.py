import pandas as pd
import utils
import os
import numpy as np
from skimage import io
import model
import sys

# Choix de path
console = False
if console:
    ROOT_DIR = os.path.abspath("/home/achauviere/Bureau/Projet_Detection_Metastase_Souris/deepmeta-master/")
    sys.path.append(ROOT_DIR)
    PATH_GIT = os.path.join(ROOT_DIR, "../deepmeta-master/")
    PATH_DATA = os.path.join(ROOT_DIR, "../DATA/")
else:
    PATH_GIT = "../deepmeta-master/"
    PATH_DATA = "../DATA/"


########################################################################################################################
########################################################################################################################
#                                              SEGMENTATION DES POUMONS                                                #
########################################################################################################################
########################################################################################################################

for n_souris in [8, 28, 56]:

    ## On récupère les paths
    path_souris = os.path.join(PATH_DATA, "Souris_Test/Souris/souris_"+str(n_souris)+".tif")
    path_souris_annoter = os.path.join(PATH_DATA, "Souris_Test/Masque_Poumons/masque_"+str(n_souris)+"/")
    tab = pd.read_csv(os.path.join(PATH_DATA, "Poumons/Tableau_General.csv")).values

    ## On récupère le masque annoté
    list_msk = utils.sorted_aphanumeric(os.listdir(path_souris_annoter))
    y = []
    for i in np.arange(len(list_msk)):
        y.append(io.imread(path_souris_annoter + list_msk[i], plugin="tifffile"))
    seg_true = np.array(y, dtype='bool')

    ## On récupère la détection annoté
    tab2 = tab[np.where(tab[:, 1] == n_souris)]
    detect_annot = tab2[:, 2][np.where(tab2[:, 3] == 1)]
    n = len(detect_annot)  # nbr d'images présentant des poumons

    ## On récupère les paths des modèles de segmentation et pour la détection
    path_model_detect = os.path.join(PATH_GIT, "Poumons/model/model_detect.h5")

    path_model_seg = [os.path.join(PATH_GIT, "Poumons/model/model_seg.h5"),
                      os.path.join(PATH_GIT, "Poumons/model/model_seg_poum_creat.h5"),
                      os.path.join(PATH_GIT, "Poumons/model/weight_map149.h5"),
                      os.path.join(PATH_GIT, "Poumons/model/weight_map124.h5"),
                      os.path.join(PATH_GIT, "Poumons/model/weight_map_creative149.h5"),
                      os.path.join(PATH_GIT, "Poumons/model/unetCoupe.h5"),
                      os.path.join(PATH_GIT, "Poumons/model/unetCoupe2Max.h5"),
                      os.path.join(PATH_GIT, "Poumons/model/model_unet_plusplus.h5"),
                      os.path.join(PATH_GIT, "Poumons/model/small_unet.h5")]

    path_model_axial = os.path.join(PATH_GIT, "Poumons/model/model_axial.h5")
    path_model_sagital = os.path.join(PATH_GIT, "Poumons/model/model_sagital.h5")
    path_model_corronal = os.path.join(PATH_GIT, "Poumons/model/model_corronal.h5")
    vote = [1, 2 , 3]

    path_result = ["a", "a", "b", "b", "b", "c", "c", "c", "a", "a", "a", 'a']  # a: ds non weight / b: ds weight / c: multi axes
    name_folder = "Souris_" + str(n_souris)

    label = ['Detect + U-Net', 'Detect + U-Net + Creat9', 'Detect + U-Net + WM149', 'Detect + U-Net + WM124',
             'Detect + U-Net + Creat + WM149', 'U_Net Multi 1', 'U_Net Multi 2', 'U_Net Multi 3', 'U-Net Coupé',
             'U-Net Manu', 'U-Net ++', 'Small U-Net']


    ## On calcule les résultats obtenus par les différents modèles
    ResultIoU = np.zeros((len(path_result), n))
    ResultDice = np.zeros((len(path_result), n))
    v = 0

    for i in np.arange(len(path_result)):

        ## On charge le modèle
        if path_result[i] == "a":
            detect, seg = model.methode_detect_seg(path_souris, path_model_detect, path_model_seg[i-v], path_result[i],
                                                   name_folder, mask=True, visu_seg=None)
        elif path_result[i] == 'b':
            detect, seg = model.methode_detect_seg(path_souris, path_model_detect, path_model_seg[i-v], path_result[i],
                                                   name_folder, mask=True, visu_seg=None, wei=True)

        else:
            seg = model.methode_multi_axe(path_souris, path_model_axial, path_model_sagital, path_model_corronal,
                                    path_result, name_folder, mask=True, vote_model=vote[v], visu_seg=False)
            v += 1


        ## On récupère les prédiction en fonction des vrais indices de détection
        seg_pred = seg[detect_annot]

        ## On calcule l'IoU pour chaque image
        IoU = [utils.stats_pixelbased(seg_true[j], seg_pred[j]).get('IoU') for j in np.arange(n)]
        Dice = [utils.stats_pixelbased(seg_true[j], seg_pred[j]).get('Dice') for j in np.arange(n)]

        ## On les stock dans la matrice de Résultats
        ResultIoU[i, :] = np.array(IoU)
        ResultDice[i, :] = np.array(Dice)
        print(i)

    df_iou = pd.DataFrame(ResultIoU.T, columns=label)
    df_iou.to_csv(os.path.join(PATH_GIT, "Poumons/stats/Souris_"+str(n_souris)+"/Result_IoU.csv"), index=None, header=True)
    df_dice = pd.DataFrame(ResultDice.T, columns=label)
    df_dice.to_csv(os.path.join(PATH_GIT, "Poumons/stats/Souris_"+str(n_souris)+"/Result_Dice.csv"), index = None, header=True)






########################################################################################################################
########################################################################################################################
#                                            SEGMENTATION DES METASTASES                                               #
########################################################################################################################
########################################################################################################################


## On choisit le num de la souris
# n_souris = 8
# n_souris = 28
# n_souris = 56

for n_souris in [8, 28, 56]:

    ## On charge les paths associés
    path_souris = os.path.join(PATH_DATA, "Souris_Test/Souris/souris_"+str(n_souris)+".tif")
    path_msk = os.path.join(PATH_DATA, "Souris_Test/Masque_Metas/Meta_"+str(n_souris)+"/")


    ## On reconstruit la souris label
    list_msk = utils.sorted_aphanumeric(os.listdir(path_msk))
    seg_true = np.zeros(((128,128,128)))
    for i in np.arange(len(list_msk)):
        seg_true[i] = io.imread(path_msk+list_msk[i], plugin="tifffile")


    ## On récupère le path du modèle
    path_model_seg_meta = [os.path.join(PATH_GIT, "Metastases/model/unet_test.h5"),
                           os.path.join(PATH_GIT, "Metastases/model/unet_creative.h5"),
                           os.path.join(PATH_GIT, "Metastases/model/weight_map149.h5"),
                           os.path.join(PATH_GIT, "Metastases/model/weight_map11050.h5"),
                           os.path.join(PATH_GIT, "Metastases/model/weight_map_creative149.h5"),
                           os.path.join(PATH_GIT, "Metastases/model/weight_map_creative11050.h5")]

    path_result = ["a", "a", "b", "b", "b", "b"]
    name_folder = "souris_56"

    label = ['U-Net', 'U-Net + Creat9', 'U-Net + WM149', 'U-Net + WM11050', 'U-Net + Creat9 + WM149',
             'U-Net + Creat9 + WM11050']

    ResultIoU = np.zeros((len(path_result), 128))
    for i in np.arange(len(path_result)):

        ## On charge le modèle
        if path_result[i] == "a":
            seg_pred = model.seg_meta_original(path_souris, path_model_seg_meta[i], path_result[i], name_folder,
                                               mask=True, visu_seg=False)
        elif path_result[i] == 'b':
            seg_pred = model.seg_meta_original(path_souris, path_model_seg_meta[i], path_result[i], name_folder,
                                               mask=True, visu_seg=False, wei=True)

        ## On calcule l'IoU pour chaque image
        IoU = [utils.stats_pixelbased(seg_true[j], seg_pred[j]).get('IoU') for j in np.arange(128)]

        ## On les stock dans la matrice de Résultats
        ResultIoU[i, :] = np.array(IoU)
        print(i)

    df_iou = pd.DataFrame(ResultIoU.T, columns=label)
    df_iou.to_csv(os.path.join(PATH_GIT, "Metastases/stats/Souris_"+str(n_souris)+"/Result_IoU.csv"),
                  index=None, header=True)



########################################################################################################################
########################################################################################################################