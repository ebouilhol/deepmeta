import model
import pandas as pd
import utils
import os
import numpy as np
from skimage import io

path_unet = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_seg.h5"
path_unetCoupe2Max = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/unetCoupe2Max.h5"
path_unet_C2_Aug = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/unet_C2_Aug.h5"
path_unet_C2_Aug_Crea9 = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/unet_C2_Aug_Crea9.h5"
path_unet_C2_Aug_wm149 = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/unet_C2_Aug_wm149.h5"


def qualite_model(n_souris, path_model_seg, wei=None):

    path_souris = "/home/achauviere/Bureau/DATA/Souris_Test/Souris/souris_"+str(n_souris)+".tif"
    path_souris_annoter = "/home/achauviere/Bureau/DATA/Souris_Test/Masque_Poumons/masque_"+str(n_souris)+"/"
    path_result = "..."
    name_folder = "..."
    tab = pd.read_csv("/home/achauviere/Bureau/DATA/Tableau_General.csv").values
    path_model_detect = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_detect.h5"

    # Detection label
    tab2 = tab[np.where(tab[:,1]==n_souris)]
    detect_annot = tab2[:,2][np.where(tab2[:,3]==1)]
    n = len(detect_annot)

    # Segmentation label
    list_msk = utils.sorted_aphanumeric(os.listdir(path_souris_annoter))
    y = [io.imread(path_souris_annoter + list_msk[i], plugin="tifffile") for i in np.arange(len(list_msk))]
    seg_true = np.array(y, dtype='bool')


    # Segmentation prédite
    detect, seg = model.methode_detect_seg(path_souris, path_model_detect, path_model_seg, path_result,
                                               name_folder, mask=True, visu_seg=None, wei=wei)
    seg_pred = seg[detect_annot]

    # Calcul IoU
    IoU = [utils.stats_pixelbased(seg_true[j], seg_pred[j]).get('IoU') for j in range(n)]

    return IoU


def csv_qualite(list_result, list_label, n_souris, name_tab):

    Result = np.zeros((len(list_result[0]), len(list_result)))

    for i in np.arange(len(list_result)):
        Result[:, i] = list_result[i]

    df = pd.DataFrame(Result, columns=list_label)
    df.to_csv("/home/achauviere/PycharmProjects/Antoine_Git/Poumons/stats"
              "/Souris_" + str(n_souris) + "/"+name_tab+".csv", index=None, header=True)



# On récupère les IoU
IoU_unet = qualite_model(56, path_unet)
IoU_unetCoupe2Max = qualite_model(56, path_unetCoupe2Max)
IoU_unet_C2_Aug = qualite_model(56, path_unet_C2_Aug)
IoU_unet_C2_Aug_Crea9 = qualite_model(56, path_unet_C2_Aug_Crea9)
IoU_unet_C2_Aug_wm149 = qualite_model(56, path_unet_C2_Aug_wm149, wei=True)

# On sauvegarde le csv de comparaison
list_result = [IoU_unet, IoU_unetCoupe2Max, IoU_unet_C2_Aug, IoU_unet_C2_Aug_Crea9, IoU_unet_C2_Aug_wm149]
list_label = ["Unet", "UnetCoupe2Max", "Unet_C2_Aug", "Unet_C2_Aug_Crea9", "Unet_C2_Aug_wm149"]
csv_qualite(list_result, list_label, 56, "Unet_vs_UnetMax2_and_Aug")