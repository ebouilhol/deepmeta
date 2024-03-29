import model
import pandas as pd
import utils
import os
import numpy as np
from skimage import io
import data
from random import sample
from keras_preprocessing import image
from skimage import exposure
from keras.callbacks import EarlyStopping
import sys

# Choix de path
console = False
if console:
    ROOT_DIR = os.path.abspath("/home/achauviere/Bureau/Projet_Detection_Metastase_Souris/deepmeta-master/")
    sys.path.append(ROOT_DIR)
    PATH_GIT = os.path.join(ROOT_DIR, "../deepmeta-master/")
    PATH_DATA = os.path.join(ROOT_DIR, "../DATA/")
    PATH_Data = os.path.join(ROOT_DIR, "../Data/")
    PATH_Synth = os.path.join(ROOT_DIR, "../Data_Synthetique/")
else:
    PATH_GIT = "../deepmeta-master/"
    PATH_DATA = "../DATA/"
    PATH_Data = "../Data/"
    PATH_Synth = "../Data_Synthetique/"


########################################################################################################################
                            ######################## POUMONS ########################
########################################################################################################################

def qualite_model(num, path_model_seg, time, wei=None):
    """
    :param num: numéro de la souris auquel on veut appliquer le modèle (8, 28 ou 56 pour nos souris test)
    :param path_model_seg: le path du modèle de segmentation que l'on veut appliquer
    :param time: si time == 0 : réseau classique sans lstm / time == 'la valeur du time' pour un modèle LstmCnn
    :param wei: utilisation ou non d'un réseau avec pixels pondérés
    :return: liste des valeurs des IoU pour l'ensemble des slices de la souris test
    """
    path_souris = os.path.join(PATH_DATA, "Souris_Test/Souris/souris_"+str(num)+".tif")
    path_souris_annoter = os.path.join(PATH_DATA, "Souris_Test/Masque_Poumons/masque_"+str(num)+"/")
    tab = pd.read_csv(os.path.join(PATH_DATA, "Poumons/Tableau_General.csv")).values
    path_result = "..."  # pas besoin de save, on veut faire des stats
    name_folder = "..."  # idem
    path_model_detect = os.path.join(PATH_GIT, "Poumons/model/model_detect.h5")

    # Detection label
    tab2 = tab[np.where(tab[:, 1] == num)]
    detect_annot = tab2[:, 2][np.where(tab2[:, 3] == 1)]
    n = len(detect_annot)  # -1 pour Souris 56 => surement erreur dans annotation...

    # Segmentation label
    list_msk = utils.sorted_aphanumeric(os.listdir(path_souris_annoter))
    y = [io.imread(path_souris_annoter + list_msk[i], plugin="tifffile") for i in np.arange(len(list_msk))]
    seg_true = np.array(y, dtype='bool')

    # Segmentation prédite
    if time == 0:
        detect, seg = model.methode_detect_seg(path_souris, path_model_detect, path_model_seg, path_result,
                                               name_folder, mask=True, visu_seg=None, wei=wei)
    else:
        detect, seg = model.seg_poum_lstm(path_souris, path_model_detect, path_model_seg, time)
    seg_pred = seg[detect_annot]

    # Calcul IoU
    IoU = [utils.stats_pixelbased(seg_true[j], seg_pred[j]).get('IoU') for j in range(n)]

    return IoU


def csv_qualite(list_result, list_label, num, name_tab):
    """
    :param list_result: liste contenant les listes résultats IoU de différents modèles
    :param list_label: liste contenant le nom des méthodes utilisées
    :param num: le numéro de la souris test
    :param name_tab: titre du tableau csv
    :return: Tableau csv comportant les résultats IoU de différentes méthodes sur une souris test
    """
    Result = np.zeros((len(list_result[0]), len(list_result)))

    for i in np.arange(len(list_result)):
        Result[:, i] = list_result[i]

    df = pd.DataFrame(Result, columns=list_label)
    df.to_csv(os.path.join(PATH_GIT, "Poumons/stats") + "/Souris_"
              + str(num) + "/" + name_tab + ".csv", index=None, header=True)


##########################################
 ### Application sur quelques modèles ###
##########################################

# Path
path_unet = os.path.join(PATH_GIT, "Poumons/model/model_seg.h5")
path_unetCoupe2Max = os.path.join(PATH_GIT, "Poumons/model/unetCoupe2Max.h5")
path_unet_C2_Aug = os.path.join(PATH_GIT, "Poumons/model/unet_C2_Aug.h5")
path_unet_C2_Aug_Crea9 = os.path.join(PATH_GIT, "Poumons/model/unet_C2_Aug_Crea9.h5")
path_unet_C2_Aug_wm149 = os.path.join(PATH_GIT, "Poumons/model/unet_C2_Aug_wm149.h5")

# On récupère les IoU
IoU_unet = qualite_model(56, path_unet, 0)
IoU_unetCoupe2Max = qualite_model(56, path_unetCoupe2Max, 0)
IoU_unet_C2_Aug = qualite_model(56, path_unet_C2_Aug, 0)
IoU_unet_C2_Aug_Crea9 = qualite_model(56, path_unet_C2_Aug_Crea9, 0)
IoU_unet_C2_Aug_wm149 = qualite_model(56, path_unet_C2_Aug_wm149, 0, wei=True)

# On sauvegarde le csv de comparaison
list_result = [IoU_unet, IoU_unetCoupe2Max, IoU_unet_C2_Aug, IoU_unet_C2_Aug_Crea9, IoU_unet_C2_Aug_wm149]
list_label = ["Unet", "UnetCoupe2Max", "Unet_C2_Aug", "Unet_C2_Aug_Crea9", "Unet_C2_Aug_wm149"]
csv_qualite(list_result, list_label, 56, "Unet_vs_UnetMax2_and_Aug")


##########################################
 ### Application sur des modèles LSTM ###
##########################################

for n_souris in [8, 28]:  # n_souris = 56

    path_souris = os.path.join(PATH_DATA, "Souris_Test/Souris/souris_" + str(n_souris) + ".tif")
    list_result = []
    list_label = []
    time = [3, 6, 9, 12, 16, 32, 64]

    for t in time:
        path_model_seg = os.path.join(PATH_GIT, "Poumons/model/lstm/bclstm_" + str(t) + "_tl.h5")
        IoU = qualite_model(n_souris, path_model_seg, t)
        list_result.append(IoU)
        list_label.append("time_"+str(t))
    csv_qualite(list_result, list_label, n_souris, "Lstm_tl")




#######################################################################################################################
                            ####################### METASTASES ########################
#######################################################################################################################


def qualite_meta(n_souris, path_model_seg, wei=None):

    path_souris = os.path.join(PATH_DATA, "Souris_Test/Souris/souris_" + str(n_souris) + ".tif")
    path_msk = os.path.join(PATH_DATA, "Souris_Test/Masque_Metas/Meta_" + str(n_souris) + "/")
    path_result = "..."
    name_folder = "..."

    ## On reconstruit la souris label
    list_msk = utils.sorted_aphanumeric(os.listdir(path_msk))
    seg_true = np.zeros((128, 128, 128))
    for i in np.arange(len(list_msk)):
        seg_true[i] = io.imread(path_msk + list_msk[i], plugin="tifffile")
    seg_true = np.array(seg_true, dtype='bool')

    ## On prédit
    seg_pred = model.seg_meta_original(path_souris, path_model_seg, path_result, name_folder, mask=True,
                                       visu_seg=False, wei=wei)

    ## On calcule l'IoU pour chaque image
    IoU = [utils.stats_pixelbased(seg_true[j], seg_pred[j]).get('IoU') for j in np.arange(128)]
    return IoU


def csv_meta(list_result, list_label, n_souris, name_tab):

    Result = np.zeros((len(list_result[0]), len(list_result)))

    for i in np.arange(len(list_result)):
        Result[:, i] = list_result[i]

    df = pd.DataFrame(Result, columns=list_label)
    df.to_csv(os.path.join(PATH_GIT, "Metastases/stats") +
              "/Souris_" + str(n_souris) + "/"+name_tab+".csv", index=None, header=True)


# Path model
# path_unet = os.path.join(PATH_GIT, "Metastases/model/unet_test.h5")
# path_crea = os.path.join(PATH_GIT, "Metastases/model/unet_creative.h5")
path_unet_final = os.path.join(PATH_GIT, "Metastases/model/unet_final.h5")
# path_unet_final_coupe = os.path.join(PATH_GIT, "Metastases/model/unet_final_coupe.h5")
# path_unet_final_pp = os.path.join(PATH_GIT, "Metastases/model/unet_final_pluspus.h5")
path_unet_final_w2040 = os.path.join(PATH_GIT, "Metastases/model/unet_final_w2040.h5")


for num in [8, 28, 56]:

    # On récupère les IoU
    IoU_unet_final = qualite_meta(num, path_unet_final)
    IoU_unet_final_w2040 = qualite_meta(num, path_unet_final_w2040, wei=True)
    # On sauvegarde le csv de comparaison

    list_result = [IoU_unet_final, IoU_unet_final_w2040]
    list_label = ["Unet_final", "Unet_final_w2040"]
    csv_meta(list_result, list_label, num, "Unet_final_w2040")


########################################################################################################################


## Boucle Weight pour comparaison :

path_img = os.path.join(PATH_DATA, "Metastases/Image/")
path_lab = os.path.join(PATH_DATA, "Metastases/Label/")
tab = pd.read_csv(os.path.join(PATH_DATA, "Metastases/Tableau.csv")).values

path_il34c = os.path.join(PATH_Data, "iL34_1c/")
path_lacz = os.path.join(PATH_Data, "LacZ/")
path_res = os.path.join(PATH_GIT, "Metastases/results")

# paramètre de poids
a = [2, 2, 3, 3, 10, 10, 20, 20, 50] # contour
b = [2, 4, 3, 9, 10, 20, 20, 40, 50] # intérieur

for i in range(len(a)):

    Data, Label, ind = data.create_data_meta(path_img, path_lab, tab)
    N = np.arange(len(ind)) ; N_sample = sample(list(N), len(N))
    Data = Data.reshape(-1, 128, 128, 1)[N_sample]
    Label = Label.reshape(-1,128, 128, 1)[N_sample]
    seed = 10
    BATCH_SIZE = int(Data.shape[0]/3)
    image_datagen = image.ImageDataGenerator(horizontal_flip=True, rotation_range=20, fill_mode='reflect',
                                             shear_range=0.2, width_shift_range=0.1, height_shift_range=0.1)

    mask_datagen = image.ImageDataGenerator(horizontal_flip=True, rotation_range=20, fill_mode='reflect',
                                            shear_range=0.2, width_shift_range=0.1, height_shift_range=0.1)
    image_datagen.fit(Data, augment=True, seed=seed)
    mask_datagen.fit(Label, augment=True, seed=seed)
    x = image_datagen.flow(Data, batch_size=BATCH_SIZE, shuffle=True, seed=seed)
    y = mask_datagen.flow(Label, batch_size=BATCH_SIZE, shuffle=True, seed=seed)
    verif_xtrain = x.next().reshape(BATCH_SIZE, 128, 128)
    verif_ytrain = y.next().reshape(BATCH_SIZE, 128, 128)
    Gen = x.next()
    y_gen = y.next()
    Gen = Gen.reshape(BATCH_SIZE, 128, 128)
    y_gen = y_gen.reshape(BATCH_SIZE, 128, 128)
    Data = Data.reshape(Data.shape[0], 128, 128)
    Label = Label.reshape(Label.shape[0], 128, 128)
    X_new = utils.concat_data(Data, Gen)
    y_new = utils.concat_data(Label, y_gen)
    r = np.arange(X_new.shape[0])
    r_sample = sample(list(r), len(r))
    X_new = X_new[r_sample]
    y_new = y_new[r_sample]
    path_new = os.path.join(PATH_Synth, "Nouvelles_Images/")
    newData, newPoum, newMeta = data.recup_new_data(path_new)
    Data = utils.concat_data(X_new, newData)  # si erreur avec conc_data => utiliser conc3D qui est commenté dans utils
    MaskMeta = utils.concat_data(y_new, newMeta)
    a2 = utils.give_img(path_lacz, 'm2PL_day6.tif', 34, 107)
    b2 = utils.give_img(path_lacz, 'NoP_day19.tif', 15, 101)
    c = utils.give_img(path_il34c, '2PLPRc_day15.tif', 21, 96)
    d = utils.give_img(path_il34c, '2PLPRc_day22.tif', 24, 100)
    e = utils.give_img(path_il34c, '2PLPRc_day29.tif', 27, 100)
    f = utils.give_img(path_il34c, '2PLPRc_day120.tif', 27, 99)
    poumon_sain = utils.concat_data(utils.concat_data(utils.concat_data(utils.concat_data(utils.concat_data(a2, b2), c),
                                                                        d), e), f)
    poumon_sain = np.array(poumon_sain, dtype="uint8")
    x_sain = [exposure.equalize_adapthist(poumon_sain[i], clip_limit=0.03) for i in range(len(poumon_sain))]
    x_sain = np.array(x_sain)
    y_sain = np.zeros((x_sain.shape[0], 128, 128))
    y_sain = np.array(y_sain, dtype=np.bool)
    Data = utils.concat_data(Data, x_sain)
    Label = utils.concat_data(MaskMeta, y_sain)
    Weight = utils.weight_map(Label, a[i], b[i])
    N = np.arange(Data.shape[0])
    N_sample = sample(list(N), len(N))
    Data = Data.reshape(-1, 128, 128, 1)[N_sample]
    Label = Label.reshape(-1, 128, 128, 1)[N_sample]
    Weight = Weight.reshape(-1, 128, 128, 1)[N_sample]
    y = np.zeros((Data.shape[0], 128, 128, 2))
    y[:, :, :, 0] = Label[:, :, :, 0]
    y[:, :, :, 1] = Weight[:, :, :, 0]
    input_shape = (128, 128, 1)
    model_seg = model.model_unet_2D(input_shape, wei=True)
    earlystopper = EarlyStopping(patience=5, verbose=1)
    model_seg.fit(Data, y, validation_split=0.2, batch_size=8, epochs=70, callbacks=[earlystopper])
    model_seg.save(path_res + 'unet_final_w'+str(a[i])+str(b[i])+'.h5')


for num in [8, 28, 56]:
    # On récupère les IoU
    IoU_unet_final_w22 = qualite_meta(num, path_res + '/unet_final_w'+str(2)+str(2)+'.h5', wei=True)
    IoU_unet_final_w24 = qualite_meta(num, path_res + '/unet_final_w' + str(2) + str(4) + '.h5', wei=True)
    IoU_unet_final_w33 = qualite_meta(num, path_res + '/unet_final_w' + str(3) + str(3) + '.h5', wei=True)
    IoU_unet_final_w39 = qualite_meta(num, path_res + '/unet_final_w' + str(3) + str(9) + '.h5', wei=True)
    IoU_unet_final_w1010 = qualite_meta(num, path_res + '/unet_final_w' + str(10) + str(10) + '.h5', wei=True)
    IoU_unet_final_w1020 = qualite_meta(num, path_res + '/unet_final_w' + str(10) + str(20) + '.h5', wei=True)
    IoU_unet_final_w2020 = qualite_meta(num, path_res + '/unet_final_w' + str(20) + str(20) + '.h5', wei=True)
    IoU_unet_final_w2040 = qualite_meta(num, path_res + '/unet_final_w' + str(20) + str(40) + '.h5', wei=True)
    IoU_unet_final_w5050 = qualite_meta(num, path_res + '/unet_final_w' + str(50) + str(50) + '.h5', wei=True)

    # On sauvegarde le csv de comparaison
    list_result = [IoU_unet_final_w22, IoU_unet_final_w24, IoU_unet_final_w33, IoU_unet_final_w39, IoU_unet_final_w1010,
                   IoU_unet_final_w1020, IoU_unet_final_w2020, IoU_unet_final_w2040, IoU_unet_final_w5050]
    list_label = ["Unet_final_w22,", "Unet_final_w24", "Unet_final_w33", "Unet_final_w39", "Unet_final_w1010",
                  "Unet_final_w1020", "Unet_final_w2020", "Unet_final_w2040", "Unet_final_w5050"]
    csv_meta(list_result, list_label, num, "Unet_Comparaison_Weight")