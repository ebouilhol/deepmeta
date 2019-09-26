import model
import os
import numpy as np
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
    PATH_Data_contr = os.path.join(ROOT_DIR, "../Data_contraste/")
else:
    PATH_GIT = "../deepmeta-master/"
    PATH_DATA = "../DATA/"
    PATH_Data = "../Data/"
    PATH_Synth = "../Data_Synthetique/"
    PATH_Data_contr = "../Data_contraste/"


# Souris Test :
souris_8 = os.path.join(PATH_DATA, "Souris_Test/Souris/souris_8.tif")
souris_28 = os.path.join(PATH_DATA, "Souris_Test/Souris/souris_28.tif")
souris_56 = os.path.join(PATH_DATA, "Souris_Test/Souris/souris_56.tif")



""" Lexique : 
Données synthétiques : données créées à la main en projetant des métastases dans les poumons sains. 
Données artificielles : données modifiées par l'application de rotation, déformation elastique etc...
     => voir package "albumentations" que j'ai découvert récemment et que je n'ai pas encore appliqué.
"""



########################################################################################################################
########################################################################################################################
#                                              SEGMENTATION DES POUMONS                                                #
########################################################################################################################
########################################################################################################################



#################################################################################
######### Application sur Souris - Modèle 2D Détection + Segmentation  #########
#################################################################################

""" Pour appliquer cette méthode, il faut un modèle de detection et un modèle de segmentation """

## Modèle de détection ##

path_model_detect = os.path.join(PATH_GIT, "Poumons/model/model_detect.h5")


## Modèle de segmentation ##

# U-Net
# path_model_seg = os.path.join(PATH_GIT, "Poumons/model/model_seg.h5")
# path_result = os.path.join(PATH_GIT, "Poumons/results/Detect_Seg/U-Net/")

# Small U-Net
# path_model_seg = os.path.join(PATH_GIT, "Poumons/model/small_unet.h5")
# path_result = os.path.join(PATH_GIT, "Poumons/results/Detect_Seg/Small_U-Net/")

# U-Net ++
# path_model_seg = os.path.join(PATH_GIT, "model/model_unet_plusplus.h5")
# path_result = os.path.join(PATH_GIT, "Poumons/results/Detect_Seg/U-Net_PlusPlus/")

# U-Net avec poids modifiés (masque-intérieur-contour)
# path_model_seg = os.path.join(PATH_GIT, "Poumons/model/weight_map124.h5")
# path_result = os.path.join(PATH_GIT, "Poumons/results/Detect_Seg/Weight_Map/wm_124/")
# path_model_seg = os.path.join(PATH_GIT, "Poumons/model/weight_map149.h5")
# path_result = os.path.join(PATH_GIT, "Poumons/results/Detect_Seg/Weight_Map/wm_149/")

# U-Net avec ajouté de données synthétiques
# path_model_seg = os.path.join(PATH_GIT, "Poumons/model/model_seg_poum_creat.h5")
# path_result = os.path.join(PATH_GIT, "Poumons/results/Detect_Seg/Creative/U-Net/")

# U-Net avec ajouté de données synthétiques et poids modifiées
# path_model_seg = os.path.join(PATH_GIT, "Poumons/model/weight_map_creative149.h5")
# path_result = os.path.join(PATH_GIT, "Poumons/results/Detect_Seg/Weight_Map/wm_crea149/")

# U-Net avec suppression des deux dernières couches de MaxPooling (U-Net Manu)
path_model_seg = os.path.join(PATH_GIT, "Poumons/model/unetCoupe2Max.h5")
path_result = os.path.join(PATH_GIT, "Poumons/results/Detect_Seg/Unet_Manu/")

# U-Net Manu avec ajout de données artificielles
# path_model_seg = os.path.join(PATH_GIT, "Poumons/model/unet_C2_Aug.h5")
# path_result = os.path.join(PATH_GIT, "Poumons/results/Detect_Seg/Unet_C2_Aug/")


# U-Net avec un bloc de convolution en moins
# path_model_seg = os.path.join(PATH_GIT, "Poumons/model/unetCoupe.h5") # à réentrainer?
# path_result = os.path.join(PATH_GIT, "Poumons/results/Detect_Seg/Unet_Coupe/")


## Choix des souris auquelles appliquées le modèle ##

path_souris = souris_8
name_folder = "Souris_8"
model.methode_detect_seg(path_souris, path_model_detect, path_model_seg, path_result,
                       name_folder, mask=None)

path_souris = souris_28
name_folder = "Souris_28"
model.methode_detect_seg(path_souris, path_model_detect, path_model_seg, path_result,
                       name_folder, mask=None)

path_souris = souris_56
name_folder = "Souris_56"
model.methode_detect_seg(path_souris, path_model_detect, path_model_seg, path_result,
                       name_folder, mask=None)



#################################################################################
################# Application sur Souris - Modèle Multi_Axes  #################
#################################################################################

""" Pour appliquer cette méthode, il faut 3 modèles de segmentation et définir un paramètre "vote" """

# Les modèles de segmentation suivant les différents plans (indépendants)
path_model_axial = os.path.join(PATH_GIT, "Poumons/model/model_axial.h5")
path_model_sagital = os.path.join(PATH_GIT, "Poumons/model/model_sagital.h5")
path_model_corronal = os.path.join(PATH_GIT, "Poumons/model/model_corronal.h5")

# Ce modèle nécessite le choix d'un paramètre 'vote' (=1 le mieux) afin d'émettre une prédiction finale.
# Application de la méthode avec les différents votes pour chaque souris :

list_vote = ["Vote_1/", "Vote_2/", "Vote_3/"]
vote = [1, 2, 3]
k = 0

for i in list_vote:

    path_result = os.path.join(PATH_GIT, "Poumons/results/Multi_Axe/"+i)

    path_souris = souris_8
    name_folder = "Souris_8"
    model.methode_multi_axe(path_souris, path_model_axial, path_model_sagital, path_model_corronal,
                            path_result, name_folder, mask=None, vote_model=vote[k])

    path_souris = souris_28
    name_folder = "Souris_28"
    model.methode_multi_axe(path_souris, path_model_axial, path_model_sagital, path_model_corronal,
                            path_result, name_folder, mask=None, vote_model=vote[k])

    path_souris = souris_56
    name_folder = "Souris_56"
    model.methode_multi_axe(path_souris, path_model_axial, path_model_sagital, path_model_corronal,
                            path_result, name_folder, mask=None, vote_model=vote[k])

    k = k+1


#################################################################################
################# Application sur Souris Contraste différents #################
#################################################################################

""" Application des méthodes sur des souris avec d'autres contraste.  """

## Coeur en blanc ##
path_souris = os.path.join(PATH_Data_contr, "Autre_contraste/Emeline/Contraste_Souris/")
path_result = os.path.join(PATH_Data_contr, "Autre_contraste/Emeline/Contraste_Souris/")
name_folder = "Results"

if not os.path.exists(path_result + str(name_folder)):
    os.makedirs(path_result + str(name_folder))

model.methode_multi_axe(path_souris, path_model_axial, path_model_sagital, path_model_corronal,
                     path_result, name_folder, mask=None, full_souris=False)

## Autre sagitale ##
# J'ai tester rapidement mais aucun résultat convainquant : à refaire.



#################################################################################
####################### Segmentation des Poumons Sains  #######################
#################################################################################

""" Segmentation de poumons sains non annotés afin de créer des données synthétiques """

# Les souris auxquelles j'ai appliqué le modèle
list_poum = ["2Pc_day15.tif", "2Pc_day22.tif", "2Pc_day29.tif", "2Pc_day36.tif", "2Pc_day43.tif", "2Pc_day50.tif",
             "2Pc_day57.tif", "2Pc_day64.tif", "2Pc_day71.tif", "2Pc_day78.tif", "2Pc_day85.tif", "2Pc_day92.tif",
             "2Pc_day99.tif", "2Pc_day106.tif", "2Pc_day113.tif", "2Pc_day120.tif", "2Pc_day127.tif","2Pc_day134.tif",
             "2Pc_day141.tif"]

list_folder = ["2Pc_day15", "2Pc_day22", "2Pc_day29", "2Pc_day36", "2Pc_day43", "2Pc_day50",
             "2Pc_day57", "2Pc_day64", "2Pc_day71", "2Pc_day78", "2Pc_day85", "2Pc_day92",
             "2Pc_day99", "2Pc_day106", "2Pc_day113", "2Pc_day120", "2Pc_day127","2Pc_day134",
             "2Pc_day141"]


# Utilisation de la méthode detection segmentation avec le U-Net original
path_model_detect = os.path.join(PATH_GIT, "Poumons/model/model_detect.h5")
path_model_seg = os.path.join(PATH_GIT, "Poumons/model/model_seg.h5")
path_result = os.path.join(PATH_Synth, "Detect_Seg/")

path = os.path.join(PATH_GIT, "iL34_1c/")

for i in np.arange(len(list_poum)):

    path_souris = path + list_poum[i]
    name_folder = list_folder[i]

    model.methode_detect_seg(path_souris, path_model_detect, path_model_seg, path_result,
                        name_folder, mask=None)


# Utilisation de la méthode Multi-Axes
path_model_axial = os.path.join(PATH_GIT, "Poumons/model//model_axial.h5")
path_model_sagital = os.path.join(PATH_GIT, "Poumons/model/model_sagital.h5")
path_model_corronal = os.path.join(PATH_GIT, "Poumons/model/model_corronal.h5")
path_result =  os.path.join(PATH_Synth, "Multi_Axes/")

for i in np.arange(len(list_poum)):

    path_souris = path + list_poum[i]
    name_folder = list_folder[i]

    model.methode_multi_axe(path_souris, path_model_axial, path_model_sagital, path_model_corronal,
                            path_result, name_folder, mask=None, vote_model=2)

"""
Evalutation manuelle des résultats afin de choisir des poumons sains parfaitement segmenté. 
J'ai choisi les poumons segmentés par la méthode detect-seg avec le U-Net.
J'ai considéré 9 souris : voir script creative_meta.
"""

######### Cnn-Lstm  #########

"""
Voir script Cnn-Lstm
"""







########################################################################################################################
########################################################################################################################
#                                              SEGMENTATION DES METASTASES                                             #
########################################################################################################################
########################################################################################################################

""" Voir Notebook pour segmentation des métastases"""

souris_8 = os.path.join(PATH_DATA, "Souris_Test/Souris/souris_8.tif")
souris_28 = os.path.join(PATH_DATA, "Souris_Test/Souris/souris_28.tif")
souris_56 = os.path.join(PATH_DATA, "Souris_Test/Souris/souris_56.tif")


######### Application UNET - Image original  #########

#path_model_seg_meta = os.path.join(PATH_GIT, "Metastases/model/unet_test.h5")
#path_model_seg_meta = os.path.join(PATH_GIT, "Metastases/model/unet_creative.h5")
#path_model_seg_meta = os.path.join(PATH_GIT, "Metastases/model/weight_map149.h5")
#path_model_seg_meta = os.path.join(PATH_GIT, "Metastases/model/weight_map11050.h5")
#path_model_seg_meta = os.path.join(PATH_GIT, "Metastases/model/weight_map_creative149.h5")
#path_model_seg_meta = os.path.join(PATH_GIT, "Metastases/model/weight_map_creative11050.h5")
#path_model_seg_meta = os.path.join(PATH_GIT, "Metastases/model/unet_final.h5")
#path_model_seg_meta = os.path.join(PATH_GIT, "Metastases/model/unet_final_coupe.h5")
#path_model_seg_meta = os.path.join(PATH_GIT, "Metastases/model/unet_final_w24.h5")
path_model_seg_meta = os.path.join(PATH_GIT, "Metastases/model/unet_final_w2040.h5")


#path_result = os.path.join(PATH_GIT, "Metastases/results/Test_Unet/Image_Normale/")
#path_result = os.path.join(PATH_GIT, "Metastases/results/Test_Unet_Creat/Image_Normale/")
#path_result = os.path.join(PATH_GIT, "Metastases/results/Weight_Map/wm_149/")
#path_result = os.path.join(PATH_GIT, "Metastases/results/Weight_Map/wm_11050/")
#path_result = os.path.join(PATH_GIT, "Metastases/results/Weight_Map/wm_crea149/")
#path_result = os.path.join(PATH_GIT, "Metastases/results/Unet_final/")
#path_result = os.path.join(PATH_GIT, "Metastases/results/Unet_final_coupe/")
path_result = os.path.join(PATH_GIT, "Metastases/results/Unet_final_w2040/")


name_folder = "souris_8"
model.seg_meta_original(souris_8, path_model_seg_meta, path_result, name_folder, wei=True)

name_folder = "souris_28"
model.seg_meta_original(souris_28, path_model_seg_meta, path_result, name_folder, wei=True)

name_folder = "souris_56"
model.seg_meta_original(souris_56, path_model_seg_meta, path_result, name_folder, wei=True)




######### Application UNET - Image Poumon Segmenté  #########


path_model_seg_poum = os.path.join(PATH_GIT, "Poumons/model/model_seg.h5")
path_model_seg_meta = os.path.join(PATH_GIT, "Metastases/modele/unet_test2.h5")
path_result =  os.path.join(PATH_GIT, "results/Test_Unet/Image_Poum_Seg/")

name_folder = "souris_8"
model.seg_meta_poum_seg(souris_8, path_model_seg_poum, path_model_seg_meta, path_result, name_folder)

name_folder = "souris_28"
model.seg_meta_poum_seg(souris_28, path_model_seg_poum, path_model_seg_meta, path_result, name_folder)

name_folder = "souris_56"
model.seg_meta_poum_seg(souris_56, path_model_seg_poum, path_model_seg_meta, path_result, name_folder)

