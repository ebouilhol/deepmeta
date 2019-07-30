import model
import os
import numpy as np


# Souris Test :
souris_8 = "/home/achauviere/Bureau/DATA/Souris_Test/Souris/souris_8.tif"
# souris_22 = "/home/achauviere/Bureau/DATA/Souris_Test/souris_22.tif"
souris_28 = "/home/achauviere/Bureau/DATA/Souris_Test/Souris/souris_28.tif"
souris_56 = "/home/achauviere/Bureau/DATA/Souris_Test/Souris/souris_56.tif"





########################################################################################################################
########################################################################################################################
#                                              SEGMENTATION DES POUMONS                                                #
########################################################################################################################
########################################################################################################################


######### Application sur Souris - Modèle 2D Détection + Segmentation  #########

path_model_detect = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_detect.h5"
#path_model_seg = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_seg.h5"
#path_model_seg = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_seg_poum_creat.h5"
#path_model_seg = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/weight_map149.h5"
#path_model_seg = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/weight_map124.h5"
path_model_seg = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/weight_map_creative149.h5"

#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/results/Detect_Seg/U-Net/"
#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/results/Detect_Seg/Creative/U-Net/"
#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/results/Detect_Seg/Weight_Map/wm_149/"
#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/results/Detect_Seg/Weight_Map/wm_124/"
path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/results/Detect_Seg/Weight_Map/wm_crea149/"


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



######### Application sur Souris - Modèle Multi_Axes  #########

path_model_axial = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_axial.h5"
path_model_sagital = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_sagital.h5"
path_model_corronal = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_corronal.h5"

list_vote = ["Vote_1/", "Vote_2/", "Vote_3/"]
vote = [1, 2 ,3]
k = 0

for i in list_vote:

    path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/results/Multi_Axe/"+i

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



######### Application U-NET COUPE #########

path_model_detect = "/home/achauviere/Bureau/Detect_Seg_coupe/model_detect.h5"
path_model_seg = "/home/achauviere/Bureau/Detect_Seg_coupe/model_seg.h5"
path_result = "/home/achauviere/Bureau/Detect_Seg_coupe/Result/"

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




######### Application Manu #########

path_model_detect = "/home/achauviere/Bureau/Detect_Seg_test_manu/model_detect.h5"
path_model_seg = "/home/achauviere/Bureau/Detect_Seg_test_manu/model_seg.h5"
path_result = "/home/achauviere/Bureau/Detect_Seg_test_manu/Result/"

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




######### Application sur Souris Contraste différents  #########

## Coeur en blanc ##
path_souris = "/home/achauviere/Bureau/Autre_contraste/Emeline/Contraste_Souris/"
path_result = "/home/achauviere/Bureau/Autre_contraste/Emeline/Contraste_Souris/"
name_folder = "Results"

if not os.path.exists(path_result + str(name_folder)):
    os.makedirs(path_result + str(name_folder))

model.methode_multi_axe(path_souris, path_model_axial, path_model_sagital, path_model_corronal,
                     path_result, name_folder, mask=None, full_souris=False)


## Autre sagitale ##




######### Segmentation des Poumons Sains  #########

list_poum = ["2Pc_day15.tif", "2Pc_day22.tif", "2Pc_day29.tif", "2Pc_day36.tif", "2Pc_day43.tif", "2Pc_day50.tif",
             "2Pc_day57.tif", "2Pc_day64.tif", "2Pc_day71.tif", "2Pc_day78.tif", "2Pc_day85.tif", "2Pc_day92.tif",
             "2Pc_day99.tif", "2Pc_day106.tif", "2Pc_day113.tif", "2Pc_day120.tif", "2Pc_day127.tif","2Pc_day134.tif",
             "2Pc_day141.tif"]

list_folder = ["2Pc_day15", "2Pc_day22", "2Pc_day29", "2Pc_day36", "2Pc_day43", "2Pc_day50",
             "2Pc_day57", "2Pc_day64", "2Pc_day71", "2Pc_day78", "2Pc_day85", "2Pc_day92",
             "2Pc_day99", "2Pc_day106", "2Pc_day113", "2Pc_day120", "2Pc_day127","2Pc_day134",
             "2Pc_day141"]


# Méthode Detect_Seg
path_model_detect = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_detect.h5"
path_model_seg = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_seg.h5"
path_result = "/home/achauviere/Bureau/Poumon_sain_seg/Detect_Seg/"

path = "/home/achauviere/Bureau/Data/Data/iL34_1c/"

for i in np.arange(len(list_poum)):

    path_souris = path + list_poum[i]
    name_folder = list_folder[i]

    model.methode_detect_seg(path_souris, path_model_detect, path_model_seg, path_result,
                        name_folder, mask=None)


# Méthode Multi-Axes
path_model_axial = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model//model_axial.h5"
path_model_sagital = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_sagital.h5"
path_model_corronal = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_corronal.h5"
path_result = "/home/achauviere/Bureau/Poumon_sain_seg/Multi_Axes/"

for i in np.arange(len(list_poum)):

    path_souris = path + list_poum[i]
    name_folder = list_folder[i]

    model.methode_multi_axe(path_souris, path_model_axial, path_model_sagital, path_model_corronal,
                            path_result, name_folder, mask=None, vote_model=2)




######### Application U-NET ++  #########

path_model_detect = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_detect.h5"
path_model_seg = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_unet_plusplus.h5"
path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/results/Detect_Seg/U-Net_PlusPlus/"

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



######### Application Small U-NET  #########

path_model_detect = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_detect.h5"
#path_model_seg = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/small_unet.h5"
#path_model_seg = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/unetCoupe2Max.h5"
path_model_seg = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/unet_C2_Aug.h5"

#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/results/Detect_Seg/Unet_Coupe/"
#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/results/Detect_Seg/Small_U-Net/"
path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/results/Detect_Seg/Unet_C2_Aug/"


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



######### Cnn-Lstm  #########






########################################################################################################################
########################################################################################################################
#                                              SEGMENTATION DES METASTASES                                             #
########################################################################################################################
########################################################################################################################


######### Application UNET - Image original  #########

souris_8 = "/home/achauviere/Bureau/DATA/Souris_Test/Souris/souris_8.tif"
souris_28 = "/home/achauviere/Bureau/DATA/Souris_Test/Souris/souris_28.tif"
souris_56 = "/home/achauviere/Bureau/DATA/Souris_Test/Souris/souris_56.tif"

#path_model_seg_meta = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/unet_test.h5"
#path_model_seg_meta = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/unet_creative.h5"
#path_model_seg_meta = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/weight_map149.h5"
#path_model_seg_meta = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/weight_map11050.h5"
#path_model_seg_meta = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/weight_map_creative149.h5"
#path_model_seg_meta = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/weight_map_creative11050.h5"
#path_model_seg_meta = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/unet_final.h5"
#path_model_seg_meta = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/unet_final_coupe.h5"
#path_model_seg_meta = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/unet_final_w24.h5"
path_model_seg_meta = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/model/unet_final_w2040.h5"


#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/results/Test_Unet/Image_Normale/"
#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/results/Test_Unet_Creat/Image_Normale/"
#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/results/Weight_Map/wm_149/"
#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/results/Weight_Map/wm_11050/"
#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/results/Weight_Map/wm_crea149/"
#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/results/Unet_final/"
#path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/results/Unet_final_coupe/"
path_result = "/home/achauviere/PycharmProjects/Antoine_Git/Metastases/results/Unet_final_w2040/"


name_folder = "souris_8"
model.seg_meta_original(souris_8, path_model_seg_meta, path_result, name_folder, wei=True)

name_folder = "souris_28"
model.seg_meta_original(souris_28, path_model_seg_meta, path_result, name_folder, wei=True)

name_folder = "souris_56"
model.seg_meta_original(souris_56, path_model_seg_meta, path_result, name_folder, wei=True)




######### Application UNET - Image Poumon Segmenté  #########

souris_8 = "/home/achauviere/Bureau/DATA/Souris_Test/souris_8.tif"
souris_28 = "/home/achauviere/Bureau/DATA/Souris_Test/souris_28.tif"
souris_56 = "/home/achauviere/Bureau/DATA/Souris_Test/souris_56.tif"

path_model_seg_poum = "/home/achauviere/PycharmProjects/Antoine_Git/Poumons/model/model_seg.h5"
path_model_seg_meta = "/home/achauviere/Bureau/Annotation_Meta/Result/Metastases/modele/unet_test2.h5"
path_result = "/home/achauviere/Bureau/Annotation_Meta/Result/Test_Unet/Image_Poum_Seg/"

name_folder = "souris_8"
model.seg_meta_poum_seg(souris_8, path_model_seg_poum, path_model_seg_meta, path_result, name_folder)

name_folder = "souris_28"
model.seg_meta_poum_seg(souris_28, path_model_seg_poum, path_model_seg_meta, path_result, name_folder)

name_folder = "souris_56"
model.seg_meta_poum_seg(souris_56, path_model_seg_poum, path_model_seg_meta, path_result, name_folder)