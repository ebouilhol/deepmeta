import pandas as pd
import os
import utils
import numpy as np
import re
import cv2
from skimage import io


####################################################################################################
    ######################### Tableau Pour Segmenter les Metastases #########################
####################################################################################################


#Endroit ou créer le tableau
path_meta = "/home/achauviere/Bureau/Annotation_Meta/"
#Tableau General sans métastases
tab = pd.read_csv("/home/achauviere/Bureau/DATA/Tableau_General.csv").values

list_meta = utils.sorted_aphanumeric(os.listdir(path_meta))

vect_annot_meta = np.zeros(len(tab))
vect_nbr_meta = np.zeros(len(tab))
vect_num_meta = np.zeros(len(tab))

# Les souris qui ont été annoté pour les métastases
for k in np.arange(len(list_meta)):
    ind = np.where(tab[:,1]==int(re.findall('\d+', list_meta[k])[0]))
    vect_annot_meta[ind] = 1
vect_annot_meta = np.array(vect_annot_meta, dtype='int')


# Reindicage concernant l'indicage des poumons pour pouvoir automatiser la suite
k = 0
for i in np.arange(len(tab)):
    if tab[i,3] == 1 :
        k = k + 1
        vect_num_meta[i] = k
    else :
        vect_num_meta[i] = 0
    if tab[i,2] == 127:
        k = 0


#On fait le tableau avec un vecteur nul pour le nbr de méta
vect_num_meta = np.array(vect_num_meta, dtype='int')
vect_nbr_meta = np.array(vect_nbr_meta, dtype='int')

tab = np.c_[tab,vect_annot_meta]
tab = np.c_[tab,vect_num_meta]
tab = np.c_[tab,vect_nbr_meta]


#Construction du vecteur représentant le nbr de méta par images
for k in np.arange(len(list_meta)):

    x = utils.sorted_aphanumeric(os.listdir(path_meta + list_meta[k]))

    ind1 = np.where(tab[:, 1] == int(re.findall('\d+', list_meta[k])[0]))[0]
    ind2 = np.where(tab[:, 6] != 0)[0]
    ind3 = np.array(utils.intersection(ind1, ind2)) - 1

    ind_meta = np.zeros(len(x))
    for p in np.arange(len(x)):
        ind_meta[p] = int(re.findall('\d+', x[p])[0])
    ind_meta = np.array(ind_meta, dtype='int')

    u = np.zeros(len(ind3))
    for i in np.arange(len(ind3)):
        u[i] = len(np.where(ind_meta == i)[0])
    u = np.array(u, dtype='int')

    tab[ind3, 7] = u


# On construit le csv
df = pd.DataFrame(tab, columns = ['Image_N', 'Mouse x', 'Slice y', 'Detection', 'Poumon', 'Annot_Meta',"Num_Meta", "Nbr_Meta"])
df.to_csv(path_meta + "Tableau.csv", index = None, header=True)




####################################################################################################
         ######################### Construction du Jeu de Data  #########################
####################################################################################################

path_meta = "/home/achauviere/Bureau/Annotation_Meta/Metastases/"
path_souris = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Souris/"
path_img = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Image/"
path_mask = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Masques/"
path_lab = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Label/"
tab = pd.read_csv("/home/achauviere/Bureau/Annotation_Meta/Tableau.csv").values

list_meta = utils.sorted_aphanumeric(os.listdir(path_meta))

numMeta = []
for k in np.arange(len(list_meta)):
    numMeta.append(int(re.findall('\d+',list_meta[k])[0]))


## Creation du jeu de data Masques pour Unet
n = 0
for k in numMeta :
    list_msk = utils.sorted_aphanumeric(os.listdir(path_meta+list_meta[n]))
    os.mkdir(path_meta + 'Mask_'+str(k))
    ind1 = np.where(tab[:,1] == k)[0]
    ind2 = np.where(tab[:,3] != 0)[0]
    ind3 = utils.intersection(ind1, ind2)
    for i in np.arange(len(tab[ind3,6])):
        mask = np.zeros((128,128))
        if tab[ind3[i],7] != 0:
            for j in np.arange(tab[ind3[i],7]):
                mask = mask + io.imread(path_meta + list_meta[n] + '/' + "m"+str(i+1)+"_m"+str(j+1)+'.tif')
        cv2.imwrite(path_meta+'Mask_'+str(k)+"/m_"+str(i+1)+".tif",mask)
    n = n +1


## Creation du jeu d'images
list_souris = utils.sorted_aphanumeric(os.listdir(path_souris))
os.mkdir(path_img)
for k in np.arange(len(list_souris)):
    ind = np.where(tab[:,1] == int(re.findall('\d+',list_souris[k])[0]))[0] #extrait num
    data = io.imread(path_souris + list_souris[k], plugin='tifffile')
    for j in np.arange(data.shape[0]):
        slices = data[j]
        cv2.imwrite(path_img + "/img_" + str(ind[j])+".tif", slices)


## Creation du jeu de labels
list_mask = utils.sorted_aphanumeric(os.listdir(path_mask))
os.mkdir(path_lab)
for k in np.arange(len(list_mask)):
    a = np.where(tab[:, 1] == int(re.findall('\d+', list_mask[k])[0]))[0]
    b = np.where(tab[:, 4] == 1)[0]
    ind = utils.intersection(a, b)
    m_list = utils.sorted_aphanumeric(os.listdir(path_mask + '/' + list_mask[k]))
    for j in np.arange(len(m_list)):
        mask = io.imread(path_mask + list_mask[k] + '/' + m_list[j], plugin='tifffile')
        cv2.imwrite(path_lab + "/m_" + str(ind[j]) + ".tif", mask)




## Creation du jeu de data pour Mask-Rcnn

path_meta = "/home/achauviere/Bureau/Annotation_Meta/Metastases/Meta/"
path_meta_train = '/home/achauviere/PycharmProjects/Mask_RCNN/datasets/metastases/train/'
path_meta_test = '/home/achauviere/PycharmProjects/Mask_RCNN/datasets/metastases/test/'

for i in np.arange(len(tab)):
    if tab[i,5]==1:
        if tab[i,6] != 0 :
            im = io.imread(path_img + 'img_'+str(i)+'.tif', plugin='tifffile')
            os.mkdir(path_meta_train + "meta_"+str(i))
            os.mkdir(path_meta_train + "meta_"+str(i) + '/images')
            os.mkdir(path_meta_train + "meta_"+str(i) + '/masks')
            cv2.imwrite(path_meta_train + "meta_"+str(i) + '/images' + "/meta_" + str(i)+".tif", im)

n = 0
for k in numMeta :
    list_msk = utils.sorted_aphanumeric(os.listdir(path_meta+list_meta[n]))
    ind1 = np.where(tab[:,1] == k)[0]
    ind2 = np.where(tab[:,3] != 0)[0]
    ind3 = utils.intersection(ind1, ind2)
    for i in np.arange(len(tab[ind3,6])):
        if tab[ind3[i],7] != 0:
            for j in np.arange(tab[ind3[i],7]):
                mask = io.imread(path_meta + list_meta[n] + '/' + "m"+str(i+1)+"_m"+str(j+1)+'.tif')
                cv2.imwrite(path_meta_train + "meta_"+str(ind3[i]) + '/masks' + "/m"+str(ind3[i])+"_m"+str(j+1)+'.tif', mask)
        else:
            mask = np.zeros((128,128))
            cv2.imwrite(path_meta_train + "meta_"+str(ind3[i]) + '/masks' + "/m"+str(ind3[i])+'.tif', mask)
    n = n +1



############ Créer jeu de données pour test ############

n_souris = 8 ; n_img = 74 ; min = 21 ; max = 95 # Poumons petites métastases
# n_souris = 28  # Poumons sains
# n_souris = 56 ; n_img = 81 ; min = 28 ; max = 109 # Poumons grosses métastases

## On charge les paths associés
path_souris = "/home/achauviere/Bureau/DATA/Souris_Test/souris_"+str(n_souris)+".tif"
path_msk = "/home/achauviere/Bureau/DATA/Souris_Test/masque_meta_"+str(n_souris)+"/"

## On reconstruit la souris label
list_msk = utils.sorted_aphanumeric(os.listdir(path_msk))

data = np.zeros(((n_img,128,128)))

for i in np.arange(len(list_msk)):
    k = int(re.findall('\d+', list_msk[i])[0])
    data[k-1] = data[k-1] + io.imread(path_msk + list_msk[i])

seg_true = np.zeros(((128,128,128)))
seg_true[min:max] = data

for i in np.arange(128):
    im = seg_true[i]
    cv2.imwrite("/home/achauviere/Bureau/Meta8/m"+str(i)+".tif", im)