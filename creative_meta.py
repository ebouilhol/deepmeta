import model
import utils
import numpy as np
import os
import re
import sys
import cv2
from random import randint, choice
from skimage import io


ROOT_DIR = os.path.abspath("/home/achauviere/Bureau/Projet_Detection_Metastase_Souris/") # dans mon cas.
sys.path.append(ROOT_DIR)

# PATH_poum_sain_seg = "./Poumon_sain_seg/"
PATH_poum_sain_seg = os.path.join(ROOT_DIR, "./Poumon_sain_seg/")

# PATH_meta = "./Annotation_Meta/"
PATH_meta = os.path.join(ROOT_DIR, "./Annotation_Meta/")

# PATH_Data = "./Data/"
PATH_Data = os.path.join(ROOT_DIR, "./Data/")

# PATH_GIT = "./Antoine_Git/"
PATH_GIT = os.path.join(ROOT_DIR, "./Antoine_Git/")

##########################################################################################
            ######### Creation de données métastasées pour une Souris  #########
##########################################################################################

# Souris :
#name_souris = "Souris_24" ; ident = "2Pc_day22"
#name_souris = "Souris_26" ; ident = "2Pc_day36"
#name_souris = "Souris_28" ; ident = "2Pc_day50"
#name_souris = "Souris_29" ; ident = "2Pc_day57"
#name_souris = "Souris_30" ; ident = "2Pc_day64"
#name_souris = "Souris_32" ; ident = "2Pc_day78"
#name_souris = "Souris_35" ; ident = "2Pc_day99"
#name_souris = "Souris_38" ; ident = "2Pc_day120"
name_souris = "Souris_39" ; ident = "2Pc_day127"

# Paramètres :
taille_min_poumon = 150
taille_min_meta = 0*255
taille_max_meta = 35*255
possibilite = [6,8,10,12,14,16]

# Path :
path_full_meta = os.path.join(PATH_poum_sain_seg, "Full_Meta/")
path_img = os.path.join(PATH_meta, "Metastases/Image/")
path_souris = os.path.join(PATH_Data, "iL34_1c/" + ident +".tif")
path_model_detect = os.path.join(PATH_GIT, "model/model_detect.h5")
path_model_seg = os.path.join(PATH_GIT, "model/model_seg.h5")

path_result = "None"  # On récupère juste valeur des masques dans un premier temps (visu_seg=False).
name_folder = "None"

path_result_souris = os.path.join(PATH_poum_sain_seg, "Nouvelles_Images/"+name_souris)

if not os.path.exists(path_result_souris):
    os.makedirs(path_result_souris)

os.mkdir(path_result_souris + "/Images/")
os.mkdir(path_result_souris + "/Masque_Poum/")
os.mkdir(path_result_souris + "/Masque_Meta/")

# On récupère la détection des Poumons et leur Masque
detect, seg = model.methode_detect_seg(path_souris, path_model_detect, path_model_seg, path_result, name_folder,
                   mask=True, visu_seg=False)

# On charge la Souris detect==1 et on modifie son contraste
souris = utils.contraste_and_reshape(io.imread(path_souris, plugin='tifffile')[np.where(detect==1)]).reshape(seg.shape[0],128,128)


for v in np.arange(souris.shape[0]):

    # On choisit l'image à modifier :
    img_new = souris[v]*1
    mask_new = seg[v]*1

    if seg[v].sum() > taille_min_poumon:

        # On récupère les coordonnées du masque de Poumon de l'image à modifié
        ligne_mask_poum, colonne_mask_poum = np.where(mask_new==1)

        # Nombre de métastase à insérer :
        N = choice(possibilite)

        #Initialisation du masque des méta
        mask_new_meta = np.zeros((128,128))

        for k in np.arange(N):
            
            compte1 = 0
            x = False
            while x==False :
                # On choisit une métastases aléatoire et on la charge :
                nbr_img_meta = len(os.listdir(path_full_meta))-1
                nbr_rand_meta = randint(0,nbr_img_meta)
                path_meta = os.listdir(path_full_meta)[nbr_rand_meta]
                meta = io.imread(path_full_meta + path_meta)
                compte1 +=1
                print("compte1", compte1)
                if taille_min_meta < meta.sum() < taille_max_meta:
                    x = True


            # On récupère les coordonnées de la méta sur son image original ainsi que le numéro N de cette image:
            ligne_mask_meta, colonne_mask_meta = np.where(meta==255)
            num_meta = int(re.findall('\d+',path_meta)[0])


            # On charge l'image d'où provient la méta + modif contraste :
            img = utils.contraste_and_reshape(io.imread(path_img + "img_" + str(num_meta) + ".tif")).reshape(128,128)


            # Test de possibilité de création : On regarde où la méta peut entrer dans le masque de Poum de l'image a modifié
            test_ligne = [t-ligne_mask_meta[0] for t in ligne_mask_meta]
            test_col = [t-colonne_mask_meta[0] for t in colonne_mask_meta]

            nbr_rand = randint(0,len(ligne_mask_poum))-1

            a = [t + ligne_mask_poum[nbr_rand] for t in test_ligne]
            b = [t + colonne_mask_poum[nbr_rand] for t in test_col]
            c = [seg[50][a[t],b[t]] for t in np.arange(len(a))]

            compte2 = 0 #eviter boucle infini si pas possible
            while sum(c)!=len(a):
                nbr_rand = randint(0,len(ligne_mask_poum)) -1
                a = [t + ligne_mask_poum[nbr_rand] for t in test_ligne]
                b = [t + colonne_mask_poum[nbr_rand] for t in test_col]
                c = [mask_new[a[t],b[t]] for t in np.arange(len(a))]
                compte2+=1
                if compte2==100:
                    print("C'est chaud à trouver ! (2)")
                    break

            if compte2!=100:
                # On ajoute la métastase sur l'image à modifié en prenant les couleurs de la méta original :
                img_new[a, b]  = img[ligne_mask_meta, colonne_mask_meta]
                mask_new[a, b] = 0

                #On update le masque méta
                mask_new_meta[a, b] = 1


        #On récupère le masque des poumons
        mask_new_poumon = seg[v]


    else :
        mask_new_poumon = seg[v] * 1
        mask_new_meta = np.zeros((128,128))

    #Modif en 0-255:
    img_new = utils.etale_hist(img_new)
    mask_new_poumon = utils.etale_hist(mask_new_poumon)
    mask_new_meta = utils.etale_hist(mask_new_meta)


    # On sauvegarde la nouvelle image et ses nouveau masque :
    cv2.imwrite(path_result_souris+"/Images/img_"+str(v)+".tif", img_new)
    cv2.imwrite(path_result_souris+"/Masque_Poum/m_"+str(v)+".tif", mask_new_poumon)
    cv2.imwrite(path_result_souris+"/Masque_Meta/m_"+str(v)+".tif", mask_new_meta)




