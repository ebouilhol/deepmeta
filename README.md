# DeepMeta
Ce projet à pour but de construire des modèles de segmentation automatique d'images IRM pour la détection des poumons et
métastases pulmonaires dans le cas du petit animal, la souris. 

## Données
Les données sont des images IRM 3D 128x128x128 voxels, acquises par le RMSB, et représentent l'état de la maladie d'une souris 
à un instant donné. 

Les images peuvent être visualiser à l'aide du logiciel Fiji et sous différents axes (axial, coronal ou sagittal). 
Elles ont été acquises suivant le plan axial mais nous les avons traitées dans le sens coronal => meilleur visualisation des poumons.

Il est possible de définir trois types d'image : 
* Souris saine : aucune métastase présente dans les poumons.
* Souris en début de maladie : plusieurs petites métastases présentes dans les poumons.
* Souris en fin de maladie : de grosses métastases matures présentes dans les poumons.

Nous disposons en tout de : 
* 87 images de souris = 11136 slices. 
* 27 souris annotées pour la segmentation des poumons => 2128 slices présentants des poumons. 
* 8 souris annotées pour la segmentation des métastases => 620 slices présentants des poumons et 387 d'entres elles présentent des métastases

## Projet
Le dossier du projet est sur /mnt/cbib/Projet_Detection_Metastase_Souris avec comme architecture : 

Projet_Detection_Metastase_Souris
    * Antoine_Git : Ensemble des script python. 
        ** okok
    * Data 
    * DATA
    * Data_contraste
    * Poumon_sain_seg
    * Annotation_Meta





Pour lancer les script, il faut se placer dans le dossier Projet_Detection_Metastase_Souris. 
