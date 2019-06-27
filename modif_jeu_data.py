import pandas as pd
import utils
import os
import numpy as np
import cv2
import re
from skimage import io


#### Path ####

path_souris = "/home/achauviere/Bureau/DATA/Souris/"
path_mask = "/home/achauviere/Bureau/DATA/Masques/"
path_img = "/home/achauviere/Bureau/DATA/Image/"
path_lab = "/home/achauviere/Bureau/DATA/Label/"
tab = pd.read_csv("/home/achauviere/Bureau/DATA/Tableau_General.csv").values



#### Images ####

list_souris = utils.sorted_aphanumeric(os.listdir(path_souris))
#print(list_souris)

os.mkdir("/home/achauviere/Bureau/DATA/Image/")
for k in np.arange(len(list_souris)):

    ind = np.where(tab[:,1] == int(re.findall('\d+',list_souris[k])[0]))[0] #extrait num
    data = io.imread(path_souris + list_souris[k], plugin='tifffile')

    for j in np.arange(data.shape[0]):
        slices = data[j]
        cv2.imwrite(path_img + "/img_" + str(ind[j])+".tif", slices)



#### Labels ####

list_mask = utils.sorted_aphanumeric(os.listdir(path_mask))
#print(list_mask)

os.mkdir("/home/achauviere/Bureau/DATA/Label/")
for k in np.arange(len(list_mask)):

    a = np.where(tab[:, 1] == int(re.findall('\d+', list_mask[k])[0]))[0]
    b = np.where(tab[:, 4] == 1)[0]
    ind = utils.intersection(a, b)
    m_list = utils.sorted_aphanumeric(os.listdir(path_mask + '/' + list_mask[k]))

    for j in np.arange(len(m_list)):
        mask = io.imread(path_mask + list_mask[k] + '/' + m_list[j], plugin='tifffile')
        cv2.imwrite(path_lab + "/m_" + str(ind[j]) + ".tif", mask)