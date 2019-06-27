import keras
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.models import *
import utils
from skimage import measure, io
import matplotlib.pyplot as plt
import os


def model_detection():
    """
    :return: Compile un réseau de détection de slices présentant des poumons
    """
    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3) ,activation='linear' ,input_shape=(128 ,128 ,1) ,padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2) ,padding='same'))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear' ,padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2) ,padding='same'))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear' ,padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2) ,padding='same'))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(Dense(2, activation='softmax'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam()
                          ,metrics=['accuracy'])
    return fashion_model



def model_unet_2D(input_shape):
    """
    :param input_shape: dimension des images en entrées, typiquement (128,128,1)
    :return: Compile un réseau U-NET
    """

    inputs = Input(input_shape)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[utils.mean_iou])
    return model



def methode_detect_seg(path_souris, path_model_detect, path_model_seg, path_result,
                       name_folder, mask=None, full_souris=True, visu_seg=True, img=None):

    """
    :param path_souris: image.tif étant un ensemble de stack
    :param path_model_detect: modèle de detection de slice pour les poumons format h5
    :param path_model_seg: modèle de segmentation des poumons format h5
    :param path_result: chemin où sauvegarder le résultat (path/)
    :param name_folder: nom du dossier où stocker les résultats
    :param mask: si True alors retourne les prédictions de détection ainsi que la valeur des masques associés
    :param full_souris: True correspond à Souris 3D.tiff sinon ensemble de slices.tiff
    :param visu_seg: True correspond à la sauvegarde des contours sur les images
    :return: Ensemble d'image avec contour des poumons segmentéssi visu
    :return: Retourne les prédictions de détection ainsi que la valeur des masques associés si mask
    """

    if full_souris:
        souris = io.imread(path_souris, plugin='tifffile')

    else:
        slices_list = utils.sorted_aphanumeric(os.listdir(path_souris))
        s = np.zeros(((len(slices_list),128,128)))
        for i in np.arange(len(slices_list)):
            s[i] = io.imread(path_souris + slices_list[i])
        souris = np.array(s)

    data = utils.contraste_and_reshape(souris)


    model_detect = keras.models.load_model(path_model_detect)
    modele_seg = keras.models.load_model(path_model_seg, custom_objects={'mean_iou': utils.mean_iou})

    detect = model_detect.predict_classes(data)
    seg = (modele_seg.predict(data) > 0.5).astype(np.uint8).reshape(128, 128, 128)
    data = data.reshape(128, 128, 128)

    if visu_seg :

        if not os.path.exists(path_result + str(name_folder)):
            os.makedirs(path_result + str(name_folder))

        for k in np.arange(128):
            if detect[k] == 1:

                cell_contours = measure.find_contours(seg[k], 0.8)
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
                for n, contour in enumerate(cell_contours):
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
                plt.xlim((0, 128))
                plt.ylim((128, 0))
                plt.imshow(data[k], cmap='gray');
                plt.savefig(path_result + str(name_folder) + "/m_" + str(k) + ".png")
                plt.close(fig)

            else:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
                plt.imshow(data[k], cmap='gray');
                plt.savefig(path_result + str(name_folder) + "/m_" + str(k) + ".png")
                plt.close(fig)

    if mask:
        ind = np.where(detect == 1)
        return detect, seg[ind]





def methode_multi_axe(path_souris, path_model_axial, path_model_sagital, path_model_corronal,
                      path_result, name_folder, vote_model = 2, mask=None, full_souris=True, visu_seg=True):
    """

    :param path_souris: image.tif étant un ensemble de stack
    :param path_model_axial: modèle de segmentation axial des poumons, format h5
    :param path_model_sagital: modèle de segmentation sagital des poumons, format h5
    :param path_model_corronal:  modèle de segmentation corronal des poumons, format h5
    :param path_result: chemin où sauvegarder le résultat (path/)
    :param name_folder: nom du dossier de résultat
    :param vote_model: Si 2 : Vote majoritaire par défault, (1: dès qu'un modèle prédit / 3: tous les modèles d'accord)
    :param mask: si True alors retourne la valeur des masques
    :param full_souris : True correspond à Souris 3D.tiff sinon ensemble de slices.tiff
    :param visu_seg: True correspond à la sauvegarde des contours sur les images
    :return: Ensemble d'image avec contour des poumons segmentés si visu
    :return: Masque si mask
    """

    if full_souris:
        souris = io.imread(path_souris, plugin='tifffile')

    else:
        s = []
        slices_list = utils.sorted_aphanumeric(os.listdir(path_souris))
        for i in np.arange(len(slices_list)):
            s.append(io.imread(path_souris + slices_list[i]))
        souris = np.array(s)

    data = utils.contraste_and_reshape(souris)

    model_axial = keras.models.load_model(path_model_axial, custom_objects={'mean_iou': utils.mean_iou})
    model_sagital = keras.models.load_model(path_model_sagital, custom_objects={'mean_iou': utils.mean_iou})
    model_corronal = keras.models.load_model(path_model_corronal, custom_objects={'mean_iou': utils.mean_iou})

    seg_ax = (model_axial.predict(data) > 0.5).astype(np.uint8).reshape(128, 128, 128)

    data = data.reshape(128, 128, 128)
    data_sag = np.zeros(((128, 128, 128)))
    data_cor = np.zeros(((128, 128, 128)))

    for i in np.arange(128):
        data_sag[i] = data[:, i, :]
        data_cor[i] = data[:, :, i]

    data_sag = data_sag.reshape(128, 128, 128, 1)
    data_cor = data_cor.reshape(128, 128, 128, 1)

    seg_sag = (model_sagital.predict(data_sag) > 0.5).astype(np.uint8).reshape(128, 128, 128)
    seg_cor = (model_corronal.predict(data_cor) > 0.5).astype(np.uint8).reshape(128, 128, 128)

    result_mask = np.zeros(((128, 128, 128)))
    for x in np.arange(128):
        for y in np.arange(128):
            for z in np.arange(128):
                if (seg_ax[x][y, z] + seg_sag[y][x, z] + seg_cor[z][x, y]) >= vote_model:
                    result_mask[x][y, z] = 1
                else:
                    result_mask[x][y, z] = 0

    if visu_seg:
        if not os.path.exists(path_result + str(name_folder)):
            os.makedirs(path_result + str(name_folder))

        for k in np.arange(128):

            cell_contours = measure.find_contours(result_mask[k], 0.8)
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
            for n, contour in enumerate(cell_contours):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
            plt.xlim((0, 128))
            plt.ylim((128, 0))
            plt.imshow(data[k], cmap='gray');
            plt.savefig(path_result + str(name_folder) + "/m_" + str(k) + ".png")
            plt.close(fig)

    if mask:
        return result_mask



def seg_meta_original(path_souris, path_model_seg_meta, path_result, name_folder):

    """
    :param path_souris: image.tif étant un ensemble de stack
    :param path_model_seg_meta: modèle de segmentation de métastases format h5
    :param path_result: chemin où sauvegarder le résultat (path/)
    :param name_folder: nom du dossier où stocker les résultats
    :return:
    """

    souris = io.imread(path_souris, plugin='tifffile')
    data = utils.contraste_and_reshape(souris)

    model_seg = keras.models.load_model(path_model_seg_meta, custom_objects={'mean_iou': utils.mean_iou})
    seg = (model_seg.predict(data) > 0.5).astype(np.uint8).reshape(128, 128, 128)
    data = data.reshape(128, 128, 128)

    if not os.path.exists(path_result + str(name_folder)):
        os.makedirs(path_result + str(name_folder))

    for k in np.arange(128):

        cell_contours = measure.find_contours(seg[k], 0.8)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        for n, contour in enumerate(cell_contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
        plt.xlim((0, 128))
        plt.ylim((128, 0))
        plt.imshow(data[k], cmap='gray');
        plt.savefig(path_result + str(name_folder) + "/m_" + str(k) + ".png")
        plt.close(fig)




def seg_meta_poum_seg(path_souris, path_model_seg_poum, path_model_seg_meta, path_result, name_folder):

    souris = io.imread(path_souris, plugin='tifffile')
    data = utils.contraste_and_reshape(souris)

    model_seg_poum = keras.models.load_model(path_model_seg_poum, custom_objects={'mean_iou': utils.mean_iou})
    seg_poum = (model_seg_poum.predict(data) > 0.5).astype(np.uint8).reshape(souris.shape[0], 128, 128)

    data = (data - data.min()) * 255 / (data.max() - data.min())

    DATA = []
    for i in np.arange(souris.shape[0]):
        DATA.append(utils.apply_mask_and_noise(data[i], seg_poum[i], 70))
    DATA = np.array(DATA).reshape(-1, 128, 128, 1)

    model_seg_meta = keras.models.load_model(path_model_seg_meta, custom_objects={'mean_iou': utils.mean_iou})
    seg_meta = (model_seg_meta.predict(data) > 0.5).astype(np.uint8).reshape(128, 128, 128)

    data = data.reshape(128, 128, 128)

    if not os.path.exists(path_result + str(name_folder)):
        os.makedirs(path_result + str(name_folder))

    for k in np.arange(128):

        cell_contours = measure.find_contours(seg_meta[k], 0.8)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        for n, contour in enumerate(cell_contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
        plt.xlim((0, 128))
        plt.ylim((128, 0))
        plt.imshow(data[k], cmap='gray');
        plt.savefig(path_result + str(name_folder) + "/m_" + str(k) + ".png")
        plt.close(fig)

