import keras
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
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
    fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(128, 128, 1), padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(Dense(2, activation='softmax'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam()
                          , metrics=['accuracy'])
    return fashion_model


def model_unet_2D(input_shape, wei=False):
    """
    :param input_shape: dimension des images en entrées, typiquement (128,128,1)
    :param wei: utilisation ou non des weight map pour entrainer le modèle
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
    if not wei:
        model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[utils.mean_iou])
    else:
        model.compile(optimizer='adam', loss=utils.weighted_cross_entropy)
    return model


def methode_detect_seg(path_souris, path_model_detect, path_model_seg, path_result,
                       name_folder, mask=None, full_souris=True, visu_seg=True, img=None, wei=False):
    """
    Méthode 2D qui permet de segmenter les poumons slice par slice. La segmentation est enfin ajustée grâce au modèle
    de détection (si le modèle ne détecte pas de poumons alors le masque est vide)
    -------------------------------------------------------------------------------------------------------------------
    :param path_souris: image.tif étant un ensemble de slices
    :param path_model_detect: modèle de detection de slices pour les poumons - format h5
    :param path_model_seg: modèle de segmentation des poumons - format h5
    :param path_result: chemin où sauvegarder le résultat (path/)
    :param name_folder: nom du dossier où sauvegarder les résultats
    :param mask: si True alors retourne les prédictions de détection ainsi que la valeur des masques associés
    :param full_souris: True correspond à Souris 3D.tiff sinon ensemble de slices.tiff
    :param visu_seg: True correspond à la sauvegarde des contours sur les images
    :param wei: utilisation d'un modèle avec poids modifiés
    -------------------------------------------------------------------------------------------------------------------
    :return: Ensemble des slices d'une image avec contour des poumons segmentés (si visu==True)
    :return: Retourne les prédictions de détection ainsi que la valeur des masques associés (si mask==True)
    -------------------------------------------------------------------------------------------------------------------
    """

    if full_souris:
        souris = io.imread(path_souris, plugin='tifffile')

    else:
        slices_list = utils.sorted_aphanumeric(os.listdir(path_souris))
        s = np.zeros(((len(slices_list), 128, 128)))
        for i in np.arange(len(slices_list)):
            s[i] = io.imread(path_souris + slices_list[i])
        souris = np.array(s)

    data = utils.contraste_and_reshape(souris)

    model_detect = keras.models.load_model(path_model_detect)

    if not wei:
        modele_seg = keras.models.load_model(path_model_seg, custom_objects={'mean_iou': utils.mean_iou})
    else:
        modele_seg = keras.models.load_model(path_model_seg,
                                             custom_objects={'weighted_cross_entropy': utils.weighted_cross_entropy})

    detect = model_detect.predict_classes(data)
    seg = (modele_seg.predict(data) > 0.5).astype(np.uint8).reshape(128, 128, 128)
    data = data.reshape(128, 128, 128)

    if visu_seg:

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
        return detect, seg#[ind]


def methode_multi_axe(path_souris, path_model_axial, path_model_sagital, path_model_corronal,
                      path_result, name_folder, vote_model=2, mask=None, full_souris=True, visu_seg=True):
    """
    -------------------------------------------------------------------------------------------------------------------
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
    -------------------------------------------------------------------------------------------------------------------
    :return: Ensemble d'image avec contour des poumons segmentés si visu
    :return: Masque si mask
    -------------------------------------------------------------------------------------------------------------------
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


def seg_meta_original(path_souris, path_model_seg_meta, path_result, name_folder, mask=None, visu_seg=True, wei=False):
    """
    :param path_souris: image.tif étant un ensemble de stack
    :param path_model_seg_meta: modèle de segmentation de métastases format h5
    :param path_result: chemin où sauvegarder le résultat (path/)
    :param name_folder: nom du dossier où stocker les résultats
    :return:
    """

    souris = io.imread(path_souris, plugin='tifffile')
    data = utils.contraste_and_reshape(souris)

    if not wei:
        model_seg = keras.models.load_model(path_model_seg_meta, custom_objects={'mean_iou': utils.mean_iou})

    else:
        model_seg = keras.models.load_model(path_model_seg_meta,
                                            custom_objects={'weighted_cross_entropy': utils.weighted_cross_entropy})

    seg = (model_seg.predict(data) > 0.5).astype(np.uint8).reshape(128, 128, 128)
    data = data.reshape(128, 128, 128)

    if visu_seg:

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

    if mask:
        return seg



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


#####################################################################
####################### ResNET #######################
#####################################################################

## Je n'ai pas utilisé le ResNet mais cela peut être intéressant de le considérer comme backbone pour le U-Net.

def identity_block(X, f, filters, stage, block):
    """
    - Implementation of the identity block -
    :param X: input tensor of shape (m,H,W,C)
    :param f: integer, specifying the shape of the middle conv's window for the main path
    :param filters: python list of integers, defining the number of filters in the conv layer of the main path
    :param stage: integer, used to name the layers, depending on their position in the network
    :param block: string/character, used to name the layers, depending on their position in the network
    :return: X -- output of the identity block, tensor of shape (H,W,C
    """

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation("relu")(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step : Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    - Implementation of the convolutional block -
    :param X: input tensor of shape (m,H,W,C)
    :param f: integer, specifying the shape of the middle conv's window for the main path
    :param filters: python list of integers, defining the number of filters in the conv layers of the main path
    :param stage: integer, used to name the layers, depending on their position in the network
    :param block: string/character, used to name the layers, depending on their position in the network
    :param s: Integer, specifying the stride to be used
    :return: X -- output of the convolutional block, tensor of shape (H,W,C)
    """
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ### Main Path ###
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation("relu")(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ### Shortcut Path ###
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a Relu activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(128, 128, 1), classes=2):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # Average Pooling
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


#######################################################################
      ####################### UNET ++ #######################
#######################################################################


def block_down(inputs, filters, drop):
    x = Conv2D(filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    x = Dropout(drop)(x)
    c = Conv2D(filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(x)
    p = MaxPooling2D((2, 2))(c)
    return c, p


def bridge(inputs, filters, drop):
    x = Conv2D(filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(x)
    return x


def block_up(inputs, conc, filters, drop):
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    for i in np.arange(len(conc)):
        x = concatenate([x, conc[i]])
    x = Conv2D(filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(x)
    return x



def unet_plusplus(input_shape):

    inputs = Input(input_shape)

    c1, p1 = block_down(inputs, filters=16, drop=0.1)
    c2, p2 = block_down(p1, filters=32, drop=0.1)
    c3, p3 = block_down(p2, filters=64, drop=0.2)
    c4, p4 = block_down(p3, filters=128, drop=0.2)

    o = bridge(p4, filters=256, drop=0.3)

    u4 = block_up(o, [c4], filters=128, drop=0.2)

    n3_1 = block_up(c4, [c3], filters=64, drop=0.2)
    u3 = block_up(u4, [n3_1, c3], filters=64, drop=0.2)

    n2_1 = block_up(c3, [c2], filters=32, drop=0.1)
    n2_2 = block_up(n3_1, [n2_1, c2], filters=32, drop=0.1)
    u2 = block_up(u3, [n2_2, n2_1, c2], filters=32, drop=0.1)

    n1_1 = block_up(c2, [c1], filters=16, drop=0.1)
    n1_2 = block_up(n2_1, [n1_1, c1], filters=16, drop=0.1)
    n1_3 = block_up(n2_2, [n1_2, n1_1, c1], filters=16, drop=0.1)

    u1 = block_up(u2, [n1_3, n1_2, n1_1, c1], filters=16, drop=0.1)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(u1)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[utils.mean_iou])
    return model




#######################################################################
      ####################### Small U-Net #######################
#######################################################################


def small_unet(input_shape):
    """
    :param input_shape: dimension des images en entrées, typiquement (128,128,1)
    :return: Compile un réseau Small-UNET
    """

    inputs = Input(input_shape)

    c1, p1 = block_down(inputs, filters=32, drop=0.1)
    c2, p2 = block_down(p1, filters=64, drop=0.1)
    c3, p3 = block_down(p2, filters=128, drop=0.2)

    o = bridge(p3, filters=256, drop=0.3)

    u4 = block_up(o, [c3], filters=128, drop=0.2)
    u5 = block_up(u4, [c2], filters=64, drop=0.1)
    u6 = block_up(u5, [c1], filters=32, drop=0.1)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u6)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[utils.mean_iou])
    return model


def unetCoupe2Max(input_shape):
    """
    :param input_shape: dimension des images en entrées, typiquement (128,128,1)
    :return: Compile un réseau Small-UNET
    """

    inputs = Input(input_shape)

    c1, p1 = block_down(inputs, filters=32, drop=0.1)
    c2, p2 = block_down(p1, filters=64, drop=0.1)

    o = bridge(p2, filters=128, drop=0.3)

    u5 = block_up(o, [c2], filters=64, drop=0.1)
    u6 = block_up(u5, [c1], filters=32, drop=0.1)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u6)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[utils.mean_iou])
    return model



#######################################################################
      ####################### U-Net Lstm #######################
#######################################################################


def bclstm_unet(input_shape):

    inputs = Input(input_shape)

    c2 = TimeDistributed(Conv2D(32, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(inputs)
    c2 = TimeDistributed(Dropout(0.1))(c2)
    c2 = TimeDistributed(Conv2D(32, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(c2)
    p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)

    c3 = TimeDistributed(Conv2D(64, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(p2)
    c3 = TimeDistributed(Dropout(0.2))(c3)
    c3 = TimeDistributed(Conv2D(64, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(c3)
    p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

    c4 = TimeDistributed(Conv2D(128, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(p3)
    c4 = TimeDistributed(Dropout(0.2))(c4)
    c4 = TimeDistributed(Conv2D(128, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(c4)
    p4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(c4)

    b1 = Bidirectional(ConvLSTM2D(128, (3, 3), activation='tanh', kernel_initializer='he_normal',
                                  padding='same', return_sequences=True), merge_mode='concat')(p4)
    b2 = Bidirectional(ConvLSTM2D(128, (3, 3), activation='tanh', kernel_initializer='he_normal',
                                  padding='same', return_sequences=True), merge_mode='concat')(b1)

    u6 = TimeDistributed(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'))(b2)
    u6 = concatenate([u6, c4])
    c6 = TimeDistributed(Conv2D(128, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(u6)
    c6 = TimeDistributed(Dropout(0.2))(c6)
    c6 = TimeDistributed(Conv2D(128, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(c6)

    u7 = TimeDistributed(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'))(c6)
    u7 = concatenate([u7, c3])
    c7 = TimeDistributed(Conv2D(64, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(u7)
    c7 = TimeDistributed(Dropout(0.2))(c7)
    c7 = TimeDistributed(Conv2D(64, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(c7)

    u8 = TimeDistributed(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same'))(c7)
    u8 = concatenate([u8, c2])
    c8 = TimeDistributed(Conv2D(32, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(u8)
    c8 = TimeDistributed(Dropout(0.1))(c8)
    c8 = TimeDistributed(Conv2D(32, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same'))(c8)

    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid'))(c8)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model



def seg_poum_lstm(path_souris, path_model_detect, path_model_seg, time):

    souris = io.imread(path_souris, plugin='tifffile')
    data = utils.contraste_and_reshape(souris)

    model_detect = keras.models.load_model(path_model_detect)
    detect = model_detect.predict_classes(data)[0:int(128/time)*time]

    model_bclstm = keras.models.load_model(path_model_seg, custom_objects={'mean_iou': utils.mean_iou})

    Data = data[0:int(128/time)*time].reshape(int(128/time), time, 128, 128, 1)
    pred = (model_bclstm.predict(Data) > 0.5).astype(np.uint8).reshape(int(128/time)*time, 128, 128)

    return detect, pred








def methode_detect_seg_2(path_souris, path_model_detect, path_model_seg, path_model_seg_meta, path_result,
                       name_folder, mask=None, full_souris=True, visu_seg=True, img=None, wei=False):

    if full_souris:
        souris = io.imread(path_souris, plugin='tifffile')

    else:
        slices_list = utils.sorted_aphanumeric(os.listdir(path_souris))
        s = np.zeros(((len(slices_list), 128, 128)))
        for i in np.arange(len(slices_list)):
            s[i] = io.imread(path_souris + slices_list[i])
        souris = np.array(s)

    data = utils.contraste_and_reshape(souris)

    model_detect = keras.models.load_model(path_model_detect)

    if not wei:
        modele_seg = keras.models.load_model(path_model_seg, custom_objects={'mean_iou': utils.mean_iou})
    else:
        modele_seg = keras.models.load_model(path_model_seg,
                                             custom_objects={'weighted_cross_entropy': utils.weighted_cross_entropy})

    model_seg_meta = keras.models.load_model(path_model_seg_meta, custom_objects={'mean_iou': utils.mean_iou})

    detect = model_detect.predict_classes(data)
    seg = (modele_seg.predict(data) > 0.5).astype(np.uint8).reshape(128, 128, 128)
    seg_meta = (model_seg_meta.predict(data) > 0.5).astype(np.uint8).reshape(128, 128, 128)
    data = data.reshape(128, 128, 128)

    if visu_seg:

        if not os.path.exists(path_result + str(name_folder)):
            os.makedirs(path_result + str(name_folder))

        for k in np.arange(128):
            cell_contours = measure.find_contours(seg[k], 0.8)
            cell_contours2 = measure.find_contours(seg_meta[k], 0.8)

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
            for n, contour in enumerate(cell_contours):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
            for n, contour in enumerate(cell_contours2):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='green')
            plt.xlim((0, 128))
            plt.ylim((128, 0))
            plt.imshow(data[k], cmap='gray');
            plt.savefig(path_result + str(name_folder) + "/m_" + str(k) + ".png")
            plt.close(fig)

    if mask:
        ind = np.where(detect == 1)
        return detect, seg#[ind]






def visu_souris(path_souris, path_result, name_folder):


    souris = io.imread(path_souris, plugin='tifffile')
    data = utils.contraste_and_reshape(souris)

    data = data.reshape(128, 128, 128)

    os.makedirs(path_result + str(name_folder))

    for k in np.arange(128):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))

        plt.xlim((0, 128))
        plt.ylim((128, 0))
        plt.imshow(data[k], cmap='gray');
        plt.savefig(path_result + str(name_folder) + "/m_" + str(k) + ".png")
        plt.close(fig)

souris_8 = "/home/achauviere/Bureau/DATA/Souris_Test/Souris/souris_8.tif"
souris_28 = "/home/achauviere/Bureau/DATA/Souris_Test/Souris/souris_28.tif"
souris_56 = "/home/achauviere/Bureau/DATA/Souris_Test/Souris/souris_56.tif"
