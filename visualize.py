from skimage import io
import utils
import keras
from keras import models
import matplotlib.pyplot as plt
import numpy as np




######## Visualisation des features maps ########

# On choisit une image :
souris_28 = "/home/achauviere/Bureau/DATA/Souris_Test/souris_28.tif"
data = utils.contraste_and_reshape(io.imread(souris_28))
img = data[50]
img = img.reshape(1,128,128,1)

# On load le modele
path_model_seg = "/home/achauviere/PycharmProjects/Antoine_Git/model/model_seg.h5"
modele_seg = keras.models.load_model(path_model_seg, custom_objects={'mean_iou': utils.mean_iou})

# Extracts the outputs of every layers
layer_name=None
outputs = [layer.output for layer in modele_seg.layers if
               layer.name == layer_name or layer_name is None][1:]

# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=modele_seg.input, outputs=outputs)

# Returns a list of five Numpy arrays: one array per layer activation
activations = activation_model.predict(img)

# Représentation des Features maps
layer_names = []
for layer in modele_seg.layers[20:]:
    layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
    n_features = layer_activation.shape[-1]  # Number of features in the feature map
    size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,  # Displays the grid
            row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')