import keras_resnet.models
import keras
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np

img_height = 32
img_width = 32
lr = 0.001

IMAGE_NAME = "test_image_femelle.png"
IMAGE_NAME_2 = "test_image_femelle.png" # ATTENTION : L'image doit être dans le même dossier que ce script

img = image.load_img(IMAGE_NAME,target_size=(img_height,img_width))
img2 = image.load_img(IMAGE_NAME_2,target_size=(img_height,img_width))
img = np.asarray(img)
img2 = np.asarray(img2)
img = np.expand_dims(img, axis=0)
img2 = np.expand_dims(img, axis=0)


print("test")
"""def resnet_model(img_height, img_width, lr):
    # CREATION DU RESEAU
    shape, classes = (img_height, img_width, 3), 2  # A MODIFIER SI TU VEUX MODIFIER LA TAILLE DES IMAGES DANS LE RESEAU
    x = keras.layers.Input(shape)
    model = keras_resnet.models.ResNet50(x, classes=classes)

    model.load_weights("resnet_model_weights.h5")  # Je n'ai pas pu récupérer tt le réseau car il ne reconnait
    opt = Adam(
        lr)  # pas certains layers à cause de la librairie keras_resnet. J'ai donc save et récupéré que les weights.
    model.compile(opt, "categorical_crossentropy", ["accuracy"])
    return model


model = resnet_model()
output = model.predict(img)"""