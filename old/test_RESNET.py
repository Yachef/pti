import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import keras_resnet.models
import keras
from keras.optimizers import Adam

# SOURCE : https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

IMAGE_NAME = "test_image_femelle.png" # ATTENTION : L'image doit être dans le même dossier que ce script

img_height = 32
img_width = 32

img = image.load_img(IMAGE_NAME,target_size=(img_height,img_width))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

#CREATION DU RESEAU
shape, classes = (img_height, img_width, 3), 2 # A MODIFIER SI TU VEUX MODIFIER LA TAILLE DES IMAGES DANS LE RESEAU
x = keras.layers.Input(shape)
model = keras_resnet.models.ResNet50(x, classes=classes)

model.load_weights("resnet_model_weights.h5") # Je n'ai pas pu récupérer tt le réseau car il ne reconnait
opt = Adam(lr=0.001) # pas certains layers à cause de la librairie keras_resnet. J'ai donc save et récupéré que les weights.
model.compile(opt, "categorical_crossentropy", ["accuracy"])


output = model.predict(img)
if output[0][0] > output[0][1]:
    print("femelle")
else:
    print('male')