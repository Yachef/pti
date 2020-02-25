import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# SOURCE : https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

IMAGE_NAME = "img_A_00194.png" # ATTENTION : L'image doit être dans le même dossier que ce script

img = image.load_img(IMAGE_NAME,target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

saved_model = load_model("vgg_model.h5")
output = saved_model.predict(img)
if output[0][0] > output[0][1]:
    print("femelle")
else:
    print('male')