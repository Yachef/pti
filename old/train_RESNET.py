import keras
import keras_resnet.models
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# SOURCE : https://github.com/broadinstitute/keras-resnet

TRAIN_DIRECTORY = "dataset/train"  #chemin des données de train
VAL_DIRECTORY = "dataset/val" #chemin des données de validation

img_height = 32
img_width = 32
#CREATION DU RESEAU
shape, classes = (img_height, img_width, 3), 2 # A MODIFIER SI TU VEUX MODIFIER LA TAILLE DES IMAGES DANS LE RESEAU
x = keras.layers.Input(shape)
model = keras_resnet.models.ResNet50(x, classes=classes)
opt = Adam(lr=0.001)
model.compile(opt, "categorical_crossentropy", ["accuracy"])

#RECUPERATION DES DATAS
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(directory=TRAIN_DIRECTORY,target_size=(img_height,img_width))
val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow_from_directory(directory=VAL_DIRECTORY, target_size=(img_height,img_width))

checkpoint = ModelCheckpoint("resnet_model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=1,generator=train_generator, validation_data= val_generator, validation_steps=10,epochs=1,callbacks=[checkpoint,early])


#VISUALISATION
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()