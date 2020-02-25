import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Input,Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from keras import applications
import glob
from numpy import concatenate
import cv2
from keras.regularizers import l2


TRAIN_FOLDER = "../dataset/train"
VAL_FOLDER ="../dataset/val"

img_height = 256
img_width = 256
batch_size = 4
train_set_size = 342
lr = 0.00001
dropout = 0.5

def data_generator(train_datas,val_datas):
    train_datagen = ImageDataGenerator(horizontal_flip=True,rescale=1./255,width_shift_range=2.0,rotation_range=180)
    train_generator = train_datagen.flow(directory=train_datas,target_size=(img_height,img_width))
    val_datagen = ImageDataGenerator(horizontal_flip=True,rescale=1./255,width_shift_range=2.0,rotation_range=180)
    val_generator = val_datagen.flow(directory=val_datas, target_size=(img_height,img_width))

input_tensor = Input(shape=(img_height, img_width, 3))
vgg_model = applications.VGG16(weights='imagenet',
           include_top=False,
           input_tensor=input_tensor)

def images_to_array(data_dir,label):
    folder = glob.glob(data_dir)
    data_array = []
    labels_array = []
    for fname in folder:
        img = cv2.imread(fname)
        data_array.append(cv2.resize(img,(img_height,img_width),interpolation = cv2.INTER_AREA))
        if(label == "male"):
            labels_array.append([0.,1.])
        elif(label =="femelle"):
            labels_array.append([1.,0,])
    return np.array(data_array),np.array(labels_array,dtype='float64')

#Train data
train_femelles,labels_femelles = images_to_array(TRAIN_FOLDER+"/femelles/*.png","femelle")
train_males,labels_males = images_to_array(TRAIN_FOLDER+"/males/*.png","male")
train_datas = np.array(concatenate((train_femelles,train_males),axis=0))
train_labels = concatenate((labels_femelles,labels_males),axis=0)

#Validation data
val_femelles,val_labels_femelles = images_to_array(VAL_FOLDER+"/femelles/*.png","femelle")
val_males,val_labels_males = images_to_array(VAL_FOLDER+"/males/*.png","male")
val_datas = np.array(concatenate((val_femelles,val_males),axis=0))
val_labels = concatenate((val_labels_femelles,val_labels_males),axis=0)

train_datagen = ImageDataGenerator(horizontal_flip=True,rescale=1./255,width_shift_range=2.0)
train_generator = train_datagen.flow(train_datas,train_labels)
val_datagen = ImageDataGenerator(horizontal_flip=True,rescale=1./255,width_shift_range=2.0)
val_generator = val_datagen.flow(val_datas,val_labels)


model = Sequential()
model.add(vgg_model)
model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu",kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(dropout))
model.add(Dense(units=2, activation="softmax"))
opt = Adam(lr=lr)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()

model_saved_name = "vgg_model"+"_height_"+str(img_height)+"_width_"+str(img_width)+"_batchsize_"+str(batch_size)+"_lr_"+str(lr)+"_dropout_"+str(dropout)+"_"+".h5"
checkpoint = ModelCheckpoint(model_saved_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping (monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')


hist = model.fit_generator(
    train_generator,
    steps_per_epoch=(train_set_size // batch_size),
    epochs=30,
    validation_data=val_generator,
    callbacks=[checkpoint,early])

#VISUALISATION
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.savefig('books_read.png')