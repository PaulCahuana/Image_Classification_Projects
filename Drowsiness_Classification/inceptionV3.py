from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import time

# redimensionamos la imagen
IMAGE_SIZE = [224, 224]

train_path = 'TRAIN'
valid_path = 'TEST'

#agregamos la capa de preprocesamiento al frente del inicio
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# no entrenamos con pesos existentes
for layer in inception.layers:
    layer.trainable = False 

# numero de clases
folders = glob('TRAIN/*')
# nuestras capas: se puede agregar más si se desea
x = Flatten()(inception.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)
# creamos el modelo
model = Model(inputs=inception.input, outputs=prediction)
# miramos la estructura del modelo
#model.summary()
# insertamos el costo, optimizador y metricas
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

#generamos nuestra data de train con un validation interno
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  validation_split = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#generamos nuestra data de test
test_datagen = ImageDataGenerator(rescale = 1./255)
# indicamos el tamaño del batch para train
training_set = train_datagen.flow_from_directory('TRAIN',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
# indicamos el tamaño del batch para test
test_set = test_datagen.flow_from_directory('TEST',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

start = time.time() #tomamos medida del tiempo
# entrenamos
r = model.fit_generator(
  training_set,
#  validation_data=test_set,
# validation_split=0.2,
  epochs=100,
  steps_per_epoch=len(training_set)/32,
  validation_steps=len(test_set)/32
)


end = time.time() #paramos el tiempo
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print()
print()
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

#guardamos el modelo
model.save('100_epochs_SISO_NTHU.h5')

# graficamos el loss
plt.plot(r.history['loss'], label='train loss')
#plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
#plt.show()
plt.savefig('100_epochs_loss_SISO_NTHU')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
#plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
#plt.show()
plt.savefig('100_epochs_acc_SISO_NTHU')

#giardamos el modelo en csv
hist_df = pd.DataFrame(r.history) 
hist_csv_file = 'history_100epochs_SISO_NTHU.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
    
print("accuracy train: ", np.mean(r.history['accuracy']))
#print("val accuracy train: ", np.mean(r.history['val_accuracy']))
