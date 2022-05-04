from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import time


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'TRAIN'
valid_path = 'TEST'

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False 

# useful for getting number of classes
folders = glob('TRAIN/*')
# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)
# create a model object
model = Model(inputs=vgg.input, outputs=prediction)
# view the structure of the model

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)



train_datagen = ImageDataGenerator(rescale = 1./255,
                                  validation_split = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('TRAIN',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('TEST',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

start = time.time() #tomamos medida del tiempo

r = model.fit_generator(
  training_set,
#  validation_data=test_set,
# validation_split=0.2,
  epochs=100,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)



end = time.time() #paramos el tiempo
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print()
print()
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

#save our model
model.save('100_epochs_SISO_NTHU_VGG16.h5')

# loss
plt.plot(r.history['loss'], label='train loss')
#plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
#plt.show()
plt.savefig('100_epochs_loss_SISO_NTHU_VGG16')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
#plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
#plt.show()
plt.savefig('100_epochs_acc_SISO_NTHU_VGG16')

#save the history of our model
hist_df = pd.DataFrame(r.history) 
hist_csv_file = 'history_100epochs_SISO_NTHU_VGG16.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
    
print("accuracy train: ", np.mean(r.history['accuracy']))
#print("val accuracy train: ", np.mean(r.history['val_accuracy']))
