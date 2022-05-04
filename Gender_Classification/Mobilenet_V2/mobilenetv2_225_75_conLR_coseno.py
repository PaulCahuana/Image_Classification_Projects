from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf

# re-size all the images
IMAGE_SIZE = [225, 75]
train_path = 'dataset_genero/train'
valid_path = 'dataset_genero/val'
# add preprocessing layer to the front of VGG
mobilenet_v2 = MobileNetV2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in mobilenet_v2.layers:
    layer.trainable = False 

# useful for getting number of classes
folders = glob(train_path+'/*')
folders
# our layers - you can add more if you want
x = Flatten()(mobilenet_v2.output)
# add  output layer
prediction = Dense(len(folders), activation='softmax')(x)
# create a model object
model = Model(inputs=mobilenet_v2.input, outputs=prediction)
# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
initial_learning_rate=0.001
decay_steps=10000
lr_schedule = tf.keras.experimental.CosineDecay(
    initial_learning_rate, decay_steps, alpha=0.0, name=None
)
optimizer_ = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(
  loss='categorical_crossentropy', #binary_crossentropy
  #optimizer='adam',
  optimizer=optimizer_,
  #optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  metrics=['accuracy']
)
train_datagen = ImageDataGenerator(rescale = 1./255)
                                   #validation_split = 0.2,
                                   #shear_range = 0.2,
                                   #zoom_range = 0.2,
                                   #horizontal_flip = True

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (225, 75),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (225, 75),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# fit the model
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')


# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


model.save('mobilenet_v2-50epochs_224.h5')