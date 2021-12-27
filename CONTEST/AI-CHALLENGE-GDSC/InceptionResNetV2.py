import tensorflow as tf
import keras

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import models, layers, optimizers, regularizers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Reading the data
training_csv = pd.read_csv('../input/gdscaidataset/trainLabels.csv')

mappings, training_labels = np.unique(
    training_csv['label'].to_numpy(copy=True), return_inverse=True)
training_labels = training_labels.reshape(-1, 1)

training_images = np.array([plt.imread(
    f'../input/gdscaidataset/train/train/{i + 1}.png') for i in range(0, 50000)])
testing_images = np.array([plt.imread(
    f'../input/gdscaidataset/test/test/{i + 1}.png') for i in range(0, 20000)])

# Import RestNet50 or InceptionResNetV2 to train
#conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_base = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))

conv_base.summary()

# calculate images size and retype it
training_images = training_images.astype('float32')
testing_images = testing_images.astype('float32')

# z-score
mean = np.mean(training_images, axis=(0, 1, 2, 3))
std = np.std(training_images, axis=(0, 1, 2, 3))
training_images = (training_images-mean)/(std+1e-7)
testing_images = (testing_images-mean)/(std+1e-7)

training_labels = np_utils.to_categorical(training_labels, 10)

print("training_images shape:", training_images.shape)
print("testing_images shape:", testing_images.shape)

baseMapNum = 2
weight_decay = 1e-4

# Add layers

model = models.Sequential()
model.add(layers.UpSampling2D((7, 7)))
model.add(conv_base)

model.add(layers.Flatten())
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(128, activation='elu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(64, activation='elu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.BatchNormalization())

model.add(Dense(4096, activation='elu', kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(4096, activation='elu', kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1000, activation='elu', kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))

# model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

#model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,decay=1e-6, nesterov=False), loss='binary_crossentropy', metrics=['accuracy'])

#model.fit(training_images, training_labels, epochs=5, batch_size=20, validation_split=0.2, verbose=1)

# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)
datagen.fit(training_images)

# Training models
batch_size = 32
epochs = 10
opt_rms = tf.keras.optimizers.SGD(
    learning_rate=0.001, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(loss='binary_crossentropy',
              optimizer=opt_rms,
              metrics=['accuracy'])
# steps_per_epoch=training_images.shape[0]// batch_size
model.fit(datagen.flow(training_images, training_labels,
          batch_size=batch_size), epochs=2*epochs, verbose=1)

opt_rms = tf.keras.optimizers.SGD(
    learning_rate=0.0005, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(loss='binary_crossentropy',
              optimizer=opt_rms,
              metrics=['accuracy'])
model.fit(datagen.flow(training_images, training_labels,
          batch_size=batch_size), epochs=epochs, verbose=1)

opt_rms = tf.keras.optimizers.SGD(
    learning_rate=0.0003, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(loss='binary_crossentropy',
              optimizer=opt_rms,
              metrics=['accuracy'])
model.fit(datagen.flow(training_images, training_labels,
          batch_size=batch_size), epochs=epochs, verbose=1)

opt_rms = tf.keras.optimizers.SGD(
    learning_rate=0.0008, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(loss='binary_crossentropy',
              optimizer=opt_rms,
              metrics=['accuracy'])
model.fit(datagen.flow(training_images, training_labels,
          batch_size=batch_size), epochs=epochs, verbose=1)

# Predict the result
prediction = model.predict(testing_images)

# Print the output images
output = pd.DataFrame({
    'label': [mappings[np.argmax(p)] for p in prediction]
})
output.index += 1
print(output)

# write the output Dataframe to a csv file. This file will be stored on your Colab engine
output.to_csv('./submission.csv', index=True, index_label='id')
