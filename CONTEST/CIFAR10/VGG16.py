import numpy as np
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import maxnorm
from tensorflow.keras import optimizers

# Load Data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

y_train = np.utils.to_categorical(y_train)
y_test = np.utils.to_categorical(y_test)


def get_model():
    # Load VGG16
    model_vgg16_conv = VGG16(
        weights='imagenet', include_top=False)

    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Add layer FC and Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1',
              kernel_constraint=maxnorm(3))(x)
    x = BatchNormalization()
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2',
              kernel_constraint=maxnorm(3))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()
    x = Dense(10, activation='softmax', name='predictions')(x)

    # Compile the model
    opt_sgd = optimizers.SGD(
        learning_rate=0.001, momentum=0.9, decay=1e-6, nesterov=False)
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy',
                     optimizer=opt_sgd, metrics=['accuracy'])
    return my_model


vggmodel = get_model()
filepath = "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
                         rescale=1./255,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         brightness_range=[0.2, 1.5], fill_mode="nearest")

aug_val = ImageDataGenerator(rescale=1./255)
vgghist = vggmodel.fit_generator(aug.flow(X_train, y_train, batch_size=64),
                                 # steps_per_epoch=len(X_train)//64,
                                 epochs=50,
                                 validation_data=aug.flow(X_test, y_test,
                                                          batch_size=64),
                                 callbacks=callbacks_list)

vggmodel.save("vggmodel.h5")
