# -*- coding: utf-8 -*-
from __future__ import print_function
import random

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K

from boss_input import extract_data, resize_with_pad, IMAGE_SIZE


class Dataset(object):

    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None

    def read(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):
        images, labels = extract_data('./data/')
        labels = np.reshape(labels, (-1,))
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=random.randint(0, 100))
        X_valid, X_test, y_valid, y_test = train_test_split(images, labels, test_size=0.5, random_state=random.randint(0, 100))
        if K.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
            X_valid = X_valid.reshape(X_valid.shape[0], 3, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
            X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 3)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')

        Y_train = to_categorical(y_train, nb_classes)
        Y_valid = to_categorical(y_valid, nb_classes)
        Y_test = to_categorical(y_test, nb_classes)

        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_valid /= 255
        X_test /= 255

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_valid = Y_valid
        self.Y_test = Y_test


class Model(object):

    FILE_PATH = './store/model.h5'

    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=2):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=dataset.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train(self, dataset, batch_size=32, epochs=40, data_augmentation=True):
        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])
        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(dataset.X_train, dataset.Y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(dataset.X_valid, dataset.Y_valid),
                          shuffle=True)
        else:
            print('Using real-time data augmentation.')
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False)

            datagen.fit(dataset.X_train)
            self.model.fit(datagen.flow(dataset.X_train, dataset.Y_train, batch_size=batch_size),
                          steps_per_epoch=dataset.X_train.shape[0] // batch_size,
                          epochs=epochs,
                          validation_data=(dataset.X_valid, dataset.Y_valid))

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    def predict(self, image):
        if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_with_pad(image)
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        elif K.image_data_format() == 'channels_last' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_with_pad(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        image = image.astype('float32')
        image /= 255
        result = self.model.predict(image)
        print(result)
        result = np.argmax(result, axis=-1)
        return result[0]

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))


if __name__ == '__main__':
    dataset = Dataset()
    dataset.read()

    model = Model()
    model.build_model(dataset)
    model.train(dataset, epochs=10)
    model.save()

    model = Model()
    model.load()
    model.evaluate(dataset)