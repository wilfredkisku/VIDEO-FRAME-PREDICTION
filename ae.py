import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras.datasets import mnist, fashion_mnist

def load_data(choice='mnist', labels=False):
    if choice not in ['mnist', 'fashion_mnist']:
        raise ('Choices are mnist and fashion_mnist')

    if choice is 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train, X_test = X_train/255., X_test/255.
    #X_train, X_test = X_train.reshape((-1, 784)), X_test.reshape((-1, 784))
    X_train = X_train.astype(np.float32, copy=False)
    X_test = X_test.astype(np.float32, copy=False)

    if labels:
        return (X_train, y_train), (X_test, y_test)

    return X_train, X_test

def model_simple():

    encoding_dim = 32

    inputs = tf.keras.layers.Input(shape=784,)
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(inputs)
    decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
    model = tf.keras.models.Model(inputs, decoded)
    return model

def model_conv():
    
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = tf.keras.models.Model(inputs, decoded)
    return model

if __name__ == '__main__':

    train, test = load_data()
    print(train.shape)
    print(test.shape)
    #autoencoder = model_simple()
    autoencoder = model_conv()
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(train, train, epochs=50, batch_size=128, shuffle=True, validation_data=(test, test))

