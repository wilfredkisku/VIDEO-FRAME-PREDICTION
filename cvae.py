from __future__ import absolute_import, division, print_function, unicode_literals
#libraries 
import tensorflow as tf
import os

import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from scipy.stats import norm
from IPython import display

#load the MNIST dataset with the test and train data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#print(train_images.shape)

#reshape the images 
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')

#print(train_images.shape)

train_images /= 255.
test_images /= 255.

#binarizing the training and test data
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

#defining the training and test data numbers and batch size
TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000

#creating the Dataset object for feeding data to the model (suffle and batch the data)
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (28,28,1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim),
            ])

        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7,7,32)),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2,2), padding='SAME', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2,2), padding='SAME', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=(1,1), padding='SAME')
            ])

    def inference_net_summary(self):
        return self.inference_net.summary()
    def generative_net_summary(self):
        return self.generative_net.summary()

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
            return self.decode(eps, apply_sigmoid=True)
    def encode(self, x):
        mean,  logvar = tf.split(self.inference_net(x), num_of_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps*tf.exp(logvar*.5)+mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

model = CVAE(2)
model.inference_net_summary()
model.generative_net_summary()
'''
