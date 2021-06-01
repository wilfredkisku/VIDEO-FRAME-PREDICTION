from __future__ import absolute_import, division, print_function, unicode_literals
#libraries 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
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
    #init constructor 
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
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1,1), padding='SAME')
            ])

    #return then summary of the encoder and the decoder network    
    def inference_net_summary(self):
        return self.inference_net.summary()
    def generative_net_summary(self):
        return self.generative_net.summary()

    #sample epsilon from the normal distribution
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
            return self.decode(eps, apply_sigmoid=True)

    #encode the input x with a dimension of (None, 28, 28, 1) grayscale or binary image
    def encode(self, x):
        mean,  logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    #reparameterize with z = (eps * var) + mean
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps*tf.exp(logvar*.5)+mean
    
    #decode with a sampled value of z from the normal distribution
    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

model = CVAE(2)
model.inference_net_summary()
model.generative_net_summary()

optimizer = tf.keras.optimizers.Adam()
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean)**2. * tf.exp(-logvar)), axis=raxis)

@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    #use the SSIM loss for evaluating the error between the two images
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1,2,3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#variables
epochs =25
latent_dim = 2

#run the training and validation
history = []
for epoch in range(1, epochs+1):
    start_time = time.time()
    training_loss = tf.keras.metrics.Mean()

    for train_x in train_dataset:
        compute_apply_gradients(model, train_x, optimizer)
        training_loss(compute_loss(model, train_x))
    training_elbo = -training_loss.result()
    history.append(training_elbo)
    end_time = time.time()

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, Train set ELBO: {}, time elapse for current epoch {}'.format(epoch,elbo,training_elbo, end_time - start_time))


