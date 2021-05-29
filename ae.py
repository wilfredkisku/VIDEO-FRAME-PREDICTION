import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras.datasets import mnist, fashion_mnist

#MNIST dataset parameters 
num_features = 784 #(image shape: 28 * 28)

#training parameters
batch_size = 128
epoch = 50

#network parameters
hidden_1 = 128
hidden_2 = 64

def load_data(choice='mnist', labels=False):
    if choice not in ['mnist', 'fashion_mnist']:
        raise ('Choices are mnist and fashion_mnist')

    if choice is 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train, X_test = X_train/255., X_test/255.
    print(X_train.shape)
    X_train, X_test = X_train.reshape((-1, 784)), X_test.reshape((-1, 784))
    print(X_train.shape)
    X_train = X_train.astype(np.float32, copy=False)
    X_test = X_test.astype(np.float32, copy=False)

    if labels:
        return (X_train, y_train), (X_test, y_test)

    return X_train, X_test

def plot_predictions(y_true, y_pred):
    
    f, ax = plt.subplots(2, 10, figsize=(15, 4))
    
    for i in range(10):
        ax[0][i].imshow(np.reshape(y_true[i], (28,28)), aspect='auto')
        ax[1][i].imshow(np.reshape(y_true[i], (28,28)), aspect='auto')
    plt.tight_layout()

if __name__ == '__main__':

    load_data()
