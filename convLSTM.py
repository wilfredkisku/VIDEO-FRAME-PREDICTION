import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

import numpy as np
import pylab as plt

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3,3), input_shape=(None, 40, 40, 1), padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(3,3), padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(3,3), padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(3,3), padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(Conv3D(filters=1, kernel_size=(3,3,3), activation='sigmoid', padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')
seq.summary()

def generate_movies(n_samples=1200, n_frames=15):
    
    row = 80
    col = 80

    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)

    for i in range(n_samples):
        n = np.random.randint(3,8)

        for j in range(n):
            #initial position
            xstart = np.random.randint(20,60)
            ystart = np.random.randint(20,60)

            #direction of motion
            directionx = np.random.randint(0,3) - 1
            directiony = np.random.randint(0,3) - 1

            #size of the square
            w = np.random.randint(2,4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t

                noisy_movies[i, t, x_shift - w : x_shift + w, y_shift - w: y_shift + w, 0] += 1
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0,2)
                    noisy_movies[i, t, x_shift - w - 1: x_shift + w + 1, y_shift - w - 1: y_shift + w + 1, 0] += noise_f * 0.1

                #shift the round truth by 1
                x_shift = xstart + directionx * (t+1)
                y_shift = ystart + directiony * (t+1)
                shifted_movies[i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w, 0] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies


a,b = generate_movies(n_samples=1500)

print(a.shape, b.shape)
