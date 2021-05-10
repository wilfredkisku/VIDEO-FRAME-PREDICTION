import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, UpSampling2D
from keras import backend as K

import random
import glob
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, figure

num_epochs = 100
batch_size = 32

#################################################################
#gify dataset
height = 96
width = 96

val_dir = '/home/wilfred/Datasets/catz/test'
train_dir = '/home/wilfred/Datasets/catz/train'
#################################################################
#################################################################
#UCF-101 dataset (augmented by traffic dataset)
#height_ucf = 320
#width_ucf = 480

#val_dir_ucf = '/home/wilfred/Datasets/UCF-101'
#train_dir_ucf = '/home/wilfred/Dataset/UCF-101'
#################################################################

steps_per_epoch = len(glob.glob(train_dir + "/*")) // batch_size
validation_steps = len(glob.glob(val_dir + "/*")) // batch_size

def my_generator(batch_size, img_dir):
    dirs = glob.glob(img_dir + '/*')
    counter = 0
    while True:
        input_images = np.zeros((batch_size, width, height, 3*5))
        output_images = np.zeros((batch_size, width, height, 3))
        random.shuffle(dirs)
        if (counter+batch_size >= len(dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(dirs[counter + i] + '/cat_[0-5]*')
            imgs = [Image.open(img) for img in sorted(input_imgs)]
            input_images[i] = np.concatenate(imgs, axis = 2)
            output_images[i] = np.array(Image.open(dirs[counter + i] + '/cat_result.jpg'))
            input_images[i] /= 255.
            output_images[i] /= 255.

        yield (input_images, output_images)
        counter += batch_size

gen = my_generator(52, train_dir)
videos, next_frame = next(gen)
print(videos[0].shape)
print(next_frame[0].shape)

print(videos.shape)
