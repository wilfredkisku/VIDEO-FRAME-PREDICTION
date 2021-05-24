import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Lambda, Reshape, Permute, Input, add, Conv3D, GaussianNoise, concatenate
from keras.layers import ConvLSTM2D, BatchNormalization, TimeDistributed, Add
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, UpSampling2D
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import Callback
from keras import backend as K

import random
import subprocess
import glob
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, figure

#define the hyper-parameters
num_epochs = 100
batch_size = 32

#traffic dataset 4:3 image ratio
height = 192
width = 256

#steps per epoch and validation steps
#steps_per_epoch = len(glob.glob(train_dir + "/*")) // batch_size
#validation_steps = len(glob.glob(val_dir + "/*")) // batch_size

#callback to log the images
#class ImageCallback(Callback):
#    def on_epoch_end(self, epoch, logs=None):
#        validation_X, validation_y = next(my_generator(15, val_dir))
#        output = self.model.predict(validation_X)
#        print("The average loss for {} epoch is {}.".format(epoch,output))

def psnr_mean(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim_loss(true, pred):
    return 1 - tf.reduce_mean(tf.image.ssim(true, pred, 1.0))

def slice(x):
    return x[:,:,:,-1]

def perceptual_distance(y_true, y_pred):
    y_true *= 255.
    y_pred *= 255.
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def create_model():
    
    #increment the filters by a factor of 4
    c = 4
    
    #model beginning
    inp = Input((height, width, 4 * 1))
    reshaped = Reshape((height,width,4,1))(inp)
    permuted = Permute((1,2,4,3))(reshaped)
    noise = GaussianNoise(0.1)(permuted)
    last_layer = Lambda(slice, input_shape=(height,width,1,4), output_shape=(height,width,1))(noise)
    x = Permute((4,1,2,3))(noise)
    x =(ConvLSTM2D(filters=c, kernel_size=(3,3),padding='same',name='conv_lstm1', return_sequences=True))(x)

    c1=(BatchNormalization())(x)
    x = Dropout(0.2)(x)
    x =(TimeDistributed(MaxPooling2D(pool_size=(2,2))))(c1)

    x =(ConvLSTM2D(filters=2*c,kernel_size=(3,3),padding='same',name='conv_lstm3',return_sequences=True))(x)
    c2=(BatchNormalization())(x)
    x = Dropout(0.2)(x)

    x =(TimeDistributed(MaxPooling2D(pool_size=(2,2))))(c2)
    x =(ConvLSTM2D(filters=4*c,kernel_size=(3,3),padding='same',name='conv_lstm4',return_sequences=True))(x)

    x =(TimeDistributed(UpSampling2D(size=(2, 2))))(x)
    x =(ConvLSTM2D(filters=4*c,kernel_size=(3,3),padding='same',name='conv_lstm5',return_sequences=True))(x)
    x =(BatchNormalization())(x)

    x =(ConvLSTM2D(filters=2*c,kernel_size=(3,3),padding='same',name='conv_lstm6',return_sequences=True))(x)
    x =(BatchNormalization())(x)
    x = Add()([c2, x])
    x = Dropout(0.2)(x)

    x =(TimeDistributed(UpSampling2D(size=(2, 2))))(x)
    x =(ConvLSTM2D(filters=c,kernel_size=(3,3),padding='same',name='conv_lstm7',return_sequences=False))(x)
    x =(BatchNormalization())(x)
    combined = concatenate([last_layer, x])
    combined = Conv2D(1, (1,1))(combined)
    model=Model(inputs=[inp], outputs=[combined])
    return model

def save_model_checkpoints():
    #tf.keras.callbacks.ModelCheckpoint allows to save the model continually
    #during and at the end of the training
    checkpoint_path = 'training/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #create a callback that saves the model weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only = True, verbose=1)
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only = True, verbose=1, save_freq=5*batch_size)
    return cp_callback

def save_model_complete():
    model.save('models/my_model.h5')
    #model.save_weights('models/my_model_weights.h5')
    #new_model = tf.keras.models.load_model('my_model.h5')

    return None

#the generator creates batches of images during training
def my_generator(batch_size, img_dir):
    dirs = glob.glob(img_dir + '/*')
    counter = 0
    while True:
        input_images = np.zeros((batch_size, width, height, 3*4))
        output_images = np.zeros((batch_size, width, height, 3))
        random.shuffle(dirs)
        if (counter+batch_size >= len(dirs)):
            counter = 0
        for i in range(batch_size):
            #change the logic for obtaining the images
            input_imgs = glob.glob(dirs[counter + i] + '/cat_[0-5]*')
            imgs = [Image.open(img) for img in sorted(input_imgs)]
            input_images[i] = np.concatenate(imgs, axis = 2)
            output_images[i] = np.array(Image.open(dirs[counter + i] + '/cat_result.jpg'))
            input_images[i] /= 255.
            output_images[i] /= 255.

        yield (input_images, output_images)
        counter += batch_size

if __name__ == '__main__':

    #gen = my_generator(52, train_dir)
    #videos, next_frame = next(gen)
    #print(videos[0].shape)
    #print(next_frame[0].shape)
    #print(videos.shape)

    #model.fit(train_images, train_labels, epochs=10, validation_data = (test_images, test_labels), callbacks=[cp_callback])
    #Loads the weights from the checkpoint path from the file cp.ckpt
    #model.load_weights(checkpoint_path)

    model = create_model()
    model.summary()
