import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Lambda, Reshape, Permute, Input, add, Conv3D, GaussianNoise, concatenate
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, TimeDistributed, Add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

import cv2
import random
import subprocess
import glob
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, figure, show

#define the hyper-parameters
num_epochs = 1000
batch_size = 32

#traffic dataset 3:4 image ratio
height = 480
width = 640

#train and validation directories
train_dir = '/home/wilfred/Datasets/Motion/final_processed/train'
val_dir = '/home/wilfred/Datasets/Motion/final_processed/test'
checkpoint_path = '/home/wilfred/Downloads/github/Python_Projects/videoPrediction/training/cp.ckpt'
saved_path = '/home/wilfred/Downloads/github/Python_Projects/videoPrediction/training'
#steps per epoch and validation steps
steps_per_epoch = len(glob.glob(train_dir + "/*"))//batch_size
validation_steps = len(glob.glob(val_dir + "/*"))//batch_size

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
def create_test_model():

    model = Sequential()
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same', input_shape=(height, width,4 * 1)))
    return model

def create_model():
    
    #increment the filters by a factor of 4
    c = 5
    
    #model beginning
    inp = Input((height, width, 4 * 1))
    reshaped = Reshape((height,width,4,1))(inp)
    permuted = Permute((1,2,4,3))(reshaped)
    noise = GaussianNoise(0.1)(permuted)
    last_layer = Lambda(slice, input_shape=(height,width,1,4), output_shape=(height,width,1))(noise)
    x = Permute((4,1,2,3))(noise)
    x =(ConvLSTM2D(filters=c, kernel_size=(3,3),padding='same',name='conv_lstm1', return_sequences=True))(x)
    c1=(BatchNormalization())(x)
    #x = Dropout(0.2)(x)
    x =(TimeDistributed(MaxPooling2D(pool_size=(2,2))))(c1)
    x =(ConvLSTM2D(filters=2*c,kernel_size=(3,3),padding='same',name='conv_lstm3',return_sequences=True))(x)
    c2=(BatchNormalization())(x)
    #x = Dropout(0.2)(x)
    x =(TimeDistributed(MaxPooling2D(pool_size=(2,2))))(c2)
    x =(ConvLSTM2D(filters=4*c,kernel_size=(3,3),padding='same',name='conv_lstm4',return_sequences=True))(x)
    x =(TimeDistributed(UpSampling2D(size=(2, 2))))(x)
    x =(ConvLSTM2D(filters=4*c,kernel_size=(3,3),padding='same',name='conv_lstm5',return_sequences=True))(x)
    x =(BatchNormalization())(x)
    x =(ConvLSTM2D(filters=2*c,kernel_size=(3,3),padding='same',name='conv_lstm6',return_sequences=True))(x)
    x =(BatchNormalization())(x)
    x = Add()([c2, x])
    #x = Dropout(0.2)(x)
    x =(TimeDistributed(UpSampling2D(size=(2, 2))))(x)
    x =(ConvLSTM2D(filters=c,kernel_size=(3,3),padding='same',name='conv_lstm7',return_sequences=False))(x)
    x =(BatchNormalization())(x)
    combined = concatenate([last_layer, x])
    combined = Conv2D(1, (1,1))(combined)
    model=Model(inputs=[inp], outputs=[combined])
    return model

#the generator creates batches of images during training for catz dataset
def my_gene(batch_size, img_dir):
    dirs = glob.glob(img_dir + '/*')
    counter = 0
    while True:
        input_images = np.zeros((batch_size, width, height, 1*4))
        output_images = np.zeros((batch_size, width, height, 1))
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

def my_generator(batch, img_dir):
    dirs = glob.glob(img_dir + '/*')
    counter = 0
    while True:
        input_images = np.zeros((batch, width, height, 1*4))
        output_images = np.zeros((batch, width, height, 1))
        random.shuffle(dirs)
        if (counter+batch >= len(dirs)):
            counter = 0 
        for i in range(batch):
            input_imgs = sorted(glob.glob(dirs[counter + i] + '/*'))
            imgs = []
            for j in range(len(input_imgs)-1):
                imgs.append(cv2.imread(input_imgs[j],0).reshape(width, height, 1))
            input_images[i] = np.concatenate(imgs,axis=2)
            output_images[i] = cv2.imread(input_imgs[4],0).reshape(width, height, 1)
            
            input_images[i] /= 255.
            output_images[i] /= 255.

        yield(input_images, output_images)
        counter += batch
        

if __name__ == "__main__":

    model = create_model()
    #model = create_test_model()
    model.summary()

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
    #model.compile(optimizer='adam', loss=ssim_loss, metrics=[perceptual_distance])
    model.compile(optimizer='adam', loss=ssim_loss)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

    history = model.fit(my_generator(batch_size,train_dir), steps_per_epoch=steps_per_epoch//4, epochs= num_epochs, validation_data = my_generator(batch_size, val_dir), callbacks=[es_callback, cp_callback], verbose = 1)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(saved_path+'/model-history.csv')
    model.save(saved_path+'/model.h5')

    
    print('End of training...')
