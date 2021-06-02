import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


#define the hyper-parameters
num_epochs = 1000
batch_size = 32


height  = 120
width = 120

#train and validation directories
train_dir = '/home/wilfred/Datasets/Motion/final_processed/train'
val_dir = '/home/wilfred/Datasets/Motion/final_processed/test'
checkpoint_path = '/home/wilfred/Downloads/github/Python_Projects/videoPrediction/training/cp.ckpt'
saved_path = '/home/wilfred/Downloads/github/Python_Projects/videoPrediction/training'
#steps per epoch and validation steps
steps_per_epoch = len(glob.glob(train_dir + "/*"))//batch_size
validation_steps = len(glob.glob(val_dir + "/*"))//batch_size

def psnr_mean(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim_loss(true, pred):
    return 1 - tf.reduce_mean(tf.image.ssim(true, pred, 1.0))

def perceptual_distance(y_true, y_pred):
    y_true *= 255.
    y_pred *= 255.
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

def model_ae():
    input_img = tf.keras.layers.Input(shape=(120, 120, 4))
    reshaped = Reshape((height,width,4,1))(input_img)
    permuted = Permute((1,2,4,3))(reshaped)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = tf.keras.models.Model(input_img, decoded)
    return model    

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

    model = model_ae()
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

