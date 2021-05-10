import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, UpSampling2D
from keras import backend as K
import cv2
import random
import glob
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, figure

height_ucf = 320
width_ucf = 480

train_dir_ucf = '/home/wilfred/Datasets/UCF-101/Traffic'

#steps_per_epoch = len(glob.glob(train_dir + "/*")) // batch_size
#validation_steps = len(glob.glob(val_dir + "/*")) // batch_size

#dirs = '/home/wilfred/Datasets/UCF-101'
#dirs_video = '/home/wilfred/Downloads/github/Python_Projects/videoPrediction/data/coastguard_cif.mp4'
#dirs_images = '/home/wilfred/Downloads/github/Python_Projects/videoPrediction/data/coastguard'

def my_generator(batch_size, img_dir):

    dirs = sorted(glob.glob(img_dir+'/*'))
    #for s in dirs:
        #print(len(glob.glob(s+'/*')))
    counter = 0
    while True:
        input_images = np.zeros((batch_size, width, height, 3*5))
        output_images = np.zeros((batch_size, width, height, 3))
        #random.shuffle(dirs)
        #if (counter+batch_size >= len(dirs)):
            #counter = 0
        #for i in range(batch_size):
            #input_imgs = glob.glob(dirs[counter + i] + '/cat_[0-5]*')
            #imgs = [Image.open(img) for img in sorted(input_imgs)]
            #input_images[i] = np.concatenate(imgs, axis = 2)
            #output_images[i] = np.array(Image.open(dirs[counter + i] + '/cat_result.jpg'))
            #input_images[i] /= 255.
            #output_images[i] /= 255.

        #yield (input_images, output_images)
        #counter += batch_size

def compileImagefiles(source,dest):
    dirs = sorted(glob.glob(source+'/*'))
    for s in dirs:
        files = sorted(glob.glob(s+'/*'))
        for i in range(len(files) - 6 + 1):
            images = files[i:i+6]
            dest = dest+'/'+s.split('/')[-1]+'_'+images[0]
            if not os.path.exists(dest):
                os.mkdir(path)
            shutil.move(images, dest+)

    return None

def convert(path):
    list_categories = sorted(glob.glob(path+'/*'))
    
    for l in list_categories:
        
        movies = glob.glob(l+'/*')
        
        for lf in movies:
            if not os.path.exists(lf[:-4]):
                os.makedirs(lf[:-4])
                extract(lf)
    return None

def resizeImage(path):

    files = sorted(glob.glob(path+'/*'))
    
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img[:,:288,:], (96,96))
        fname = os.path.splitext(f)[0][-10:]
        print(fname)
        cv2.imwrite(path+'/'+fname+'_scaled.jpg', img)
    return None

def extract(path):

    video = cv2.VideoCapture(path)
    success, image = video.read()
    count = 0

    while success:
        cv2.imwrite(path[:-4]+'/'+'frame%s.jpg'%str(count).zfill(5), image)
        success, image = video.read()
        print('Frame read : ',success)
        count += 1

    return None

def process():
    
    dirs_process = sorted(glob.glob(dirs+'/*'))

    for sub in dirs_process:
        lst = glob.glob(sub+'/*')
        print(len(lst))


    return None

if __name__ == "__main__":

    #process()
    #extract(dirs_video)
    #resizeImage(dirs_images)
    #convert(dirs)

    my_generator(32, train_dir_ucf)
    #videos, next_frame = next(gen)
    #print(videos[0].shape)
    #print(next_frame[0].shape)

    #print(videos.shape)

