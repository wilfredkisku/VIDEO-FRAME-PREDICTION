import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob

dirs = '/home/wilfred/Datasets/UCF-101'
dirs_video = '/home/wilfred/Downloads/github/Python_Projects/videoPrediction/data/coastguard_cif.mp4'
dirs_images = '/home/wilfred/Downloads/github/Python_Projects/videoPrediction/data/coastguard'

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
    convert(dirs)
