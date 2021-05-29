import os
import shutil
import cv2
import random
import glob
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, figure

height = 480
width = 640

dirs_f1 = '/home/wilfred/Datasets/Motion/final_5/f1'
dirs_f2 = '/home/wilfred/Datasets/Motion/final_5/f2'
dirs_f3 = '/home/wilfred/Datasets/Motion/final_5/f3'
dirs_f4 = '/home/wilfred/Datasets/Motion/final_5/f4'
dest = '/home/wilfred/Datasets/Motion/final_processed_'
num = 6

def curate_f1():
    
    dirs = sorted(glob.glob(dirs_f1+'/*'))
    for s in dirs:
        files = sorted(glob.glob(s+'/frames/*'))
        
        for i in range(len(files)//num):
            images = files[i*num:i*num+num]
            dest_ = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]

            if not os.path.exists(dest_):
                os.mkdir(dest_)
                for img in images:
                    shutil.copy(img, dest_)
    return None

def curate_f2():

    dirs = sorted(glob.glob(dirs_f2+'/*'))
    for s in dirs:
        files = sorted(glob.glob(s+'/frames/*'))

        for i in range(len(files)//num):
            images = files[i*num:i*num+num]
            dest_ = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]

            if not os.path.exists(dest_):
                os.mkdir(dest_)
                for img in images:
                    im = cv2.imread(img)
                    im = cv2.resize(im, (640,480), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(dest_+'/'+img.split('/')[-1], im)
                    #shutil.copy(img, dest_)
    return None

def curate_f3():

    dirs = sorted(glob.glob(dirs_f3+'/*'))
    for s in dirs:
        files = sorted(glob.glob(s+'/frames/*'))

        for i in range(len(files)//num):
            images = files[i*num:i*num+num]
            dest_ = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]

            if not os.path.exists(dest_):
                os.mkdir(dest_)
                for img in images:
                    im = cv2.imread(img)
                    w_c = im.shape[1]//2 
                    h_c = im.shape[0]//2

                    im = im[h_c-240:h_c+240,w_c-320:w_c+320]
                    #im = cv2.resize(im, (640,480), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(dest_+'/'+img.split('/')[-1], im)
    return None

def curate_f4():

    dirs = sorted(glob.glob(dirs_f4+'/*'))
    for s in dirs:
        files = sorted(glob.glob(s+'/frames/*'))

        for i in range(len(files)//num):
            images = files[i*num:i*num+num]
            dest_ = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]

            if not os.path.exists(dest_):
                os.mkdir(dest_)
                for img in images:
                    im = cv2.imread(img)
                    w_c = im.shape[1]//2
                    h_c = im.shape[0]//2

                    im = im[h_c-240:h_c+240,w_c-320:w_c+320]
                    #im = cv2.resize(im, (640,480), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(dest_+'/'+img.split('/')[-1], im)
    return None

def compileImagefiles(source,dest):
    dirs = sorted(glob.glob(source+'/*'))
    for s in dirs:
        files = sorted(glob.glob(s+'/*'))
        for i in range(len(files)//6):
            images = files[i*6:i*6+6]
            dest_ = dest+'/'+s.split('/')[-1]+'_'+images[0].split('/')[-1].split('.')[-2]

            if not os.path.exists(dest_):
                os.mkdir(dest_)
                for img in images:
                    shutil.copy(img, dest_)
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

def datasetCreator():
    dir_='/home/wilfred/Datasets/Motion/VIRAT'
    lst = glob.glob(dir_+'/*')
    print(lst)
    for l in lst:
        st = str(os.path.splitext(l)[0].split('/')[-1])
        video = cv2.VideoCapture(l)
        success, image = video.read()
        count = 0    
        print(dir_+'/'+st)
        os.mkdir(dir_+'/'+st)
        while success:
            
            cv2.imwrite(dir_+'/'+st+'/frame%s.jpg'%str(count).zfill(5), image)
            success, image = video.read()
            #print('Frame read : ',success)
            count += 1
        
    return None

if __name__ == "__main__":

    curate_f1()
    curate_f2()
    curate_f3()
    curate_f4()
