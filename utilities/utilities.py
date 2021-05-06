import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob

dirs = '/home/wilfred/Datasets/UCF-101'

def process():
    
    dirs_process = sorted(glob.glob(dirs+'/*'))

    for sub in dirs_process:
        lst = glob.glob(sub+'/*')
        print(len(lst))


    return None

if __name__ == "__main__":

    process()
