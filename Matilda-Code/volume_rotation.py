import numpy as np
import PIL
from PIL import Image
import os
import pylab as plt
import scipy.ndimage
from scipy.ndimage import median_filter
import skimage
import math
from scipy.ndimage.filters import sobel
import fabio.tifimage as tif

def find_rotation(image):
    img = Image.open(image)
    img = np.array(image)
    center = center_rotation(img, 200, 200)





if __name__ == '__main__':
    path = 'U:\\whaitiri\\Data\\Data_Processing_July_2022\\Reconstructions\\'
    out_path = 'U:\\whaitiri\\Data\\Data_Processing_July_2022\\Polished_datasets\\'


    for dataset in os.listdir(path):
        if ('P28A_ISC_FT_H_Exp5'in dataset):
            data_path = path + dataset + '\\'
            path_out = out_path + dataset +'\\'
            if not os.path.isdir(path_out):
                os.mkdir(path_out)
            for timestamp in os.listdir(data_path):
                time_path = data_path + timestamp + '\\'
                path_out = path_out + timestamp + '\\'
                if not os.path.isdir(path_out):
                    os.mkdir(path_out)
                list_images = []
                for i, image in enumerate(os.listdir(time_path)):
                    im_path = time_path + image
                    list_images.append(im_path)
                print('Finding rotation angle')
                rotation_angle, center = find_rotation(list_images[50])
                print('Rotating all images')
                rotate_all(list_images, rotation_angle, center, path_out)