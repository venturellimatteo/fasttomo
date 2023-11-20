from tqdm import tqdm
from numpy.lib.format import open_memmap
from PIL import Image as im
import myfunctions as mf
import numpy as np
import os

OS = 'MacOS_SSD'

for exp in tqdm(mf.bad_exp_list()):
    exp_dir = mf.OS_path(exp, OS)
    hypervolume = open_memmap(os.path.join(mf.OS_path(exp, OS), 'hypervolume.npy'), mode='r')
    img_folder = os.path.join(exp_dir, 'img')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    for t in range(hypervolume.shape[0]):
        img = np.copy(hypervolume[t, 90, :, :])
        img[img <= 0.15] = 0.15
        img[img >= 2.5] = 2.5
        img = (img - 0.15) / (2.5 - 0.15)
        img = (img * 255).astype(np.uint8)
        img = im.fromarray(img)
        img.save(os.path.join(img_folder, str(t).zfill(3) + '.png'))