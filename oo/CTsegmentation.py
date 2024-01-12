import numpy as np    
from numpy.lib.format import open_memmap                           
from skimage.measure import label, regionprops
from skimage.morphology import erosion, dilation, ball    
from tqdm import tqdm                               
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   
import napari
import os


def OS_path(exp, OS):
    if OS=='Windows': return 'Z:/rot_datasets/' + exp
    elif OS=='MacOS': return '/Volumes/T7/Thesis/' + exp
    elif OS=='Linux': return '/data/projects/whaitiri/Data/Data_Processing_July2022/rot_datasets/' + exp
    else: raise ValueError('OS not recognized')

# Missing features: 4d filtering, df construction, plots

class CT_data:

    def __init__(self, exp, OS='MacOS'):
        parent_dir = OS_path(exp, OS)
        self._ct = open_memmap(os.path.join(parent_dir, 'hypervolume.npy'), mode='r')  # 4D CT-scan
        mode = 'r+' if os.path.exists(os.path.join(parent_dir, 'hypervolume_mask.npy')) else 'w+'
        self._mask = open_memmap(os.path.join(parent_dir, 'hypervolume_mask.npy'), dtype=np.ushort, mode=mode, shape=self._ct.shape)  # 4D CT-scan segmentation map
        self._exp = exp  # Experiment name
        self._index = 0  # Integer value representing the index of the current time instant: used to determine which volume has to be segmented
        self._threshold = 0  # Value used for thresholding (this value is obtained by _find_threshold method

    def _find_biggest_area(self, smaller_ct, SLICES):
        mask = np.greater(smaller_ct, self._threshold)
        mask = label(np.logical_and(mask, dilation(erosion(mask, ball(2)), ball(4))))
        areas = [rp.area for rp in regionprops(mask)]
        return np.max(areas)/SLICES

    def _find_threshold(self):
        DELTA = 100; SLICES = 10; step = 1; flag=False; add=True
        smaller_ct = np.array([self._ct[0,i] for i in np.linspace(0, self._ct.shape[1]-1, SLICES, dtype=int)])
        while not flag:
            current_area = self._find_biggest_area(smaller_ct, SLICES)
            if current_area > self._threshold_target + DELTA:  # if the area is larger than target, the threshold is increased in order to reduce the area
                step = step/2 if not add else step  # step is halved every time the direction of the step is changed
                self._threshold += step
                add = True
            elif current_area < self._threshold_target - DELTA:  # if the area is smaller than target, the threshold is decreased in order to increase the area
                step = step/2 if add else step              
                self._threshold -= step
                add = False
            else:  # if the area is close to target, the threshold is found
                flag = True
        return

    def _remove_small_3d_agglomerates(self, mask):
        bincount = np.bincount(mask.flatten())
        lookup_table = np.where(bincount < self._smallest_3Dvolume, 0, np.arange(len(bincount)))
        return np.take(lookup_table, mask)

    def _segment3d(self):
        mask = np.greater(self._ct[self._index], self._threshold)
        mask = label(np.logical_and(mask, dilation(erosion(mask, ball(2)), ball(4))))
        if self._filtering3D:
            mask = self._remove_small_3d_agglomerates(mask)
        if self._index == 0:
            rps = regionprops(mask)
            current_labels = np.array([rp.label for rp in rps])[np.argsort([rp.area for rp in rps])][::-1]
            lookup_table = np.zeros(np.max(current_labels)+1, dtype=np.ushort)
            lookup_table[current_labels] = np.arange(len(current_labels)) + 1
            mask = np.take(lookup_table, mask)
        self._mask[self._index] = mask
        return
    
    def _dictionary_update_condition(self, current_mask_label, count):
        if (current_mask_label > 0 and 
            (current_mask_label not in self._label_propagation_map or
             count >= self._label_propagation_map[current_mask_label][1])):
            return True
        return False

    def _create_label_propagation_map(self):
        self._label_propagation_map = dict()  # Dictionary used for label propagation 
        previous_mask = self._mask[self._index - 1]
        current_mask = self._mask[self._index]
        rps = regionprops(previous_mask)
        ordered_previous_mask_labels = np.array([rp.label for rp in rps])[np.argsort([rp.area for rp in rps])]
        for previous_mask_label in ordered_previous_mask_labels:
            current_mask_overlapping_labels = current_mask[previous_mask == previous_mask_label]
            if np.any(current_mask_overlapping_labels):
                current_mask_overlapping_labels, counts = np.unique(current_mask_overlapping_labels, return_counts=True)
                for current_mask_label, count in zip(current_mask_overlapping_labels, counts):
                    if self._dictionary_update_condition(current_mask_label, count):
                        self._label_propagation_map[current_mask_label] = np.array([previous_mask_label, count])
        return

    def _propagate(self):
        previous_mask_max_label = np.max(self._mask[self._index - 1])
        current_mask = self._mask[self._index]
        current_mask[current_mask > 0] += previous_mask_max_label + 1
        self._create_label_propagation_map()
        lookup_table = np.zeros(np.max(current_mask)+1, dtype=np.ushort)
        for current_mask_label, [previous_mask_label, _] in self._label_propagation_map.items():
            lookup_table[current_mask_label] = previous_mask_label
        new_labels = np.setdiff1d([rp.label for rp in regionprops(current_mask)], [*self._label_propagation_map])  # current mask labels that were not propagated from previous mask
        lookup_table[new_labels] = np.arange(len(new_labels)) + previous_mask_max_label + 1
        self._mask[self._index] = np.take(lookup_table, current_mask)
        return

    def segment(self, threshold_target=6800, filtering3D=True, smallest_3Dvolume=50):
        self._threshold_target = threshold_target  # Target area (in pixels) of the external shell of the battery
        self._filtering3D = filtering3D  # Boolean variable: if True, small agglomerate filtering is computed
        self._smallest_3Dvolume = smallest_3Dvolume  # Lower bound for the agglomerate volume
        progress_bar = tqdm(total=self._ct.shape[0], desc=f'{self._exp}', leave=False)
        progress_bar.set_postfix_str(f'Evaluating threshold')
        self._find_threshold()
        progress_bar.set_postfix_str(f'Segmentation #{self._index + 1}')
        self._segment3d()
        progress_bar.update()
        self._index += 1
        while self._index < self._ct.shape[0]:
            progress_bar.set_postfix_str(f'Segmentation #{self._index + 1}')
            self._segment3d()
            progress_bar.set_postfix_str(f'Propagation #{self._index + 1}')
            self._propagate()
            progress_bar.update()
            self._index += 1
        return

    def view(self, mask=True):
        viewer = napari.Viewer()
        images = viewer.add_image(self._ct, name='Volume')
        if mask:
            viewer.layers['Volume'].opacity = 0.4
            labels = viewer.add_labels(self._mask, name='Mask', opacity=0.8)
        settings = napari.settings.get_settings()
        settings.application.playback_fps = 5
        viewer.dims.current_step = (0, 0)
