import numpy as np    
from numpy.lib.format import open_memmap      
from skimage.io import imshow                     
from skimage.measure import label, regionprops, marching_cubes
from skimage.morphology import erosion, dilation, ball    
from cv2 import imread, VideoWriter, VideoWriter_fourcc
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from stl import mesh                               
import napari
import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D   
 

def OS_path(exp, OS):
    if OS=='Windows': return 'Z:/rot_datasets/' + exp
    elif OS=='MacOS': return '/Volumes/T7/Thesis/' + exp
    elif OS=='Linux': return '/data/projects/whaitiri/Data/Data_Processing_July2022/rot_datasets/' + exp
    else: raise ValueError('OS not recognized')

# Missing features: ct resizing, df construction, plots

class CT_slice():
    
    def __init__(self, array):
        self._np = np.copy(array)
        self._PIL = Image.fromarray(self._np)

    def _PIL_conversion(self):
        self._PIL = Image.fromarray(self._np)
        return

    def scale(self, img_min, img_max):
        self._np[self._np > img_max] = img_max
        self._np[self._np < img_min] = img_min
        self._np = 255 * (self._np - img_min) / (img_max - img_min)
        self._np = self._np.astype(np.uint8)
        self._PIL_conversion()
        return

    def show(self):
        _ = imshow(self._np, cmap='gray')
        return

    def add_time_text(self, time):
        draw = ImageDraw.Draw(self._PIL)
        image_size = self._np.shape[0]
        draw.text(xy=(image_size - 120, image_size - 30),
                  text=f'Time = {time * 50} ms', 
                  font=ImageFont.truetype('Roboto-Regular.ttf', 15),
                  fill='#FFFFFF')
        self._np = np.array(self._PIL, dtype=np.uint8)
        return
        
    def save(self, path):
        _ = self._PIL.save(os.path.join(path, str(self._time).zfill(3)+'.png'))
        return


class Movie:

    def __init__(self, path, img_path, exp):
        self._path = path
        self._img_path = img_path
        self._exp = exp
        sample = imread(os.path.join(self._img_path, 
                                     [f for f in os.listdir(self._img_path) if f.endswith('.png')][0]))
        self._height, self._width, _ = sample.shape

    def write(self, fps):
        self._fps = fps
        frame_files = sorted([f for f in os.listdir(self._img_path) if f.endswith('.png')])
        fourcc = VideoWriter_fourcc(*'mp4v')
        self._video = VideoWriter(os.path.join(self._path, self._exp + '.mp4'), 
                                  fourcc, self._fps, (self._width, self._height))
        for frame_file in frame_files:
            frame_path = os.path.join(self._img_path, frame_file)
            frame = imread(frame_path)
            self._video.write(frame)
        self._video.release()


class CT_data:

    def __init__(self, exp, OS='MacOS'):
        self._path = OS_path(exp, OS)
        self._ct = open_memmap(os.path.join(self._path, 'ct.npy'), mode='r')  # 4D CT-scan
        if os.path.exists(os.path.join(self._path, 'mask.npy')):
            self._mask = open_memmap(os.path.join(self._path, 'mask.npy'),
                                     dtype=np.ushort, mode='r+', shape=self._ct.shape)  # 4D CT-scan segmentation map
        else:
            self._mask = None
        if os.path.exists(os.path.join(self._path, 'binary_mask.npy')):
            self._binary_mask = open_memmap(os.path.join(self._path, 'binary_mask.npy'),
                                     dtype=np.ushort, mode='r+', shape=self._ct.shape)  # 4D CT-scan binary mask for sidewall rupture rendering
        else:
            self._binary_mask = None
        self._exp = exp  # Experiment name
        self._index = 0  # Integer value representing the index of the current time instant: used to determine which volume has to be segmented
        self._threshold = 0  # Value used for thresholding (this value is obtained by _find_threshold method
        return

    def _find_biggest_area(self, smaller_ct, SLICES):
        mask = np.greater(smaller_ct, self._threshold)
        mask = label(np.logical_and(mask, dilation(erosion(mask, ball(1)), ball(4))))
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
        mask = label(np.logical_and(mask, dilation(erosion(mask, ball(1)), ball(4))))
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
        new_labels = np.setdiff1d([rp.label for rp in regionprops(current_mask)],
                                  [*self._label_propagation_map])  # current mask labels that were not propagated from previous mask
        lookup_table[new_labels] = np.arange(len(new_labels)) + previous_mask_max_label + 1
        self._mask[self._index] = np.take(lookup_table, current_mask)
        return
    
    def _compute_regionprops(self):
        self._rps = [regionprops(self._mask[time]) for time in range(self._mask.shape[0])]
        self._labels_to_remove = set()
        return

    def _find_isolated_agglomerates(self):
        previous_labels = current_labels = []
        next_labels = [rp.label for rp in self._rps[0]]
        for next_rps in self._rps[1:]:
            previous_labels = current_labels
            current_labels = next_labels
            next_labels = [rp.label for rp in next_rps]
            for current_label in current_labels:
                if current_label not in previous_labels and current_label not in next_labels:
                    self._labels_to_remove.add(current_label)
        return
    
    def _find_small_4d_agglomerates(self, smallest_4Dvolume):
        volumes = dict()
        for rps in self._rps:
            for rp in rps:
                if rp.label not in volumes:
                    volumes[rp.label] = rp.area
                else:
                    volumes[rp.label] += rp.area
        for label, area in volumes.items():
            if area < smallest_4Dvolume:
                self._labels_to_remove.add(label)
        return
    
    def _find_pre_TR_agglomerates(self):
        TR_map = {'P28A_FT_H_Exp1':2, 'P28A_FT_H_Exp2':2, 'P28A_FT_H_Exp3_3':2, 'P28A_FT_H_Exp4_2':6,
                  'P28B_ISC_FT_H_Exp2':11, 'VCT5_FT_N_Exp1':4,'VCT5_FT_N_Exp3':5, 'VCT5_FT_N_Exp4':4,
                  'VCT5_FT_N_Exp5':4, 'VCT5A_FT_H_Exp2':1, 'VCT5A_FT_H_Exp5':4}
        for rps in self._rps[:TR_map[self._exp]]:
            for rp in rps:
                if rp.label != 1:
                    self._labels_to_remove.add(rp.label)
        return
    
    def _update_labels(self):
        all_labels = set()
        for rps in self._rps:
            for rp in rps:
                all_labels.add(rp.label)
        filtered_labels = all_labels - self._labels_to_remove
        lookup_table = np.zeros(max(all_labels) + 1, dtype=np.ushort)
        lookup_table[sorted(list(filtered_labels))] = np.arange(len(filtered_labels)) + 1
        for time in range(self._ct.shape[0]):
            self._mask[time] = np.take(lookup_table, self._mask[time])
        return
    
    def _filtering(self, smallest_4Dvolume, pre_TR_filtering):
        progress_bar = tqdm(total=5, desc=f'{self._exp}', leave=False)
        progress_bar.set_postfix_str('Regionprops computation')
        self._compute_regionprops()
        progress_bar.update() #1
        progress_bar.set_postfix_str('Isolated agglomerates removal')
        self._find_isolated_agglomerates()
        progress_bar.update() #2
        progress_bar.set_postfix_str('Small agglomerates removal')
        self._find_small_4d_agglomerates(smallest_4Dvolume)
        progress_bar.update() #3
        progress_bar.set_postfix_str('Pre-TR agglomerates removal')
        if pre_TR_filtering:
            self._find_pre_TR_agglomerates()
        progress_bar.update() #4
        progress_bar.set_postfix_str('Label removal/renaming')
        self._update_labels()
        progress_bar.close() #5
        return

    def segment(self, threshold_target=6800, filtering3D=True, filtering4D=True, pre_TR_filtering=True,
                smallest_3Dvolume=50, smallest_4Dvolume=250):
        if self._mask is None:
            self._mask = open_memmap(os.path.join(self._path, 'mask.npy'),
                                     dtype=np.ushort, mode='w+', shape=self._ct.shape)  # 4D CT-scan segmentation map
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
        progress_bar.close()
        if filtering4D:
            self._filtering(smallest_4Dvolume, pre_TR_filtering)
        return
    
    def binary_mask(self, threshold=1, smallest_3Dvolume=50):
        if self._binary_mask is None:
            self._binary_mask = open_memmap(os.path.join(self._path, 'binary_mask.npy'),
                                     dtype=np.ushort, mode='w+', shape=self._ct.shape)  # 4D CT-scan segmentation map
        self._smallest_3Dvolume = smallest_3Dvolume
        progress_bar = tqdm(range(self._ct.shape[0]), desc=f'{self._exp}', leave=False)
        progress_bar.set_postfix_str(f'Binary masking: threshold = {threshold}')
        for time in progress_bar:
            mask = np.greater(self._ct[time], threshold)
            self._binary_mask[time] = np.logical_and(mask, dilation(erosion(mask, ball(1)), ball(3)))
        return

    def view(self, mask=False, binary_mask=False):
        viewer = napari.Viewer()
        settings = napari.settings.get_settings()
        settings.application.playback_fps = 5
        viewer.add_image(self._ct, name=f'{self._exp} Volume', contrast_limits = [0, 6])
        if binary_mask and not self._binary_mask is None:
            viewer.add_labels(self._binary_mask, name=f'{self._exp} Binary mask', opacity=0.8)
        if mask and not self._mask is None:
            viewer.layers[f'{self._exp} Volume'].opacity = 0.4
            viewer.add_labels(self._mask, name=f'{self._exp} Mask', opacity=0.8)
        viewer.dims.current_step = (0, 0)
        return

    def slice_movie(self, z, img_min=0, img_max=5, fps=7):
        movie_path = os.path.join(self._path, 'movies', f'slice {z} movie')
        img_path = os.path.join(movie_path, 'frames')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        for time in range(self._ct.shape[0]):
            image = CT_slice(self._ct[time, z])
            image.scale(img_min, img_max)
            image.add_time_text(time)
            image.save(img_path)
        movie = Movie(movie_path, img_path, self._exp)
        movie.write(fps)
        return

    def render_movie(self, fps=5):
        movie_path = os.path.join(self._path, 'renders')
        for view in ['top view', 'side view', 'perspective view']:
            movie = Movie(movie_path, os.path.join(movie_path, view), self._exp)
            movie.write(fps)
            print(f'{self._exp} {view} done!')
        return
    
    def _find_mesh(self, time, is_binary_mask):
        s = self._mask[time].shape
        mask = np.zeros((s[2], s[1], s[0]+2), dtype=np.ushort)
        mask[:, :, 1:-1] = np.swapaxes(self._binary_mask[time], 0, 2) if is_binary_mask else np.swapaxes(self._mask[time], 0, 2)
        verts, faces, _, values = marching_cubes(mask, 0)
        verts = (0.004 * verts * np.array([1, 1, -1])) + np.array([-1, -1, 0.5])
        values = values.astype(np.ushort)
        return verts, faces, values
    
    def _save_stl(self, label, verts, faces, values, time_path, is_binary_mask):
        label_faces = faces[np.where(values[faces[:,0]] == label)]
        stl_mesh = mesh.Mesh(np.zeros(label_faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(label_faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = verts[face[j]]
        name = str(0).zfill(5) if is_binary_mask else str(label).zfill(5)
        stl_mesh.save(os.path.join(time_path, name + '.stl'))
        return
    
    def _create_agglomerate_stls(self, stl_path, is_sidewall_rupture, times):
        if self._mask is None:
            raise NameError('Mask not found, run CT_data.segment() first!')
        iterator = range(self._mask.shape[0]) if times is None else times
        for time in tqdm(iterator, desc='Creating stl files'):
            time_path = os.path.join(stl_path, str(time).zfill(3))
            if not os.path.exists(time_path):
                os.makedirs(time_path)
            verts, faces, values = self._find_mesh(time, is_binary_mask=False)
            for label in [label for label in np.unique(values) if label == 1 or not is_sidewall_rupture]:
                self._save_stl(label, verts, faces, values, time_path, is_binary_mask=False)
        return
    
    def _create_sidewall_stls(self, stl_path, times):
        if self._binary_mask is None:
            raise NameError('Binary mask not found, run CT_data.binary_mask() first!')
        iterator = range(self._mask.shape[0]) if times is None else times
        for time in tqdm(iterator, desc='Creating binary stl files'):
            time_path = os.path.join(stl_path, str(time).zfill(3))
            if not os.path.exists(time_path):
                os.makedirs(time_path)
            verts, faces, values = self._find_mesh(time, is_binary_mask=True)
            self._save_stl(1, verts, faces, values, time_path, is_binary_mask=True)
        return
    
    def create_stls(self, is_sidewall_rupture=False, times=None):
        stl_path = os.path.join(self._path, 'stls') if not is_sidewall_rupture else os.path.join(self._path, 'sidewall_stls')
        self._create_agglomerate_stls(stl_path, is_sidewall_rupture, times)
        if is_sidewall_rupture:
            self._create_sidewall_stls(stl_path, times)
        return

    