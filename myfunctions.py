import numpy as np                                  # type: ignore
from skimage.measure import label, regionprops      # type: ignore
from skimage.io import imread                       # type: ignore
from dataclasses import dataclass
from tqdm import tqdm                               # type: ignore
import shutil
import os
import time as time_lib



@dataclass
class feature:
    values: np.ndarray                      # values mediated by the different samples considered
    mean: float                             # mean of all the values
    std: float                              # standard deviation of all the values

@dataclass
class experiment:
    name: str                               # name of the experiment
    volume_number: feature                  # number of agglomerates contained in each slice of a volume
    volume_area: feature                    # area of the agglomerates contained in each slice of a volume
    slice_number: feature                   # number of agglomerates contained in a fixed slice at different time instants
    slice_area: feature                     # area of the agglomerates contained in a fixed slice at different time instants
    slice_stability_time: feature           # time needed to see stabilization in the number of agglomerates contained in a fixed slice

def exp_list():
    return ['P28A_FT_H_Exp1', 'P28A_FT_H_Exp2', 'P28A_FT_H_Exp3_3',  
            'P28A_FT_H_Exp4_2', 'P28A_FT_H_Exp5_2',  
            'P28A_FT_N_Exp1', 'P28A_FT_N_Exp4', 'P28B_ICS_FT_H_Exp5', 'P28B_ICS_FT_H_Exp2', 
            'P28B_ICS_FT_H_Exp3', 'P28B_ICS_FT_H_Exp4', 'P28B_ICS_FT_H_Exp4_2', 'VCT5_FT_N_Exp1', 
            'VCT5_FT_N_Exp3', 'VCT5_FT_N_Exp4', 'VCT5_FT_N_Exp5', 'VCT5A_FT_H_Exp1',
            'VCT5A_FT_H_Exp2', 'VCT5A_FT_H_Exp3', 'VCT5A_FT_H_Exp4', 'VCT5A_FT_H_Exp5']

def exp_start_time():
    return [112, 99, 90, 90, 108, 127, 130, 114, 99, 105, 104, 115, 155, 70, 54, 7, 71, 52, 4, 66, 66]

def rotate180(image):
    return np.rot90(np.rot90(image))

def mask(image, threshold):
    return np.vectorize(label, signature='(n,m)->(n,m)')(image > threshold)

def remove_small_agglomerates(sequence_mask, smallest_volume):
    bincount = np.bincount(sequence_mask.flatten())
    sequence_mask[np.isin(sequence_mask, np.where(bincount < smallest_volume))] = 0
    return sequence_mask

def find_biggest_area(sequence, threshold):
    sequence_mask = np.zeros_like(sequence).astype(int)
    for i in range(sequence.shape[0]):
        sequence_mask[i,:,:] = mask(sequence[i,:,:], threshold)
    unique_labels, label_counts = np.unique(sequence_mask, return_counts=True)
    label_counts[label_counts == np.max(label_counts)] = 0
    return np.max(label_counts)/sequence.shape[0]




def image_path(exp, time, slice, isSrc=True, dst='', win=False):

    folder_name = 'entry' + str(time).zfill(4) + '_no_extpag_db0100_vol'
    image_name = 'entry' + str(time).zfill(4) + '_no_extpag_db0100_vol_' + str(slice).zfill(6) + '.tiff'

    if isSrc:
        if win:
            path = 'Z:/Reconstructions/' + exp
        else:
            path = '../MasterThesisData/' + exp
    else:
        path = os.path.join(dst, exp)
        if not os.path.exists(os.path.join(path, folder_name)):
            os.makedirs(os.path.join(path, folder_name))

    return os.path.join(path, folder_name, image_name)



def move_sequence(exp, first_slice, last_slice, start_time, end_time, dst, win=True):
    for time in range(start_time, end_time+1):
        for slice in range(first_slice, last_slice+1):
            src_dir = image_path(exp, time, slice, isSrc=True, win=win)
            dst_dir = image_path(exp, time, slice, isSrc=False, dst=dst, win=win)
            shutil.copyfile(src_dir, dst_dir)



def read_3Dsequence(exp, time=0, slice=0, start_time=0, end_time=220, first_slice=20, last_slice=260, volume=True, win=False):

    if volume:
        test_image = imread(image_path(exp, time, first_slice, win=win))
        sequence = np.zeros((last_slice-first_slice, test_image.shape[0], test_image.shape[1]))
        for slice in range(first_slice, last_slice):
            image = imread(image_path(exp, time, slice, win=win))
            sequence[slice-first_slice,:,:] = image
    else:
        test_image = imread(image_path(exp, start_time, slice, win))
        sequence = np.zeros((end_time-start_time, test_image.shape[0], test_image.shape[1]))
        for time in range(start_time, end_time):
            image = imread(image_path(exp, time, slice, win=win))
            sequence[time-start_time,:,:] = image

    return sequence



def read_4Dsequence(exp, first_slice, last_slice, end_time=220, win=False):

    print(f'Collecting sequence for experiment {exp}...')
    tic = time_lib.time()
    start_time = exp_start_time()[exp_list().index(exp)]
    test_image = imread(image_path(exp, start_time, first_slice, win=win))
    time_steps = np.arange(start_time, end_time, 2, dtype=int)
    sequence = np.zeros((len(time_steps), last_slice-first_slice, test_image.shape[0], test_image.shape[1]))
    for t, time in enumerate(time_steps):
        for slice in range(first_slice, last_slice):
            image = imread(image_path(exp, time, slice, win=win))
            # if time%2 == 0:
            #     image = rotate180(image)
            sequence[t, slice-first_slice,:,:] = image
    toc = time_lib.time()
    print(f'Sequence collected in {toc-tic:.2f} s\n')
    return sequence



def propagate_labels(previous_mask, current_mask, forward=True):
    if forward:
        current_mask[current_mask > 0] = current_mask[current_mask > 0] + np.max(previous_mask)
    unique_labels, label_counts = np.unique(previous_mask, return_counts=True)
    ordered_labels = unique_labels[np.argsort(label_counts)]
    for previous_slice_label in ordered_labels:
        bincount = np.bincount(current_mask[previous_mask == previous_slice_label])
        if len(bincount) <= 1:
            continue
        bincount[0] = 0
        current_slice_label = np.argmax(bincount)
        current_mask[current_mask == current_slice_label] = previous_slice_label
    return current_mask



def segment3D(sequence, threshold, smallest_volume=100, filtering=True):

    sequence_mask = np.zeros_like(sequence).astype(int)
    sequence_mask[0,:,:] = mask(sequence[0,:,:], threshold)
    # masking of current slice and forward propagation from the first slice
    for i in range(1, sequence.shape[0]):
        sequence_mask[i,:,:] = mask(sequence[i,:,:], threshold)
        sequence_mask[i,:,:] = propagate_labels(sequence_mask[i-1,:,:], sequence_mask[i,:,:], forward=True)
    # backward propagation from the last slice
    for i in range(sequence_mask.shape[0]-1, 0, -1):
        sequence_mask[i-1,:,:] = propagate_labels(sequence_mask[i,:,:], sequence_mask[i-1,:,:], forward=False)
    # removal of the agglomerates with volume smaller than smallest_volume
    if filtering:
        sequence_mask = remove_small_agglomerates(sequence_mask, smallest_volume)
    return sequence_mask



def segment4D(sequence, threshold, smallest_3Dvolume=50, smallest_4Dvolume=200, filtering3D=False, filtering4D=True):
    sequence_mask = np.zeros_like(sequence).astype(int)
    sequence_mask[0,:,:,:] = segment3D(sequence[0,:,:,:], threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
    # masking of current volume and forward propagation from the first volume
    for t in tqdm(range(1, sequence.shape[0]), desc='Volume segmentation and forward propagation'):
        sequence_mask[t,:,:,:] = segment3D(sequence[t,:,:,:], threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
        sequence_mask[t,:,:,:] = propagate_labels(sequence_mask[t-1,:,:,:], sequence_mask[t,:,:,:], forward=True)
    # backward propagation from the last volume
    for t in tqdm(range(sequence_mask.shape[0]-1, 0, -1), desc='Backward propagation'):
        sequence_mask[t-1,:,:,:] = propagate_labels(sequence_mask[t,:,:,:], sequence_mask[t-1,:,:,:], forward=False)
    # removal of the agglomerates with volume smaller than smallest_4Dvolume
    if filtering4D:
        print('Filtering...')
        tic = time_lib.time()
        sequence_mask = remove_small_agglomerates(sequence_mask, smallest_4Dvolume)
        toc = time_lib.time()
        print(f'Filtering completed in {toc-tic:.2f} s\n')
    return sequence_mask



# these exploration algorithms have to be adapted to the new segmentation algorithm (applied on segmented masks)
# the biggest agglomerate has to be removed since it is the external shell
def explore_volume(exp, start_time, end_time, first_slice, last_slice, time_steps_number, step, win):
    
    time_steps = np.arange(start_time, min(start_time+step*time_steps_number, end_time), time_steps_number, dtype=int)
    temp_area = np.zeros((len(time_steps), last_slice-first_slice))
    temp_number = np.zeros_like(temp_area)

    for t, time in enumerate(time_steps):
        sequence = read_3Dsequence(exp, time=time, first_slice=first_slice, last_slice=last_slice, volume=True, win=win)
        segmented_image = (np.zeros_like(sequence)).astype(int)

        for z in range(sequence.shape[0]):
            segmented_image[z,:,:] = segment3D(sequence[z,:,:])
        new_segmented_image = propagate_labels(segmented_image)

        for z in range(sequence.shape[0]):
            rps = regionprops(new_segmented_image[z,:,:])
            areas = [r.area for r in rps]
            areas.pop(np.argmax(areas))
            if areas == []:
                temp_area[t, z] = 0
                temp_number[t, z] = 0
            else:
                temp_area[t, z] = np.mean(areas)
                temp_number[t, z] = len(areas)

    volume_area = feature(np.mean(temp_area, axis=1), np.mean(temp_area), np.std(temp_area))
    volume_number = feature(np.mean(temp_number, axis=1), np.mean(temp_number), np.std(temp_number))
    
    return volume_number, volume_area
            


def explore_slice(exp, start_time, end_time, first_slice, last_slice, volumes_number, win):

    slices = np.linspace(first_slice, last_slice, volumes_number, dtype=int)
    temp_area = np.zeros((len(slices), end_time-start_time)) 
    temp_number = np.zeros_like(temp_area)

    for z, slice in enumerate(slices):
        sequence = read_3Dsequence(exp, slice=slice, start_time=start_time, end_time=end_time,  volume=False, win=win)
        for i in range(0, sequence.shape[0], 2):
            sequence[i,:,:] = rotate180(sequence[i,:,:])
        segmented_image = (np.zeros_like(sequence)).astype(int)

        for t in range(sequence.shape[0]):
            segmented_image[t,:,:] = segment3D(sequence[t,:,:])
        new_segmented_image = propagate_labels(segmented_image)
        for t in range(sequence.shape[0]):
            rps = regionprops(new_segmented_image[t,:,:])
            areas = [r.area for r in rps]
            areas.pop(np.argmax(areas))
            if areas == []:
                temp_area[z, t] = 0
                temp_number[z, t] = 0
            else:
                temp_area[z, t] = np.mean(areas)
                temp_number[z, t] = len(areas)
    slice_area = feature(np.mean(temp_area, axis=1), np.mean(temp_area), np.std(temp_area))
    slice_number = feature(np.mean(temp_number, axis=1), np.mean(temp_number), np.std(temp_number))
    slice_stability_time = slice_number

    return slice_number, slice_area, slice_stability_time



def explore_experiment(exp, time_steps_number=5, volumes_number=5, end_time=220, first_slice=20, last_slice=260, step=5, win=True):

    start_time = exp_start_time()[exp_list().index(exp)]

    volume_number, volume_area = explore_volume(exp, start_time, end_time, first_slice, last_slice, time_steps_number, step, win)
    slice_number, slice_area, slice_stability_time = explore_slice(exp, start_time, end_time, first_slice, last_slice, volumes_number, win)
    
    return experiment(exp, volume_number, volume_area, slice_number, slice_area, slice_stability_time)



# this function has to be optimized since it is very slow
# da utilizzare il bincount!!!
def find_threshold(sequence, threshold=0, step=1, target=5000, delta=50):

    print('Finding threshold...')
    tic = time_lib.time()
    flag = False
    add = True
    while not flag:
        current_area = find_biggest_area(sequence, threshold)
        if current_area < target - delta:
            threshold -= step
            if add:
                step = step/2
                add = True
        elif current_area > target + delta:
            threshold += step
            if not add:
                step = step/2
                add = False
        else:
            flag = True
    toc = time_lib.time()
    print(f'Threshold={threshold:.2f} found in {toc-tic:.2f} s\n')

    return threshold