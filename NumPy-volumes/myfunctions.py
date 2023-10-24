import numpy as np                                  # type: ignore
from skimage.measure import label                   # type: ignore
from skimage.io import imread                       # type: ignore
from tqdm import tqdm                               # type: ignore
from numba import jit                               # type: ignore
import napari                                       # type: ignore
from dataclasses import dataclass
import time as clock
import shutil
import os


# class defined for data exploration purposes
@dataclass
class experiment:
    fixed_t_volume: np.ndarray                   # volume of the agglomerates contained in each slice of a volume
    fixed_t_number: np.ndarray                   # number of agglomerates contained in each slice of a volume
    fixed_z_volume: np.ndarray                   # volume of the agglomerates contained in a fixed slice at different time instants
    fixed_z_number: np.ndarray                   # number of agglomerates contained in a fixed slice at different time instants

# function returning the list of the experiments names
def exp_list():
    return ['P28A_FT_H_Exp1', 'P28A_FT_H_Exp2', 'P28A_FT_H_Exp3_3', 'P28A_FT_H_Exp4_2', 'P28A_FT_H_Exp5_2', 
            'P28A_FT_N_Exp1', 'P28A_FT_N_Exp4', 'P28B_ICS_FT_H_Exp5', 'P28B_ISC_FT_H_Exp2', 
            'P28B_ISC_FT_H_Exp3', 'P28B_ISC_FT_H_Exp4', 'P28B_ISC_FT_H_Exp4_2', 'VCT5_FT_N_Exp1', 
            'VCT5_FT_N_Exp3', 'VCT5_FT_N_Exp4', 'VCT5_FT_N_Exp5', 'VCT5A_FT_H_Exp1',
            'VCT5A_FT_H_Exp2', 'VCT5A_FT_H_Exp3', 'VCT5A_FT_H_Exp4', 'VCT5A_FT_H_Exp5']

# function returning the list of the time entries in which the degradation of the battery starts
def exp_start_time():
    return [112, 99, 90, 90, 108, 127, 130, 114, 99, 105, 104, 115, 155, 70, 54, 7, 71, 52, 4, 66, 65]

# function
def exp_flag():
    return [False, False, False, True, True, False, True, True, False, True, False, False, False, False, False, 
            True, False, False, True, False, False]

# function returning an iterator if verbose=False, otherwise it returns a tqdm iterator
def iterator(iterable, verbose=False, desc=''):
    if verbose:
        return tqdm(iterable, desc=desc)
    else:
        return iterable

# function returning the image rotated by 180 degrees
def rotate180(image):
    return np.rot90(np.rot90(image))

# function returning the mask of an image given a threshold: agglomerates are labeled with different values
def mask(image, threshold):
    return np.vectorize(label, signature='(n,m)->(n,m)')(image > threshold)

# function removing the agglomerates with volume smaller than smallest_volume. volume can be intended as both 3D and 4D
def remove_small_agglomerates(sequence_mask, smallest_volume):
    bincount = np.bincount(sequence_mask.flatten())
    sequence_mask[np.isin(sequence_mask, np.where(bincount < smallest_volume))] = 0
    return sequence_mask

# function used to remove the agglomerates that are not present in neighboring slices
def remove_isolated_agglomerates(sequence_mask, verbose=False):
    for i in iterator(range(sequence_mask.shape[0]), verbose, desc='Isolated agglomerates removal'):
        current_slice = sequence_mask[i,:,:]
        if not i == 0:
            previous_slice = sequence_mask[i-1,:,:]
        else:
            previous_slice = np.zeros_like(current_slice)   # previous_slice for first slice is set ad-hoc in order to avoid going out of bounds
        if not i == sequence_mask.shape[0]-1:
            next_slice = sequence_mask[i+1,:,:]
        else:
            next_slice = np.zeros_like(current_slice)       # next_slice for last slice is set ad-hoc in order to avoid going out of bounds
        current_labels = np.unique(current_slice)
        for current_label in current_labels:
            if current_label == 0:
                continue
            if current_label not in previous_slice and current_label not in next_slice:
                current_slice[current_slice == current_label] = 0
        sequence_mask[i,:,:] = current_slice
    return sequence_mask

# function used to remove the agglomerates that are not appearing consecutively for at least (time_steps) time instants
# THIS HAS TO BE REVISED: IN THIS WAY THE AGGLOMERATES THAT ARE NOT PRESENT IN ONE OF THE TIME INSTANTS AFTER APPEARENCE ARE REMOVED
# INSTEAD ONLY AGGLOMERATES WHICH APPEAR FOR A SHORT TIME AND THEN DISAPPEAR SHOULD BE REMOVED
def remove_inconsistent_agglomerates(sequence_mask, time_steps=10):
    for i in tqdm(range(sequence_mask.shape[0]-time_steps), desc='Inconsistent agglomerates removal'):
        current_volume = sequence_mask[i,:,:,:]
        next_volumes = sequence_mask[i+1:i+time_steps,:,:,:]
        current_labels = np.unique(current_volume)
        for current_label in current_labels:
            if current_label == 0:
                continue
            for j in range(next_volumes.shape[0]):
                if current_label not in next_volumes[j,:,:]:
                    sequence_mask[sequence_mask == current_label] = 0
                    break
    return sequence_mask

# function returning the area associated to the biggest agglomerate in the sequence
def find_biggest_area(sequence, threshold):
    sequence_mask = np.zeros_like(sequence, dtype=np.ushort)
    max_area = np.zeros(sequence.shape[0])
    for i in range(sequence.shape[0]):
        sequence_mask[i,:,:] = mask(sequence[i,:,:], threshold)
        _, label_counts = np.unique(sequence_mask[i,:,:], return_counts=True)
        label_counts[0] = 0
        max_area[i] = np.max(label_counts)
    return np.mean(max_area)

# function returning the path of the experiment given the experiment name and the OS
def OS_path(exp, OS):
    if OS=='Windows':
        return 'Z:/rot_datasets/' + exp
    elif OS=='MacOS':
        return '../MasterThesisData/' + exp
    elif OS=='Linux':
        return '/data/projects/whaitiri/Data/Data_Processing_July2022/rot_datasets/' + exp
    elif OS=='Tyrex':
        return 'U:/whaitiri/Data/Data_Processing_July2022/rot_datasets/' + exp
    else:
        raise ValueError('OS not recognized')



def volume_path(exp, time, isImage=True, OS='Windows', flag=False):
    vol = '0050' if flag else '0100'
    folder_name = 'entry' + str(time).zfill(4) + '_no_extpag_db' + vol + '_vol'
    volume_name = 'volume_v2.npy' if isImage else 'segmented.npy'
    return os.path.join(OS_path(exp, OS), folder_name, volume_name)



def save_volume(volume, exp, time, OS):
    try:
        np.save(volume_path(exp=exp, time=time, isImage=False, OS=OS), volume)
    except:
        print('Error saving segmentation map')
    return None



def load_volume(exp, time, isImage, OS):
    try:
        return np.load(volume_path(exp=exp, time=time, isImage=isImage, OS=OS))
    except:
        print('Error loading segmentation map')
    return None



# function returning the threshold value that allows to segment the sequence in a way that the area of the biggest agglomerate is equal to target
def find_threshold(sequence, threshold=0, step=1, target=5700, delta=500, slices=5):
    print('\nFinding threshold...')
    tic = clock.time()
    if sequence.shape[0] > slices:              # if the sequence is too long, it is reduced to n=slices slices
        sequence = np.array([sequence[i,:,:] for i in np.linspace(0, sequence.shape[0]-1, slices, dtype=int)])
    flag = False                                # flag used to stop the while loop
    add = True                                  # flag used to decide whether to add or subtract the step
    while not flag:
        current_area = find_biggest_area(sequence, threshold)
        if current_area > target + delta:       # if the area is larger than target, the threshold is increased in order to reduce the area
            if not add:
                step = step/2                   # step is halved every time the direction of the step is changed
            threshold += step
            add = True
        elif current_area < target - delta:     # if the area is smaller than target, the threshold is decreased in order to increase the area
            if add:
                step = step/2               
            threshold -= step
            add = False
        else:                                   # if the area is close to target, the threshold is found
            flag = True
    toc = clock.time()
    print('Threshold={:.2f} found in {:.2f} s\n'.format(threshold, toc-tic))
    return threshold



# function used to propagate labels from previous_mask to current_mask
# the update is carried out in increasing order of the area of the agglomerates -> the agglomerates with the biggest area are updated last
# in this way the agglomerates with the biggest area will most probably propagate their labels and overwrite the labels of the smaller agglomerates
# if biggest=True, only the agglomerate with biggest overlap in current mask is considered for each label
# otherwise all the agglomerates with overlap>propagation_threshold in current mask are considered for each label
# if forward=True the new labels that didn't exist in the previous mask are renamed in order to achieve low values for the labels
def propagate_labels(previous_mask, current_mask, forward=True, biggest=False, propagation_threshold=10, verbose=False):
    if forward:
        max_label = np.max(previous_mask)
        current_mask[current_mask > 0] += max_label
    unique_labels, label_counts = np.unique(previous_mask, return_counts=True)
    ordered_labels = unique_labels[np.argsort(label_counts)]
    for previous_slice_label in ordered_labels:
        if previous_slice_label == 0:   # the background is not considered
            continue
        bincount = np.bincount(current_mask[previous_mask == previous_slice_label])
        if len(bincount) <= 1:      # if the agglomerate is not present in the current mask (i.e. bincount contains only background), the propagation is skipped
            continue
        bincount[0] = 0     # the background is not considered
        current_slice_label = np.argmax(bincount)
        current_mask[current_mask == current_slice_label] = previous_slice_label
        if not biggest:
            for current_slice_label in np.where(bincount > propagation_threshold)[0]:
                current_mask[current_mask == current_slice_label] = previous_slice_label
    if forward:
        new_labels = np.unique(current_mask[current_mask > np.max(previous_mask)])
        label_mapping = {new_label: max_label + i + 1 for i, new_label in enumerate(new_labels)}
        current_mask = np.vectorize(label_mapping.get)(current_mask, current_mask)
        # for i, new_label in enumerate(new_labels):
        #     current_mask[current_mask == new_label] = max_label + i + 1
    return current_mask



# function returning the sequence of segmented images given the sequence of images and the threshold
# if filtering is True, the agglomerates with volume smaller than smallest_volume are removed
def segment3D(sequence, threshold, smallest_volume=10, filtering=True):
    sequence_mask = np.zeros_like(sequence, dtype=np.ushort)
    sequence_mask[0,:,:] = mask(sequence[0,:,:], threshold)
    # masking of current slice and forward propagation from the first slice
    for i in range(1, sequence.shape[0]):
        sequence_mask[i,:,:] = mask(sequence[i,:,:], threshold)
        sequence_mask[i,:,:] = propagate_labels(sequence_mask[i-1,:,:], sequence_mask[i,:,:], forward=True)
    # backward propagation from the last slice
    for i in range(sequence_mask.shape[0]-1, 0, -1):
        sequence_mask[i-1,:,:] = propagate_labels(sequence_mask[i,:,:], sequence_mask[i-1,:,:], forward=False)
    # removal of the agglomerates with volume smaller than smallest_volume and of the agglomerates that are not present in neighboring slices
    if filtering:
        sequence_mask = remove_isolated_agglomerates(sequence_mask)
        sequence_mask = remove_small_agglomerates(sequence_mask, smallest_volume)
    return sequence_mask



# function returning the sequence of segmented volumes given the sequence of volumes and the threshold
# if filtering3D is True, the agglomerates with volume smaller than smallest_3Dvolume are removed
# if filtering4D is True, the agglomerates with volume smaller than smallest_4Dvolume are removed
# if backward is True, backward propagation is performed
def segment4D(exp, end_time=220, skip180=True, smallest_3Dvolume=10, smallest_4Dvolume=100, time_steps=10, filtering3D=True, filtering4D=True, OS='Windows', show=False):
    
    print(f'\nExp {exp} segmentation started\n')
    start_time = exp_start_time()[exp_list().index(exp)]
    print('Loading first volume...')
    previous_volume = load_volume(exp=exp, time=0, isImage=True, OS=OS)
    threshold = find_threshold(previous_volume)
    print('Segmenting first volume...')    
    previous_mask = segment3D(previous_volume, threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
    print('Saving first volume...')
    save_volume(volume=previous_mask, exp=exp, time=0, OS=OS)
    time_steps = range(start_time, end_time+1, 2) if skip180 else range(start_time, end_time+1)

    for time in tqdm(time_steps[1:], desc='Volume segmentation and forward propagation'):
        current_volume = load_volume(exp=exp, time=time, isImage=True, OS=OS)
        current_mask = segment3D(current_volume, threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
        current_mask = propagate_labels(previous_mask, current_mask, forward=True)
        save_volume(volume=current_mask, exp=exp, time=time, OS=OS)
        previous_mask = current_mask

    for time in tqdm(time_steps[:-1][::-1], desc='Backward propagation'):
        current_volume = load_volume(exp=exp, time=time, isImage=False, OS=OS)
        current_mask = propagate_labels(previous_mask, current_mask, forward=False)
        save_volume(volume=current_mask, exp=exp, time=time, OS=OS)
        previous_mask = current_mask

    if filtering4D:
        print('Loading 4D volume...')
        mask = np.zeros((len(time_steps), current_mask.shape[0], current_mask.shape[1], current_mask.shape[2]) , dtype=np.ushort)
        for i, time in enumerate(time_steps):
            mask[i,:,:,:] = load_volume(exp=exp, time=time, isImage=False, OS=OS)
        print('Removing small agglomerates...')
        mask = remove_small_agglomerates(mask, smallest_4Dvolume)
        mask = remove_inconsistent_agglomerates(mask, time_steps=10)

    if show:
        viewer = napari.Viewer()
        vol4D = np.zeros((len(time_steps), current_mask.shape[0], current_mask.shape[1], current_mask.shape[2]))
        seg4D = np.zeros((len(time_steps), current_mask.shape[0], current_mask.shape[1], current_mask.shape[2]), dtype=np.ushort)
        for i, time in enumerate(time_steps):
            vol4D[i,:,:,:] = load_volume(exp=exp, time=time, isImage=True, OS=OS)
            seg4D[i,:,:,:] = load_volume(exp=exp, time=time, isImage=False, OS=OS)
        images = [viewer.add_image(vol4D, name='Volume', opacity=0.4)]
        labels = [viewer.add_labels(seg4D, name='Labels', blending='additive', opacity=0.8)]
        settings = napari.settings.get_settings()
        settings.application.playback_fps = 5
        viewer.dims.current_step = (0, 0)

    print(f'\nExp {exp} segmentation completed\n')
    return None