import numpy as np                                  # type: ignore
from skimage.measure import label                   # type: ignore
from skimage.io import imread                       # type: ignore
from tqdm import tqdm                               # type: ignore
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
        return 'Z:/Reconstructions/' + exp
    elif OS=='MacOS':
        return '../MasterThesisData/' + exp
    elif OS=='Linux':
        return '/data/projects/whaitiri/Data/Data_Processing_July2022/Reconstructions/' + exp
    elif OS=='Tyrex':
        return 'U:/whaitiri/Data/Data_Processing_July2022/Reconstructions/' + exp
    else:
        raise ValueError('OS not recognized')

# function returning the path of the image given the experiment name, the time entry and the slice number
# if isSrc is True, the path is the one of the source images, otherwise it is the one of the destination images
def image_path(exp, time, slice, isSrc=True, dst='', OS='MacOS', flag=False):
    vol = '0050' if flag else '0100'
    folder_name = 'entry' + str(time).zfill(4) + '_no_extpag_db' + vol + '_vol'
    image_name = 'entry' + str(time).zfill(4) + '_no_extpag_db' + vol + '_vol_' + str(slice).zfill(6) + '.tiff'
    if isSrc:
        path = OS_path(exp, OS)
    else:
        path = os.path.join(dst, exp)
        if not os.path.exists(os.path.join(path, folder_name)):
            os.makedirs(os.path.join(path, folder_name))
    return os.path.join(path, folder_name, image_name)



# function copying the images from the source folder to the destination folder
def move_sequence(exp, first_slice, last_slice, start_time, end_time, dst, OS='Windows'):
    for time in range(start_time, end_time+1):
        for slice in range(first_slice, last_slice+1):
            src_dir = image_path(exp, time, slice, isSrc=True, OS=OS)
            dst_dir = image_path(exp, time, slice, isSrc=False, dst=dst, OS=OS)
            shutil.copyfile(src_dir, dst_dir)
    return None


# function to save the segmentation map as a numpy array in (t, z, y, x) form
def save_segmentation_map(segmented_sequence, exp, OS):
    print('Saving segmentation map...')
    try:
        tic = clock.time()
        np.save(os.path.join(OS_path(exp, OS), f'{exp}_segmented.npy'), segmented_sequence)
        toc = clock.time()
        print(f'Segmentation map saved successfully in {toc-tic:.2f} s')
    except:
        print('Error saving segmentation map')
    return None



# function to load the saved segmentation map as a numpy array in (t, z, y, x) form
def load_segmentation_map(exp, OS):
    print('Loading segmentation map...')
    try:
        tic = clock.time()
        segmented_sequence = np.load(os.path.join(OS_path(exp, OS), f'{exp}_segmented.npy'))
        toc = clock.time()
        print(f'Segmentation map loaded successfully in {toc-tic:.2f} s')
    except:
        print('Error loading segmentation map')
    return segmented_sequence



# function returning the threshold value that allows to segment the sequence in a way that the area of the biggest agglomerate is equal to target
def find_threshold(sequence, threshold=0, step=1, target=5700, delta=500, slices=5):
    print('\nFinding threshold...')
    tic = clock.time()
    if len(sequence.shape) == 4:                # if the sequence is in the form (t, z, y, x), it is converted to (z, y, x)
        sequence = sequence[0,:,:,:]
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
        clock.sleep(1)
    toc = clock.time()
    print(f'Threshold={threshold:.2f} found in {toc-tic:.2f} s\n')
    return threshold



# function returning the sequence of images given the experiment name, the time range and the slice range
# if volume is True, the sequence is in the form (z, y, x), otherwise it is in the form (t, y, x)
def read_3Dsequence(exp, time=0, slice=0, start_time=0, end_time=220, first_slice=20, last_slice=260, volume=True, OS='MacOS'):
    if volume:
        test_image = imread(image_path(exp, time, first_slice, OS=OS))  # test_image is used to determine the shape of the sequence
        sequence = np.zeros((last_slice-first_slice, test_image.shape[0], test_image.shape[1]))
        for slice in range(first_slice, last_slice):
            image = imread(image_path(exp, time, slice, OS=OS))
            sequence[slice-first_slice,:,:] = image
    else:
        test_image = imread(image_path(exp, start_time, slice, OS=OS))
        sequence = np.zeros((end_time-start_time, test_image.shape[0], test_image.shape[1]))
        for time in range(start_time, end_time):
            image = imread(image_path(exp, time, slice, OS=OS))
            sequence[time-start_time,:,:] = image
    return sequence



# function returning the sequence of images given the experiment name, the time range and the slice range in the form (t, z, y, x)
# half of the images are discarded because of the 180 degrees rotation and poor reconstruction
def read_4Dsequence(exp, first_slice=0, last_slice=279, start_time=0, end_time=220, OS='MacOS', skip180=True):
    step = 2 if skip180 else 1
    print(f'Collecting sequence for experiment {exp}...')
    if start_time == 0:
        start_time = exp_start_time()[exp_list().index(exp)]    # start_time is the time entry in which the degradation of the battery starts (picked from exp_start_time)
    flag = exp_flag()[exp_list().index(exp)]                    # flag is True if the experiment is 0050, False if it is 0100
    test_image = imread(image_path(exp, start_time, first_slice, OS=OS, flag=flag))  # test_image is used to determine the shape of the sequence
    time_steps = np.arange(start_time, end_time+1, step, dtype=np.ushort)
    sequence = np.zeros((len(time_steps), last_slice-first_slice+1, test_image.shape[0], test_image.shape[1]))
    for t, time in enumerate(iterator(time_steps, verbose=True, desc='Collecting sequence')):
        for slice in range(first_slice, last_slice+1):
            try:
                image = imread(image_path(exp, time, slice, OS=OS, flag=flag))
                sequence[t, slice-first_slice,:,:] = image
            except:
                print(f'Error reading image {image_path(exp, time, slice, OS=OS, flag=flag)}')
    return sequence



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
    for previous_slice_label in iterator(ordered_labels, verbose=verbose, desc='Propagating labels'):
        if previous_slice_label == 0:   # the background is not considered
            continue
        current_slice_labels = current_mask[previous_mask == previous_slice_label]
        bincount = np.bincount(current_slice_labels)
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
        for i, new_label in enumerate(new_labels):
            current_mask[current_mask == new_label] = max_label + i + 1
    return current_mask



# function returning the sequence of segmented images given the sequence of images and the threshold
# if filtering is True, the agglomerates with volume smaller than smallest_volume are removed
def segment3D(sequence, threshold, smallest_volume=50, filtering=True):
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
def segment4D(sequence, threshold, smallest_3Dvolume=50, smallest_4Dvolume=100, time_steps=10, filtering3D=True, filtering4D=True, backward=True, save=False, exp='', OS='Windows'):
    print('\nSegmenting and propagating labels...')
    sequence_mask = np.zeros_like(sequence, dtype=np.ushort)
    sequence_mask[0,:,:,:] = segment3D(sequence[0,:,:,:], threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
    # masking of current volume and forward propagation from the first volume
    for t in tqdm(range(1, sequence.shape[0]), desc='Volume segmentation and forward propagation'):
        sequence_mask[t,:,:,:] = segment3D(sequence[t,:,:,:], threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
        sequence_mask[t,:,:,:] = propagate_labels(sequence_mask[t-1,:,:,:], sequence_mask[t,:,:,:], forward=True)
    # backward propagation from the last volume
    if backward:
        for t in tqdm(range(sequence_mask.shape[0]-1, 0, -1), desc='Backward propagation'):
            sequence_mask[t-1,:,:,:] = propagate_labels(sequence_mask[t,:,:,:], sequence_mask[t-1,:,:,:], forward=False)
    # removal of the agglomerates with volume smaller than smallest_4Dvolume and of the agglomerates that are not present for at least time_steps time instants
    if filtering4D:
        print('\nFiltering...')
        tic = clock.time()
        sequence_mask = remove_small_agglomerates(sequence_mask, smallest_4Dvolume)
        toc = clock.time()
        print(f'Small agglomerates removed in {toc-tic:.2f} s')
        sequence_mask = remove_inconsistent_agglomerates(sequence_mask, time_steps=time_steps)
    if save and exp in exp_list():
        save_segmentation_map(sequence_mask, exp, OS)
    print('\n\n')
    return sequence_mask



# function returning the mean area and number of agglomerates (mean computed with respect to time and to z)
# the removal of the external shell can surely be done in a better way
def explore_experiment(segmented_sequence, threshold=1000):

    fixed_t_volume = np.zeros(segmented_sequence.shape[0], dtype=np.ushort)
    fixed_t_number = np.zeros(segmented_sequence.shape[0], dtype=np.ushort)
    fixed_z_volume = np.zeros(segmented_sequence.shape[1], dtype=np.ushort)
    fixed_z_number = np.zeros(segmented_sequence.shape[1], dtype=np.ushort)

    for t in iterator(range(segmented_sequence.shape[0]), verbose=True, desc='Evaluating the t axis'):
        _, label_counts = np.unique(segmented_sequence[t,:,:,:], return_counts=True)
        if len(label_counts) > 1:
            label_counts = label_counts[2:] # here the background and the external shell are removed
            fixed_t_volume[t] = np.mean(label_counts)
            fixed_t_number[t] = len(label_counts)
        else:
            fixed_t_volume[t] = 0
            fixed_t_number[t] = 0
    for z in iterator(range(segmented_sequence.shape[1]), verbose=True, desc='Evaluating the z axis'):
        _, label_counts = np.unique(segmented_sequence[:,z,:,:], return_counts=True)
        if len(label_counts) > 1:
            label_counts = label_counts[2:] # here the background and the external shell are removed
            fixed_z_volume[z] = np.mean(label_counts)
            fixed_z_number[z] = len(label_counts)
        else:
            fixed_z_volume[z] = 0
            fixed_z_number[z] = 0

    return experiment(fixed_t_volume=fixed_t_volume, fixed_t_number=fixed_t_number, fixed_z_volume=fixed_z_volume, fixed_z_number=fixed_z_number)