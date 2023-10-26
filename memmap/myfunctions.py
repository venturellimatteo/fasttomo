import numpy as np                                 
from skimage.measure import label                  
from tqdm import tqdm                               
from numpy.lib.format import open_memmap
from dataclasses import dataclass
import time as clock
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
def iterator(iterable, verbose=False, desc='', leave=True):
    if verbose:
        return tqdm(iterable, desc=desc, leave=leave)
    else:
        return iterable

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

# function returning the path of the volume or the segmentation map given the experiment name, the time instant and the OS
def volume_path(exp, time, isImage=True, OS='Windows'):
    flag = exp_flag()[exp_list().index(exp)]
    vol = '0050' if flag else '0100'
    folder_name = 'entry' + str(time).zfill(4) + '_no_extpag_db' + vol + '_vol'
    volume_name = 'volume_v2.npy' if isImage else 'segmented.npy'
    return os.path.join(OS_path(exp, OS), folder_name, volume_name)



# function returning the threshold value that allows to segment the sequence in a way that the area of the biggest agglomerate is equal to target
def find_threshold(sequence, threshold=0, step=1, target=5500, delta=250, slices=5):
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
# all the agglomerates with overlap>propagation_threshold in current mask are considered for each label
# if forward=True the new labels that didn't exist in the previous mask are renamed in order to achieve low values for the labels
def propagate_labels(previous_mask, current_mask, forward=True, propagation_threshold=10, verbose=False, leave=False):
    if forward:
        max_label = np.max(previous_mask)
        current_mask[current_mask > 0] += max_label
    unique_labels, label_counts = np.unique(previous_mask, return_counts=True)
    ordered_labels = unique_labels[np.argsort(label_counts)]
    for previous_slice_label in iterator(ordered_labels, verbose=verbose, desc='Label propagation', leave=leave):
        if previous_slice_label == 0:   # the background is not considered
            continue
        bincount = np.bincount(current_mask[previous_mask == previous_slice_label])
        if len(bincount) <= 1:  # if the agglomerate is not present in the current mask (i.e. bincount contains only background), the propagation is skipped
            continue
        bincount[0] = 0     # the background is not considered
        current_slice_label = np.argmax(bincount)
        current_mask[current_mask == current_slice_label] = previous_slice_label
        for current_slice_label in np.where(bincount > propagation_threshold)[0]:
            current_mask[current_mask == current_slice_label] = previous_slice_label
    if forward:
        new_labels = np.unique(current_mask[current_mask > np.max(previous_mask)])
        for i, new_label in enumerate(new_labels):
            current_mask[current_mask == new_label] = max_label + i + 1
    return current_mask



# function used to create the memmaps for the 4D volume and the 4D segmentation map
# if cropping is True, the volume is cropped in order to reduce the size of the memmap
def create_memmaps(exp, time_steps, OS='Windows', cropping=True):
    print(f'\nExp {exp} memmaps creation started\n')
    # define shape as (len(time_steps), 270, 500, 500) if cropping is True, otherwise as (len(time_steps), volume.shape[0], volume.shape[1], volume.shape[2])
    if cropping:
        shape = (len(time_steps), 270, 500, 500)
    else:
        volume = open_memmap(volume_path(exp=exp, time=time_steps[0], OS=OS, isImage=True), mode='r')
        shape = (len(time_steps), volume.shape[0], volume.shape[1], volume.shape[2])
    # create the 4D volume memmap and load it with already existing volumes in .npy files
    hypervolume = open_memmap(os.path.join(OS_path(exp, OS), 'hypervolume.npy'), dtype=np.half, mode='w+', shape=shape)
    for t, time in tqdm(enumerate(time_steps), desc='Loading hypervolume memmap', total=len(time_steps)):
        volume = open_memmap(volume_path(exp=exp, time=time, OS=OS, isImage=True), mode='r')
        hypervolume[t,:,:,:] = volume[10:,208:708,244:744] if cropping else volume
    # create the 4D segmentation mask memmap
    hypervolume_mask = open_memmap(os.path.join(OS_path(exp, OS), 'hypervolume_mask.npy'), dtype=np.ushort, mode='w+', shape=shape)
    print('Hypervolume_mask memmap created\n')
    return hypervolume, hypervolume_mask



# function used to compute the 4D filtering consisting in the removal of small agglomerates and the removal of inconsistent ones
# inconsistent agglomerates are the ones that disappear at a certain time instant
def compute_4Dfiltering (hypervolume_mask, bincounts, time_index, smallest_4Dvolume=100):
    # computation of the total bincount in the 4D mask
    max_label = np.max(hypervolume_mask[-1])
    total_bincount = np.zeros(max_label+1)
    for bincount in tqdm(bincounts, desc='Total bincount computation', leave=False):
        total_bincount[:len(bincount)] += bincount
    # removing the agglomerates with volume smaller than smallest_4Dvolume
    for label in tqdm(np.where(total_bincount < smallest_4Dvolume)[0], desc='Removing small agglomerates'):
        for time in time_index:
            hypervolume_mask[time][hypervolume_mask[time] == label] = 0
    # removing the agglomerates that are not appearing consecutively for at least (min_presence) time instants
    labels_to_remove = []
    print('Removing inconsistent agglomerates...')
    # finding the labels to remove
    for i, bincount in tqdm(enumerate(bincounts[:-1]), total=len(bincounts)-1, desc='Finding labels to remove'):
        next_bincount = bincounts[i+1]
        for label, count in enumerate(bincount):
            if count == 0 or label in labels_to_remove:
                continue
            if next_bincount[label] == 0:
                labels_to_remove.append(label)
                break
    # removing the labels
    for time in tqdm(time_index, desc='Removing labels'):
        for label in labels_to_remove:
            hypervolume_mask[time][hypervolume_mask[time] == label] = 0
    return None



# function used to rename the labels in the 4D segmentation map so that they are in the range [0, n_labels-1]
# using a set to store the labels allows analyze the labels present in each volume separately instead of computing np.unique on the whole 4D mask
def rename_labels(hypervolume_mask, time_index):
    unique_labels = set()
    for time in time_index:
        for label in np.unique(hypervolume_mask[time]):
            unique_labels.add(label)
    for time in tqdm(time_index, desc='Renaming labels'):
        for new_label, old_label in enumerate(unique_labels):
            hypervolume_mask[time][hypervolume_mask[time] == old_label] = new_label
    return None



# function returning the sequence of segmented images given the sequence of images and the threshold
# if filtering is True, the agglomerates with volume smaller than smallest_volume are removed
def segment3D(volume, threshold, smallest_volume=10, filtering=True):
    volume_mask = np.zeros_like(volume, dtype=np.ushort)
    volume_mask[0,:,:] = mask(volume[0,:,:], threshold)
    # masking of current slice and forward propagation from the first slice
    for i in tqdm(range(1, volume.shape[0]), desc='Volume masking and forward propagation', leave=False):
        volume_mask[i,:,:] = mask(volume[i,:,:], threshold)
        volume_mask[i,:,:] = propagate_labels(volume_mask[i-1,:,:], volume_mask[i,:,:], forward=True)
    # backward propagation from the last slice
    for i in tqdm(range(volume_mask.shape[0]-1, 0, -1), desc='Volume backward propagation', leave=False):
        volume_mask[i-1,:,:] = propagate_labels(volume_mask[i,:,:], volume_mask[i-1,:,:], forward=False)
    # removal of the agglomerates with volume smaller than smallest_volume and of the agglomerates that are not present in neighboring slices
    if filtering:
        volume_mask = remove_isolated_agglomerates(volume_mask)
        volume_mask = remove_small_agglomerates(volume_mask, smallest_volume)
    return volume_mask



# function returning the sequence of segmented volumes given the sequence of volumes and the threshold
# if filtering3D is True, the agglomerates with volume smaller than smallest_3Dvolume are removed
# if filtering4D is True, the agglomerates with volume smaller than smallest_4Dvolume are removed
# if backward is True, backward propagation is performed
def segment4D(exp, end_time=220, skip180=True, smallest_3Dvolume=10, smallest_4Dvolume=100, filtering3D=True, filtering4D=True, OS='Windows'):
    print(f'\nExp {exp} segmentation started\n')
    # defining the time steps for the current experiment
    start_time = exp_start_time()[exp_list().index(exp)]
    time_steps = range(start_time, end_time+1, 2) if skip180 else range(start_time, end_time+1)
    time_index = range(len(time_steps))
    hypervolume, hypervolume_mask = create_memmaps(exp, time_steps, OS)

    # dealing with the first volume
    previous_volume = hypervolume[0]
    # evaluating the threshold on the first volume
    threshold = find_threshold(previous_volume)  
    # segmenting the first volume
    print('Segmenting first volume...')
    previous_mask = segment3D(previous_volume, threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
    print('First volume segmentation completed\n')
    # reassigning the labels after the filtering
    for new_label, old_label in enumerate(np.unique(previous_mask)):
        if old_label == 0:
            continue
        previous_mask[previous_mask == old_label] = new_label
    # writing the first mask in the hypervolume_mask memmap
    hypervolume_mask[0] = previous_mask

    # initializing the bincount list
    if filtering4D:
        bincounts = [np.bincount(previous_mask.flatten())]
    # segmenting the remaining volumes and propagating the labels from previous volumes
    for time in tqdm(time_index[1:], desc='Volume segmentation and forward propagation'):
        current_volume = hypervolume[time]
        current_mask = segment3D(current_volume, threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
        current_mask = propagate_labels(previous_mask, current_mask, forward=True, verbose=True, leave=False)
        if filtering4D:     # computing the bincount of the current mask that will be used for the 4D filtering
            bincounts.append(np.bincount(current_mask.flatten()))
        hypervolume_mask[time] = current_mask
        previous_mask = current_mask

    # propagating the labels from the last volume to the previous ones
    # for time in tqdm(time_index[:-1][::-1], desc='Backward propagation'):
    #     current_mask = propagate_labels(previous_mask, current_mask, forward=False, verbose=True, leave=False)
    #     hypervolume_mask[time] = current_mask
    #     previous_mask = current_mask

    # removing the agglomerates with volume smaller than smallest_4Dvolume and the agglomerates that disappear after a certain time instant
    if filtering4D:
        compute_4Dfiltering(hypervolume_mask, bincounts, time_index, smallest_4Dvolume)
        rename_labels(hypervolume_mask, time_index)    
    
    print(f'\nExp {exp} segmentation completed!\n')
    return None