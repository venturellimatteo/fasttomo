import numpy as np                                 
from skimage.measure import label, regionprops                  
from tqdm import tqdm                               
from numpy.lib.format import open_memmap
# from dataclasses import dataclass
import time as clock
import pandas as pd
import os



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
        return '../../MasterThesisData/' + exp
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
def find_threshold(sequence, threshold=0, step=1, target=5400, delta=200, slices=5):
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
def segment4D(exp, end_time=220, skip180=True, smallest_3Dvolume=50, filtering3D=True, OS='Windows'):
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

    # segmenting the remaining volumes and propagating the labels from previous volumes
    for time in tqdm(time_index[1:], desc='Volume segmentation and forward propagation'):
        current_volume = hypervolume[time]
        current_mask = segment3D(current_volume, threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
        current_mask = propagate_labels(previous_mask, current_mask, forward=True, verbose=True, leave=False)
        hypervolume_mask[time] = current_mask
        previous_mask = current_mask
            
    print(f'\nExp {exp} segmentation completed!\n')
    return hypervolume_mask



# function used to remove small agglomerates in terms of 4D volume
def remove_small_4D_agglomerates (hypervolume_mask, bincounts, time_index, smallest_4Dvolume=100):
    # computation of the total bincount in the 4D mask
    max_label = np.max(hypervolume_mask[-1])
    total_bincount = np.zeros(max_label+1)
    for bincount in bincounts:
        total_bincount[:len(bincount)] += bincount
    # removing the agglomerates with volume smaller than smallest_4Dvolume
    for time in tqdm(time_index, desc='Removing small agglomerates'):
        for label in np.where(total_bincount < smallest_4Dvolume)[0]:
            if label < len(bincounts[time]):
                if bincounts[time][label] > 0:
                    hypervolume_mask[time][hypervolume_mask[time] == label] = 0
    return None



# function used to remove the agglomerates that are not present in each of the following n_steps bincounts
def remove_inconsistent_4D_agglomerates (hypervolume_mask, bincounts, time_index, n_steps=10):
    # remove list is a list containing the labels that will be removed from start_time to end_time volumes in the format [label, start_time, end_time]
    remove_list = []
    max_label = np.max(hypervolume_mask)
    # looping through the bincounts related to each time instant
    for i, bincount in tqdm(enumerate(bincounts[:-n_steps]), total=len(bincounts)-n_steps, desc='Finding labels to remove', leave=False):
        # define previous bincount as the bincount of the previous time instant if i != 0, otherwise define it as an array of zeros
        prev_bincount = bincounts[i-1] if i != 0 else np.zeros_like(bincount)
        # looping through the labels in the current bincount
        for label, count in enumerate(bincount):
            # if the label is contained in the previous bincount, it is not considered (we check only for the first time a label appears)
            if label < len(prev_bincount):
                if prev_bincount[label] != 0 or count == 0:
                    continue
            # if the label is not contained in each of the following n_steps bincounts, it is added to the set of labels to remove
            for j in range(i+1, i+1+n_steps):
                next_bincount = bincounts[j]
                if next_bincount[label] == 0:
                    remove_list.append(np.array([label, i, j]))
                    break
    # converting the remove_list to more useful format for looping
    # remove_array is a boolean array of shape (len(time_index), max_label) where each row contains True for the labels that will be removed in that time instant
    remove_array = np.zeros((len(time_index), int(max_label)), dtype=bool)
    for element in remove_list:
        remove_array[element[1]:element[2]+1, element[0]] = True
    # removing the labels
    for time in tqdm(time_index, desc='Removing inconsistent agglomerates'):
        for label in np.where(remove_array[time])[0]:
            hypervolume_mask[time][hypervolume_mask[time] == label] = 0
    return None



# function used to compute the 4-dimensional filtering of the segmentation map
def filtering4D(hypervolume_mask, smallest_4Dvolume=250, n_steps=10):
    print('\n4D filtering started\n')
    bincounts = []
    time_index = range(hypervolume_mask.shape[0])
    for t in time_index:
        bincounts.append(np.bincount(hypervolume_mask[t].flatten()))
    remove_small_4D_agglomerates(hypervolume_mask, bincounts, time_index, smallest_4Dvolume)
    remove_inconsistent_4D_agglomerates(hypervolume_mask, bincounts, time_index, n_steps)
    rename_labels(hypervolume_mask, time_index)
    print(f'\n4D filtering completed!\n')
    return None



# function used to compute the ratio between pixels and physical units in meters, and the ratio between time steps and physical units in seconds
def find_ratio(hypervolume_mask, exp):
    if 'P28A' in exp:
        m_diameter = 0.0186
    elif 'P28B' in exp:
        m_diameter = 0.0186
    elif 'VCT5A' in exp:
        m_diameter = 0.01835
    elif 'VCT5' in exp:
        m_diameter = 0.0182
    else:
        raise ValueError('Experiment not recognized')
    # here I find the pixel width of the external shell
    rps = regionprops(hypervolume_mask[0,135])
    shell_index = np.argmax([rp.area for rp in rps])
    pixel_diameter = np.sqrt(rps[shell_index].area_bbox)
    m_z = 0.012 # total field of view in z direction in meters
    pixel_z = 280 # total field of view in z direction in pixels
    fps = 20 # frames per second
    # computing the actual quantities
    xy_ratio = m_diameter/pixel_diameter
    z_ratio = m_z/pixel_z
    V_ratio = xy_ratio * xy_ratio * z_ratio
    t_ratio = 1/fps
    radius = pixel_diameter/2
    return xy_ratio, z_ratio, V_ratio, t_ratio, radius



# function returning the position and the volume of the agglomerates in the 4D segmentation map
def motion_matrix(hypervolume_mask, exp, steps=3):
    print('\nComputing motion matrix...')
    max_label = np.max(hypervolume_mask)
    n_slices = hypervolume_mask.shape[1]
    n_time_instants = hypervolume_mask.shape[0]
    # the ratios between pixels and physical units are computed
    print('Computing ratios...')
    x_ratio, y_ratio, z_ratio, v_ratio, t_ratio, radius = find_ratio(hypervolume_mask, exp)
    # radii and slices are the values used to divide the volume in <steps> regions
    radii = np.linspace(0, radius*x_ratio, steps+1)
    slices = np.linspace(0, n_slices*z_ratio, steps+1)
    # position is a matrix containing the position of each agglomerate in each time instant, x and y coordinates are centered [m, m, m]
    position = np.zeros((n_time_instants, max_label-1, 3), dtype=np.double)
    # volume is a matrix containing the volume of each agglomerate in each time instant [m^3]
    volume = np.zeros((n_time_instants, max_label-1), dtype=np.double)
    for t in tqdm(range(n_time_instants), desc='Computing agglomerates position and volume'):
        labels, counts = np.unique(hypervolume_mask[t], return_counts=True)
        for index, label in enumerate(labels):
            if label <= 1:  # the background and the external shell are not considered
                continue
            position[t, label-2] = (np.mean(np.where(hypervolume_mask[t] == label), axis=1) - np.array([0, 249.5, 249.5])) * np.array([z_ratio, y_ratio, x_ratio])
            volume[t, label-2] = counts[index] * v_ratio
    # speed is a matrix containing the speed of each agglomerate in each time instant [m/s, m/s, m/s]
    speed = np.diff(position, axis=0) * t_ratio
    # volume_exp_rate is a matrix containing the expansion rate of each agglomerate in each time instant [m^3/s]
    volume_exp_rate = np.diff(volume, axis=0) * t_ratio
    # avg_volume is a matrix containing the average volume of the agglomerates each region of the battery [m^3]
    avg_volume = np.zeros((n_time_instants, steps, steps), dtype=np.double)
    # agg_number is a matrix containing the number of agglomerates each region of the battery [-]
    agg_number = np.zeros((n_time_instants, steps, steps), dtype=np.ushort)
    for t in tqdm(range(n_time_instants), desc='Computing average volume and number of agglomerates'):
        for z in range(steps):
            for r in range(steps):
                for label in range(max_label-1):
                    if (position[t, label, 0] != 0 and position[t, label, 1] != 0 and position[t, label, 2] != 0 and 
                        slices[z] <= position[t, label, 0] and position[t, label, 0] < slices[z+1] and 
                        radii[r] <= np.linalg.norm(position[t, label, 1:]) and np.linalg.norm(position[t, label, 1:]) < radii[r+1]):
                        avg_volume[t, z, r] += volume[t, label]
                        agg_number[t, z, r] += 1
    agg_number[agg_number == 0] = 1
    avg_volume = avg_volume / agg_number
    return position, volume, speed, volume_exp_rate, avg_volume, agg_number



# function returning the dataframe containing the motion properties of the agglomerates
def motion_df(hypervolume_mask, exp):
    print('\nComputing motion matrix...')
    df = pd.DataFrame(columns=['t', 'label', 'x', 'y', 'z', 'r', 'vx', 'vy', 'vz', 'v', 'V', 'dVdt', 'r_sect', 'z_sect'])
    # max_label = np.max(hypervolume_mask)
    n_time_instants = hypervolume_mask.shape[0]
    n_slices = hypervolume_mask.shape[1]
    # the ratios between pixels and physical units are computed
    xy_ratio, z_ratio, V_ratio, t_ratio, radius = find_ratio(hypervolume_mask, exp)
    center = np.array([0, 249.5, 249.5])
    rescale = np.array([z_ratio, xy_ratio, xy_ratio])
    # radii and slices are the values used to divide the volume in <steps> regions
    radii = np.linspace(0, radius*xy_ratio, 4)
    radii[3] = 1    # the last value is set to 1 in order to avoid going out of bounds
    slices = np.linspace(0, n_slices*z_ratio, 4)
    slices[3] = 1   # the last value is set to 1 in order to avoid going out of bounds
    r_sect_str = ['Core', 'Intermediate', 'External']
    z_sect_str = ['Top', 'Middle', 'Bottom'] # HERE I HAVE TO DOUBLE CHECK THE ORDER OF THE SECTIONS!!!
    current_labels = []
    # computing the actual quantities
    for time in tqdm(range(n_time_instants), desc='Computing motion dataframe'):
        prev_labels = current_labels
        current_labels = []
        labels, counts = np.unique(hypervolume_mask[time], return_counts=True)
        # converting the time index into seconds
        t = time * t_ratio
        for index, label in enumerate(labels):
            if label <= 1:
                continue
            current_labels.append(label)
            # evaluating the position of the centroid of the agglomerate
            z, y, x = (np.mean(np.where(hypervolume_mask[time] == label), axis=1) - center) * rescale
            # evaluating the distance of the agglomerate from the central axis of the battery
            r = np.linalg.norm([x, y])
            # assigning the agglomerate to a section of the battery
            for i in range(3):
                if slices[i] <= z and z < slices[i+1]:
                    z_sect = z_sect_str[i]
                if radii[i] <= r and r < radii[i+1]:
                    r_sect = r_sect_str[i]
            # evaluating the volume of the agglomerate
            V = counts[index] * V_ratio
            # evaluating the velocity and volume expansion rate of the agglomerate if it was present in the previous time instant
            # otherwise set these values to 0
            if label in prev_labels:
                x0, y0, z0 = df[(df['t'] == time-1) & (df['label'] == label)][['x', 'y', 'z']].values[0]
                vx, vy, vz = (x-x0)/t_ratio, (y-y0)/t_ratio, (z-z0)/t_ratio
                v = np.linalg.norm([vx, vy, vz])
                dVdt = (V - df[(df['t'] == time-1) & (df['label'] == label)]['V'].values[0])/t_ratio
            else:
                vx, vy, vz, v, dVdt = 0, 0, 0, 0, V/t_ratio
            # adding the row to the dataframe
            df = pd.concat([df, pd.DataFrame([[t, label, x, y, z, r, vx, vy, vz, v, V, dVdt, r_sect, z_sect]], 
                                             columns=['t', 'label', 'x', 'y', 'z', 'r', 'vx', 'vy', 'vz', 'v', 'V', 'dVdt', 'r_sect', 'z_sect'])], ignore_index=True)
    return df