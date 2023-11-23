import numpy as np    
from numpy.lib.format import open_memmap                           
from skimage.measure import label, regionprops, marching_cubes
from skimage.morphology import erosion, dilation, ball    
from tqdm import tqdm                               
import pandas as pd
import os



# functions returning the lists of the experiments names and related features
def exp_list():
    return ['P28A_FT_H_Exp1', 'P28A_FT_H_Exp2', 'P28A_FT_H_Exp3_3', 'P28A_FT_H_Exp4_2', 'P28B_ISC_FT_H_Exp2', 'VCT5_FT_N_Exp1', 
            'VCT5_FT_N_Exp3', 'VCT5_FT_N_Exp4', 'VCT5_FT_N_Exp5', 'VCT5A_FT_H_Exp2', 'VCT5A_FT_H_Exp5']

def bad_exp_list():
    return ['P28B_ISC_FT_H_Exp3','P28B_ISC_FT_H_Exp4','P28B_ISC_FT_H_Exp4_2','P28B_ISC_FT_H_Exp5','VCT5A_FT_H_Exp1','VCT5A_FT_H_Exp4']

# function returning the diameter of the battery given the experiment name
def find_diameter(exp):
    if 'P28A' in exp:
        return 18.6 # [mm]
    elif 'P28B' in exp:
        return 18.6 # [mm]
    elif 'VCT5A' in exp:
        return 18.35 # [mm]
    elif 'VCT5' in exp:
        return 18.2 # [mm]
    else:
        raise ValueError('Experiment not recognized')

def update_pb(progress_bar, postfix):
    progress_bar.update()
    progress_bar.set_postfix_str(postfix)
    return None

# function returning the area associated to the biggest agglomerate in the sequence
def find_biggest_area(sequence, threshold):
    sequence_mask = segment3D(sequence, threshold)
    areas = [rp.area for rp in regionprops(sequence_mask)]
    return np.max(areas)/sequence_mask.shape[0]

def remove_small_agglomerates(hypervolume_mask, smallest_volume):
    bincount = np.bincount(hypervolume_mask.flatten())
    hypervolume_mask[np.isin(hypervolume_mask, np.where(bincount < smallest_volume))] = 0
    return None

# function used to remove the agglomerates that are not present in neighboring slices
def remove_isolated_agglomerates(hypervolume_mask):
    for time in range(hypervolume_mask.shape[0]):
        current_slice = hypervolume_mask[time]
        previous_slice = np.zeros_like(current_slice) if time==0 else hypervolume_mask[time-1] # previous_slice for first slice is set ad-hoc in order to avoid going out of bounds
        next_slice = np.zeros_like(current_slice) if time == hypervolume_mask.shape[0]-1 else hypervolume_mask[time+1]  # next_slice for last slice is set ad-hoc in order to avoid going out of bounds
        current_labels = [rp.label for rp in regionprops(current_slice)]
        for current_label in current_labels:
            if current_label not in previous_slice and current_label not in next_slice:
                current_slice[current_slice == current_label] = 0
        hypervolume_mask[time] = current_slice
    return None

# function used to rename the labels in the 4D segmentation map so that they are in the range [0, n_labels-1]
def rename_labels(hypervolume_mask, time_index):
    unique_labels = set()
    for t in time_index:
        for rp in regionprops(hypervolume_mask[t]):
            unique_labels.add(rp.label)
    unique_labels = np.array(list(unique_labels))
    total_labels = len(unique_labels)
    old_labels = unique_labels[unique_labels >= total_labels]
    new_labels = np.setdiff1d(np.arange(total_labels), unique_labels)
    lookup_table = np.arange(np.max(unique_labels)+1)
    lookup_table[old_labels] = new_labels
    hypervolume_mask = np.take(lookup_table, hypervolume_mask)
    return None

# function returning the 3D segmentation map given the 3D volume and the threshold
def segment3D(volume, threshold, filtering3D=True, smallest_3Dvolume=50):
    mask1 = np.greater(volume, threshold)
    mask2 = dilation(erosion(mask1, ball(1)), ball(4))
    mask = label(np.logical_and(mask1, mask2))
    if filtering3D:
        remove_small_agglomerates(mask, smallest_3Dvolume)
        remove_isolated_agglomerates(mask)
    return mask

# function returning the path of the experiment given the experiment name and the OS
def OS_path(exp, OS):
    if OS=='Windows':
        return 'Z:/rot_datasets/' + exp
    elif OS=='MacOS':
        return '/Users/matteoventurelli/Documents/VS Code/MasterThesisData/' + exp
    elif OS=='MacOS_SSD':
        return '/Volumes/T7/Thesis/' + exp
    elif OS=='Linux':
        return '/data/projects/whaitiri/Data/Data_Processing_July2022/rot_datasets/' + exp
    elif OS=='Tyrex':
        return 'U:/whaitiri/Data/Data_Processing_July2022/rot_datasets/' + exp
    else:
        raise ValueError('OS not recognized')

# function used to create the memmaps for the 4D volume and the 4D segmentation map
# if cropping is True, the volume is cropped in order to reduce the size of the memmap
def create_memmaps(exp, OS):
    # create the 4D volume memmap
    hypervolume = open_memmap(os.path.join(OS_path(exp, OS), 'hypervolume.npy'), mode='r')
    # create the 4D segmentation mask memmap
    hypervolume_mask = open_memmap(os.path.join(OS_path(exp, OS), 'hypervolume_mask.npy'), dtype=np.ushort, mode='w+', shape=hypervolume.shape)
    return hypervolume, hypervolume_mask



# function returning the threshold value that allows to segment the sequence in a way that the area of the biggest agglomerate is equal to target
def find_threshold(sequence, threshold=0, step=1, target=6800, delta=100, slices=10):
    if sequence.shape[0] > slices:              # if the sequence is too long, it is reduced to n=slices slices
        sequence = np.array([sequence[i] for i in np.linspace(0, sequence.shape[0]-1, slices, dtype=int)])
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
    return threshold



# function used to update the propagation map dictionary used in label propagation
def update_map(current_mask, previous_mask, previous_mask_label, propagation_map):
    current_mask_labels = current_mask[previous_mask == previous_mask_label]
    if np.any(current_mask_labels):
        current_mask_labels, counts = np.unique(current_mask_labels, return_counts=True)
        for current_slice_label, count in zip(current_mask_labels, counts):
            if current_slice_label > 0:
                if current_slice_label not in propagation_map:
                    propagation_map[current_slice_label] = np.array([previous_mask_label, count])
                elif count >= propagation_map[current_slice_label][1]:
                    propagation_map[current_slice_label] = np.array([previous_mask_label, count])
    return propagation_map



# function used to propagate labels from previous_mask to current_mask
# the update is carried out in increasing order of the area of the agglomerates -> the agglomerates with the biggest area are updated last
# in this way the agglomerates with the biggest area will most probably propagate their labels and overwrite the labels of the smaller agglomerates
# if forward=True the new labels that didn't exist in the previous mask are renamed in order to achieve low values for the labels
def propagate_labels(previous_mask, current_mask, forward=True):
    if forward:
        max_label = np.max(previous_mask)
        current_mask[current_mask > 0] += max_label + 1

    rps = regionprops(previous_mask)
    labels = [rp.label for rp in rps]
    areas = [rp.area for rp in rps]
    ordered_labels = np.array(labels)[np.argsort(areas)]
    propagation_map = dict()

    for previous_mask_label in ordered_labels:
        propagation_map = update_map(current_mask, previous_mask, previous_mask_label, propagation_map)
    for current_slice_label, previous_mask_label in propagation_map.items():
        current_mask[current_mask == current_slice_label] = previous_mask_label[0]

    if forward:
        new_labels = np.unique(current_mask[current_mask > max_label])
        lookup_table = np.arange(np.max(new_labels)+1)
        lookup_table[new_labels] = np.arange(len(new_labels)) + max_label + 1
        current_mask = np.take(lookup_table, current_mask)
    return current_mask



# function returning the sequence of segmented volumes given the sequence of volumes and the threshold
# if filtering3D is True, the agglomerates with volume smaller than smallest_3Dvolume are removed
# if filtering4D is True, the agglomerates with volume smaller than smallest_4Dvolume are removed
# if backward is True, backward propagation is performed
def segment4D(exp, OS='Windows', offset=0, filtering3D=True, smallest_3Dvolume=50):
    # setting up the progress_bar
    progress_bar = tqdm(total=4, desc=f'{exp} preparation', position=offset, leave=False)

    # creating the memmaps for the 4D volume and the 4D segmentation map
    progress_bar.set_postfix_str('Creating memmaps')
    hypervolume, hypervolume_mask = create_memmaps(exp, OS) 
    time_index = range(hypervolume.shape[0]) # defining the time steps for the current experiment
    previous_volume = hypervolume[0] # dealing with the first volume

    # evaluating the threshold on the first volume
    update_pb(progress_bar, 'Finding threshold')
    threshold = find_threshold(previous_volume) 

    # segmenting the first volume
    update_pb(progress_bar, 'Segmenting first volume')
    previous_mask = segment3D(previous_volume, threshold, filtering3D, smallest_3Dvolume)

    # reassigning the labels after the filtering
    update_pb(progress_bar, 'Reassigning labels')
    rps = regionprops(previous_mask)
    old_labels = [rp.label for rp in rps]
    old_labels = np.array(old_labels)[np.argsort([rp.area for rp in rps])][::-1]
    lookup_table = np.arange(np.max(old_labels)+1)
    lookup_table[old_labels] = np.arange(len(old_labels))+1
    previous_mask = np.take(lookup_table, previous_mask)

    # writing the first mask in the hypervolume_mask memmap
    hypervolume_mask[0] = previous_mask
    progress_bar.close()

    # segmenting the remaining volumes and propagating the labels from previous volumes
    progress_bar = tqdm(time_index[1:], desc=f'{exp} segmentation', position=offset, leave=False)
    for time in progress_bar:
        progress_bar.set_postfix_str('Segmenting volume')
        current_volume = hypervolume[time]
        current_mask = segment3D(current_volume, threshold, filtering3D, smallest_3Dvolume)
        progress_bar.set_postfix_str('Propagating labels')
        current_mask = propagate_labels(previous_mask, current_mask, forward=True)
        hypervolume_mask[time] = current_mask
        previous_mask = current_mask

    return hypervolume_mask



# function used to compute the 4-dimensional filtering of the segmentation map
def filtering4D(hypervolume_mask, exp, smallest_4Dvolume=250, offset=0):
    time_index = range(hypervolume_mask.shape[0])
    progress_bar = tqdm(total=3, desc=f'{exp} filtering', position=offset, leave=False)
    progress_bar.set_postfix_str('Isolated labels removal')
    remove_isolated_agglomerates(hypervolume_mask, time_index, progress_bar)
    update_pb(progress_bar, 'Small agglomerates removal')
    remove_small_agglomerates(hypervolume_mask, smallest_4Dvolume)
    # remove_pre_TR_agglomerates(hypervolume_mask, time_index, exp)
    update_pb(progress_bar, 'Labels renaming')
    rename_labels(hypervolume_mask, time_index, progress_bar)
    progress_bar.close()
    return None



# function used to update the dataframe containing the motion properties of the agglomerates
def update_df(df, label, V, centroid, slices, radii, z_sect_str, r_sect_str, t_ratio, t, prev_labels):
    # evaluating the position of the centroid of the agglomerate
    z, y, x = centroid
    # evaluating the distance of the agglomerate from the central axis of the battery
    r = np.linalg.norm([x, y])
    # assigning the agglomerate to a section of the battery
    for i in range(3):
        if slices[i] <= z and z < slices[i+1]:
            z_sect = z_sect_str[i]
        if radii[i] <= r and r < radii[i+1]:
            r_sect = r_sect_str[i]
    # evaluating the velocity and volume expansion rate of the agglomerate if it was present in the previous time instant
    # otherwise set these values to 0
    if label in prev_labels:
        x0, y0, z0 = (df.iloc[prev_labels[label]][['x', 'y', 'z']]).values
        vx, vy, vz = (x-x0)/t_ratio, (y-y0)/t_ratio, (z-z0)/t_ratio
        vxy = np.linalg.norm([vx, vy])
        v = np.linalg.norm([vx, vy, vz])
        dVdt = (V - (df.iloc[prev_labels[label]]['V']))/t_ratio
    else:
        vx, vy, vxy, vz, v, dVdt = 0, 0, 0, 0, V/t_ratio
    # adding the row to the dataframe
    df = pd.concat([df, pd.DataFrame([[t, label, x, y, z, r, vx, vy, vxy, vz, v, V, dVdt, r_sect, z_sect]],
                                     columns=['t', 'label', 'x', 'y', 'z', 'r', 'vx', 'vy', 'vxy', 'vz', 'v', 'V', 'dVdt', 'r_sect', 'z_sect'])])
    return df



# function returning the dataframe containing the motion properties of the agglomerates
def motion_df(hypervolume_mask, exp, offset=0):
    df = pd.DataFrame(columns=['t', 'label', 'x', 'y', 'z', 'r', 'vx', 'vy', 'vxy', 'vz', 'v', 'V', 'dVdt', 'r_sect', 'z_sect'])
    n_time_instants = hypervolume_mask.shape[0]
    n_slices = hypervolume_mask.shape[1]
    xyz_ratio, V_ratio, t_ratio = 0.04, 0.000064, 0.05
    xy_center = (hypervolume_mask.shape[2]-1)/2
    center = np.array([0, xy_center, xy_center])
    radius = find_diameter(exp)/(2*xyz_ratio)
    # radii and slices are the values used to divide the volume in <steps> regions
    radii = np.linspace(0, radius*xyz_ratio, 4)
    radii[3] = np.inf    # the last value is set to inf in order to avoid going out of bounds
    slices = np.linspace(0, n_slices*xyz_ratio, 4)
    slices[3] = np.inf   # the last value is set to inf in order to avoid going out of bounds
    r_sect_str = ['Core', 'Intermediate', 'External']
    z_sect_str = ['Bottom', 'Middle', 'Top'] 
    current_labels = dict()
    # computing the actual quantities
    for time in tqdm(range(n_time_instants), desc=f'Exp {exp} dataframe computation', position=offset, leave=False):
        prev_labels = current_labels
        current_labels = dict()
        rps = regionprops(hypervolume_mask[time])
        labels = [rp.label for rp in rps]
        volumes = [(rp.area * V_ratio) for rp in rps] 
        centroids = [((rp.centroid - center) * xyz_ratio) for rp in rps]
        # converting the time index into seconds
        t = time * t_ratio
        for index, label in enumerate(labels):
            if label > 1:
                current_labels[label] = len(df)
                df = update_df(df, label, volumes[index], centroids[index], slices, radii, z_sect_str, r_sect_str, t_ratio, t, prev_labels)  
    return df


# def exp_start_TR():
#     return [7, 4, 4, 6, 3, 1, 4, 5, 3, 1, 1]

# this is used to mask the inside of the external shell
# a method to find the correct slice is needed (sometimes agglomerates inside share the same label as the shell)
# def create_shell_border(hypervolume, threshold, z=50):
#     mask = label(hypervolume[z]>threshold)
#     rps = regionprops(mask)
#     labels = [rp.label for rp in rps]
#     shell_inside = clear_border(np.ones_like(mask) - (mask==labels[np.argmax([rp.area for rp in rps])]))
#     eroded_shell_inside = erosion(shell_inside, footprint=np.ones((5,5)))
#     return 1 - (shell_inside - eroded_shell_inside)


# # function used to update the remove_list used to remove the agglomerates that are not present in each of the following n_steps bincounts
# def update_remove_list(bincounts, bincount, remove_list, n_steps, i):
#     # define previous bincount as the bincount of the previous time instant if i != 0, otherwise define it as an array of zeros
#     prev_bincount = bincounts[i-1] if i != 0 else np.zeros_like(bincount)
#     # looping through the labels in the current bincount
#     for label, count in enumerate(bincount):
#         # if the label is contained in the previous bincount, it is not considered (we check only for the first time a label appears)
#         if label < len(prev_bincount):
#             if prev_bincount[label] != 0 or count == 0:
#                 continue
#         # if the label is not contained in each of the following n_steps bincounts, it is added to the set of labels to remove
#         for j in range(i+1, i+1+n_steps):
#             next_bincount = bincounts[j]
#             if next_bincount[label] == 0:
#                 remove_list.append(np.array([label, i, j]))
#                 break
#     return remove_list



# # function used to remove the agglomerates that are not present in each of the following n_steps bincounts
# def remove_inconsistent_4D_agglomerates (hypervolume_mask, bincounts, time_index, n_steps, progress_bar):
#     # remove list is a list containing the labels that will be removed from start_time to end_time volumes in the format [label, start_time, end_time]
#     remove_list = []
#     max_label = np.max(hypervolume_mask)
#     # looping through the bincounts related to each time instant and updating the remove_list
#     for i, bincount in enumerate(bincounts[:-n_steps]):
#         remove_list = update_remove_list(bincounts, bincount, remove_list, n_steps, i)
#     # converting the remove_list to more useful format for looping
#     # remove_array is a boolean array of shape (len(time_index), max_label) where each row contains True for the labels that will be removed in that time instant
#     remove_array = np.zeros((len(time_index), int(max_label)), dtype=bool)
#     for element in remove_list:
#         remove_array[element[1]:element[2]+1, element[0]] = True
#     # removing the labels
#     for time in time_index:
#         for label in np.where(remove_array[time])[0]:
#             hypervolume_mask[time][hypervolume_mask[time] == label] = 0
#         progress_bar.update()
#     return None



# # function used to remove small agglomerates in terms of 4D volume
# def remove_small_4D_agglomerates (hypervolume_mask, bincounts, time_index, smallest_4Dvolume, progress_bar):
#     # computation of the total bincount in the 4D mask
#     max_label = np.max(hypervolume_mask)
#     total_bincount = np.zeros(max_label+1)
#     for bincount in bincounts:
#         total_bincount[:len(bincount)] += bincount
#     # removing the agglomerates with volume smaller than smallest_4Dvolume
#     for time in time_index:
#         for label in np.where(total_bincount < smallest_4Dvolume)[0]:
#             if label < len(bincounts[time]):
#                 if bincounts[time][label] > 0:
#                     hypervolume_mask[time][hypervolume_mask[time] == label] = 0
#         progress_bar.update()
#     return None

# # function used to remove the agglomerates that appear before the thermal runaway
# def remove_pre_TR_agglomerates(hypervolume_mask, time_index, exp):
#     TR_start_time = exp_start_TR()[exp_list().index(exp)]
#     labels_to_remove = set()
#     for time in range(TR_start_time):
#         rps = regionprops(hypervolume_mask[time])
#         for rp in rps:
#             if rp.area != max([rp.area for rp in rps]):
#                 labels_to_remove.add(rp.label)
#     for time in time_index:
#         for label in labels_to_remove:
#             hypervolume_mask[time][hypervolume_mask[time] == label] = 0
#     return None

# # function removing the agglomerates with volume smaller than smallest_volume. volume can be intended as both 3D and 4D
# def remove_small_agglomerates(hypervolume_mask, smallest_volume, bincounts, time_index, progress_bar):
#     max_label = np.max(hypervolume_mask)
#     total_bincount = np.zeros(max_label+1)
#     for bincount in bincounts:
#         total_bincount[:len(bincount)] += bincount
#     bincount = np.bincount(hypervolume_mask.flatten())
#     for time in time_index:
#         hypervolume_mask[time][np.isin(hypervolume_mask[time], np.where(bincount < smallest_volume))] = 0
#         progress_bar.update()
#     return hypervolume_mask



# # function used to remove the agglomerates that are not present in neighboring slices
# def remove_isolated_agglomerates(hypervolume_mask, time_index, progress_bar):
#     for time in time_index:
#         current_slice = hypervolume_mask[time]
#         previous_slice = np.zeros_like(current_slice) if time==0 else hypervolume_mask[time-1] # previous_slice for first slice is set ad-hoc in order to avoid going out of bounds
#         next_slice = np.zeros_like(current_slice) if time == hypervolume_mask.shape[0]-1 else hypervolume_mask[time+1]  # next_slice for last slice is set ad-hoc in order to avoid going out of bounds
#         current_labels = [rp.label for rp in regionprops(current_slice)]
#         for current_label in current_labels:
#             if current_label not in previous_slice and current_label not in next_slice:
#                 current_slice[current_slice == current_label] = 0
#         hypervolume_mask[time] = current_slice
#         progress_bar.update()
#     return hypervolume_mask