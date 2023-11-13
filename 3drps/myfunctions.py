import numpy as np    
from numpy.lib.format import open_memmap
import seaborn as sns
import matplotlib.pyplot as plt           
from skimage.segmentation import clear_border                             
from skimage.measure import label, regionprops  
from skimage.morphology import erosion    
from tqdm import tqdm                               
import pandas as pd
import os



# function returning the list of the experiments names
def exp_list():
    return ['P28A_FT_H_Exp1', 'P28A_FT_H_Exp2', 'P28A_FT_H_Exp3_3', 'VCT5_FT_N_Exp3', 'VCT5_FT_N_Exp4', 'VCT5_FT_N_Exp5',
            'VCT5A_FT_H_Exp2', 'VCT5A_FT_H_Exp5']

def exp_start_TR():
    return [5, 6, 5, 4, 4, 4, 2, 2]

# function returning the area associated to the biggest agglomerate in the sequence
def find_biggest_area(sequence, threshold):
    sequence_mask = np.zeros_like(sequence, dtype=np.ushort)
    max_area = np.zeros(sequence.shape[0])
    for i in range(sequence.shape[0]):
        sequence_mask[i,:,:] = label(sequence[i,:,:] > threshold)
        areas = [rp.area for rp in regionprops(sequence_mask[i,:,:])]
        max_area[i] = np.max(areas)
    return np.mean(max_area)

# this is used to mask the inside of the external shell
# a method to find the correct slice is needed (sometimes agglomerates inside share the same label as the shell)
def create_shell_border(hypervolume, threshold, z=50):
    mask = label(hypervolume[z]>threshold)
    rps = regionprops(mask)
    labels = [rp.label for rp in rps]
    shell_inside = clear_border(np.ones_like(mask) - (mask==labels[np.argmax([rp.area for rp in rps])]))
    eroded_shell_inside = erosion(shell_inside, footprint=np.ones((5,5)))
    return 1 - (shell_inside - eroded_shell_inside)

# function removing the agglomerates with volume smaller than smallest_volume. volume can be intended as both 3D and 4D
def remove_small_agglomerates(sequence_mask, smallest_volume):
    bincount = np.bincount(sequence_mask.flatten())
    sequence_mask[np.isin(sequence_mask, np.where(bincount < smallest_volume))] = 0
    return sequence_mask

# function used to remove the agglomerates that are not present in neighboring slices
def remove_isolated_agglomerates(sequence_mask):
    for i in range(sequence_mask.shape[0]):
        current_slice = sequence_mask[i,:,:]
        if not i == 0:
            previous_slice = sequence_mask[i-1,:,:]
        else:
            previous_slice = np.zeros_like(current_slice)   # previous_slice for first slice is set ad-hoc in order to avoid going out of bounds
        if not i == sequence_mask.shape[0]-1:
            next_slice = sequence_mask[i+1,:,:]
        else:
            next_slice = np.zeros_like(current_slice)       # next_slice for last slice is set ad-hoc in order to avoid going out of bounds
        current_labels = [rp.label for rp in regionprops(current_slice)]
        for current_label in current_labels:
            if current_label not in previous_slice and current_label not in next_slice:
                current_slice[current_slice == current_label] = 0
        sequence_mask[i,:,:] = current_slice
    return sequence_mask

# function returning the path of the experiment given the experiment name and the OS
def OS_path(exp, OS):
    if OS=='Windows':
        return 'Z:/rot_datasets/' + exp
    elif OS=='MacOS':
        return '/Users/matteoventurelli/Documents/VS Code/MasterThesisData/' + exp
    elif OS=='Linux':
        return '/data/projects/whaitiri/Data/Data_Processing_July2022/rot_datasets/' + exp
    elif OS=='Tyrex':
        return 'U:/whaitiri/Data/Data_Processing_July2022/rot_datasets/' + exp
    else:
        raise ValueError('OS not recognized')



# function returning the threshold value that allows to segment the sequence in a way that the area of the biggest agglomerate is equal to target
def find_threshold(sequence, threshold=0, step=1, target=5400, delta=200, slices=5):
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
    return threshold



# function used to create the memmaps for the 4D volume and the 4D segmentation map
# if cropping is True, the volume is cropped in order to reduce the size of the memmap
def create_memmaps(exp, OS='Windows'):
    # create the 4D volume memmap
    hypervolume = open_memmap(os.path.join(OS_path(exp, OS), 'hypervolume.npy'), mode='r')
    # create the 4D segmentation mask memmap
    hypervolume_mask = open_memmap(os.path.join(OS_path(exp, OS), 'hypervolume_mask.npy'), dtype=np.ushort, mode='w+', shape=hypervolume.shape)
    return hypervolume, hypervolume_mask



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
        for i, new_label in enumerate(new_labels):
            current_mask[current_mask == new_label] = max_label + i + 1
    return current_mask



# function returning the 3D segmentation map given the 3D volume and the threshold
def segment3D(volume, threshold, smallest_volume, filtering):
    mask = label(volume > threshold)
    if filtering:
        mask = remove_small_agglomerates(mask, smallest_volume)
        mask = remove_isolated_agglomerates(mask)
    return mask



# function returning the sequence of segmented volumes given the sequence of volumes and the threshold
# if filtering3D is True, the agglomerates with volume smaller than smallest_3Dvolume are removed
# if filtering4D is True, the agglomerates with volume smaller than smallest_4Dvolume are removed
# if backward is True, backward propagation is performed
def segment4D(exp, OS='Windows', smallest_3Dvolume=50, filtering3D=True, offset=0):
    hypervolume, hypervolume_mask = create_memmaps(exp, OS) # creating the memmaps for the 4D volume and the 4D segmentation map
    time_index = range(hypervolume.shape[0]) # defining the time steps for the current experiment
    TR_start_time = exp_start_TR()[exp_list().index(exp)] # defining the time instant of the beginning of the thermal runaway
    previous_volume = hypervolume[0] # dealing with the first volume
    threshold = find_threshold(previous_volume) # evaluating the threshold on the first volume
    shell_border = create_shell_border(previous_volume, threshold) # creating the shell border mask (it is used to remove the agglomerates inside the shell
    # segmenting the first volume
    previous_mask = segment3D(previous_volume*shell_border, threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D) 
    # reassigning the labels after the filtering
    rps = regionprops(previous_mask)
    labels = [rp.label for rp in rps]
    labels = np.array(labels)[np.argsort([rp.area for rp in rps])][::-1]
    for new_label, old_label in enumerate(labels):
        previous_mask[previous_mask == old_label] = new_label + 1
    # writing the first mask in the hypervolume_mask memmap
    hypervolume_mask[0] = previous_mask

    # segmenting the remaining volumes and propagating the labels from previous volumes
    for time in tqdm(time_index[1:], desc=f'{exp} segmentation', position=offset, leave=False):
        current_volume = hypervolume[time]*shell_border if time < TR_start_time else hypervolume[time]
        current_mask = segment3D(current_volume, threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
        current_mask = propagate_labels(previous_mask, current_mask, forward=True)
        hypervolume_mask[time] = current_mask
        previous_mask = current_mask

    return hypervolume_mask



# function used to update the remove_list used to remove the agglomerates that are not present in each of the following n_steps bincounts
def update_remove_list(bincounts, bincount, remove_list, n_steps, i):
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
    return remove_list



# function used to remove the agglomerates that are not present in each of the following n_steps bincounts
def remove_inconsistent_4D_agglomerates (hypervolume_mask, bincounts, time_index, n_steps, progress_bar):
    # remove list is a list containing the labels that will be removed from start_time to end_time volumes in the format [label, start_time, end_time]
    remove_list = []
    max_label = np.max(hypervolume_mask)
    # looping through the bincounts related to each time instant and updating the remove_list
    for i, bincount in enumerate(bincounts[:-n_steps]):
        remove_list = update_remove_list(bincounts, bincount, remove_list, n_steps, i)
    # converting the remove_list to more useful format for looping
    # remove_array is a boolean array of shape (len(time_index), max_label) where each row contains True for the labels that will be removed in that time instant
    remove_array = np.zeros((len(time_index), int(max_label)), dtype=bool)
    for element in remove_list:
        remove_array[element[1]:element[2]+1, element[0]] = True
    # removing the labels
    for time in time_index:
        for label in np.where(remove_array[time])[0]:
            hypervolume_mask[time][hypervolume_mask[time] == label] = 0
        progress_bar.update()
    return None



# function used to remove small agglomerates in terms of 4D volume
def remove_small_4D_agglomerates (hypervolume_mask, bincounts, time_index, smallest_4Dvolume, progress_bar):
    # computation of the total bincount in the 4D mask
    max_label = np.max(hypervolume_mask)
    total_bincount = np.zeros(max_label+1)
    for bincount in bincounts:
        total_bincount[:len(bincount)] += bincount
    # removing the agglomerates with volume smaller than smallest_4Dvolume
    for time in time_index:
        for label in np.where(total_bincount < smallest_4Dvolume)[0]:
            if label < len(bincounts[time]):
                if bincounts[time][label] > 0:
                    hypervolume_mask[time][hypervolume_mask[time] == label] = 0
        progress_bar.update()
    return None



# function used to remove the agglomerates that appear before the thermal runaway
def remove_pre_TR_agglomerates(hypervolume_mask, time_index, exp):
    TR_start_time = exp_start_TR()[exp_list().index(exp)]
    labels_to_remove = set()
    for time in range(TR_start_time):
        rps = regionprops(hypervolume_mask[time])
        for rp in rps:
            if rp.area != max([rp.area for rp in rps]):
                labels_to_remove.add(rp.label)
    for time in time_index:
        for label in labels_to_remove:
            hypervolume_mask[time][hypervolume_mask[time] == label] = 0
    return None



# function used to rename the labels in the 4D segmentation map so that they are in the range [0, n_labels-1]
def rename_labels(hypervolume_mask, time_index, progress_bar):
    unique_labels = set()
    for t in time_index:
        for rp in regionprops(hypervolume_mask[t]):
            unique_labels.add(rp.label)
    unique_labels = np.array(list(unique_labels))
    total_labels = len(unique_labels)
    old_labels = unique_labels[unique_labels >= total_labels]
    new_labels = np.setdiff1d(np.arange(total_labels), unique_labels)
    timewise_labels = [[rp.label for rp in regionprops(hypervolume_mask[t])] for t in time_index]
    for time in time_index:
        for new_label, old_label in zip(new_labels, old_labels):
            if old_label in timewise_labels[time]:
                hypervolume_mask[time][hypervolume_mask[time] == old_label] = new_label
        progress_bar.update()
    return None



# function used to compute the 4-dimensional filtering of the segmentation map
def filtering4D(hypervolume_mask, exp, smallest_4Dvolume=250, n_steps=10, offset=0):
    bincounts = []
    time_index = range(hypervolume_mask.shape[0])
    progress_bar = tqdm(total=4*len(time_index), desc=f'{exp} bincounts computation', position=offset, leave=False)
    for t in time_index:
        bincounts.append(np.bincount(hypervolume_mask[t].flatten()))
        progress_bar.update()
    progress_bar.set_description(f'Exp {exp} inconsistent labels removal')
    remove_inconsistent_4D_agglomerates(hypervolume_mask, bincounts, time_index, n_steps, progress_bar)
    progress_bar.set_description(f'Exp {exp} small agglomerates removal')
    remove_small_4D_agglomerates(hypervolume_mask, bincounts, time_index, smallest_4Dvolume, progress_bar)
    remove_pre_TR_agglomerates(hypervolume_mask, time_index, exp)
    progress_bar.set_description(f'Exp {exp} labels renaming')
    rename_labels(hypervolume_mask, time_index, progress_bar)
    progress_bar.close()
    return None



# function used to compute the ratio between pixels and physical units in millimeters
# and the ratio between time steps and physical units in seconds
def find_geometry(hypervolume_mask, exp):
    if 'P28A' in exp:
        m_diameter = 18.6 # [mm]
    elif 'P28B' in exp:
        m_diameter = 18.6 # [mm]
    elif 'VCT5A' in exp:
        m_diameter = 18.35 # [mm]
    elif 'VCT5' in exp:
        m_diameter = 18.2 # [mm]
    else:
        raise ValueError('Experiment not recognized')
    # here I find the pixel width of the external shell
    rps = regionprops(hypervolume_mask[0,135])
    shell_index = np.argmax([rp.area for rp in rps])
    pixel_diameter = np.sqrt(rps[shell_index].area_bbox)
    m_z = 12 # total field of view in z direction [mm]
    pixel_z = 280 # total field of view in z direction in pixels
    fps = 20 # frames per second
    # computing the actual quantities
    xy_ratio = m_diameter/pixel_diameter
    z_ratio = m_z/pixel_z
    V_ratio = xy_ratio * xy_ratio * z_ratio
    t_ratio = 1/fps
    radius = pixel_diameter/2
    return xy_ratio, z_ratio, V_ratio, t_ratio, radius



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
        v = np.linalg.norm([vx, vy, vz])
        dVdt = (V - (df.iloc[prev_labels[label]]['V']))/t_ratio
    else:
        vx, vy, vz, v, dVdt = 0, 0, 0, 0, V/t_ratio
    # adding the row to the dataframe
    df = pd.concat([df, pd.DataFrame([[t, label, x, y, z, r, vx, vy, vz, v, V, dVdt, r_sect, z_sect]],
                                     columns=['t', 'label', 'x', 'y', 'z', 'r', 'vx', 'vy', 'vz', 'v', 'V', 'dVdt', 'r_sect', 'z_sect'])])
    return df



# function returning the dataframe containing the motion properties of the agglomerates
def motion_df(hypervolume_mask, exp, offset=0):
    df = pd.DataFrame(columns=['t', 'label', 'x', 'y', 'z', 'r', 'vx', 'vy', 'vz', 'v', 'V', 'dVdt', 'r_sect', 'z_sect'])
    # max_label = np.max(hypervolume_mask)
    n_time_instants = hypervolume_mask.shape[0]
    n_slices = hypervolume_mask.shape[1]
    # the ratios between pixels and physical units are computed
    xy_ratio, z_ratio, V_ratio, t_ratio, radius = find_geometry(hypervolume_mask, exp)
    center = np.array([0, 249.5, 249.5])
    rescale = np.array([z_ratio, xy_ratio, xy_ratio])
    # radii and slices are the values used to divide the volume in <steps> regions
    radii = np.linspace(0, radius*xy_ratio, 4)
    radii[3] = np.inf    # the last value is set to inf in order to avoid going out of bounds
    slices = np.linspace(0, n_slices*z_ratio, 4)
    slices[3] = np.inf   # the last value is set to inf in order to avoid going out of bounds
    r_sect_str = ['Core', 'Intermediate', 'External']
    z_sect_str = ['Top', 'Middle', 'Bottom'] # HERE I HAVE TO DOUBLE CHECK THE ORDER OF THE SECTIONS!!!
    current_labels = dict()
    # computing the actual quantities
    for time in tqdm(range(n_time_instants), desc=f'Exp {exp} dataframe computation', position=offset, leave=False):
        prev_labels = current_labels
        current_labels = dict()
        rps = regionprops(hypervolume_mask[time])
        labels = [rp.label for rp in rps]
        volumes = [(rp.area * V_ratio) for rp in rps] 
        centroids = [((rp.centroid - center) * rescale) for rp in rps]
        # converting the time index into seconds
        t = time * t_ratio
        for index, label in enumerate(labels):
            if label > 1:
                current_labels[label] = len(df)
                df = update_df(df, label, volumes[index], centroids[index], slices, radii, z_sect_str, r_sect_str, t_ratio, t, prev_labels)  
    return df


def plot_data(exp, OS, save=False):

    df = pd.read_csv(os.path.join(OS_path(exp, OS), 'motion_properties.csv'))
    plt.style.use('seaborn-v0_8-paper')
    time_axis = np.arange(len(np.unique(df['t'])))/20
    heigth = 3.5
    length = 5
    fig = plt.figure(figsize=(3*length, 4*heigth), dpi=150)
    subfigs = fig.subfigures(4, 1, hspace=0.3)

    # VOLUME
    subfigs[0].suptitle('Agglomerates volume vs time', y=1.1, fontsize=14)
    axs = subfigs[0].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df, x='t', y='V')
    axs[0].set_title('Whole battery')
    sns.lineplot(ax=axs[1], data=df, x='t', y='V', hue='r_sect')
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper right')
    sns.lineplot(ax=axs[2], data=df, x='t', y='V', hue='z_sect')
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper right')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Volume [$m^3$]')

    # SPEED
    subfigs[1].suptitle('Agglomerates speed vs time', y=1.1, fontsize=14)
    axs = subfigs[1].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df, x='t', y='v')
    axs[0].set_title('Whole battery')
    sns.lineplot(ax=axs[1], data=df, x='t', y='v', hue='r_sect')
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper right')
    sns.lineplot(ax=axs[2], data=df, x='t', y='v', hue='z_sect')
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper right')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Speed [$m/s$]')

    # EXPANSION RATE
    subfigs[2].suptitle('Agglomerates volume expansion rate vs time', y=1.1, fontsize=14)
    axs = subfigs[2].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df, x='t', y='dVdt')
    axs[0].set_title('Whole battery')
    sns.lineplot(ax=axs[1], data=df, x='t', y='dVdt', hue='r_sect')
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper right')
    sns.lineplot(ax=axs[2], data=df, x='t', y='dVdt', hue='z_sect')
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper right')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Volume expansion rate [$m^3/s$]')

    # DENSITY
    subfigs[3].suptitle('Agglomerates density vs time', y=1.1, fontsize=14)
    densityfig = subfigs[3].subfigures(1, 3, width_ratios=[1, 4, 1])
    axs = densityfig[1].subplots(1, 2)
    agg_number_r = pd.DataFrame(columns=['Time', 'Number', 'r_sect'])
    agg_number_z = pd.DataFrame(columns=['Time', 'Number', 'z_sect'])
    r_sect_list = ['Core', 'Intermediate', 'External']
    z_sect_list = ['Top', 'Middle', 'Bottom']
    for t in (np.unique(df['t'])):
        for r, z in zip(r_sect_list, z_sect_list):
            agg_number_r = pd.concat([agg_number_r, pd.DataFrame([[t, 0, r]], columns=['Time', 'Number', 'r_sect'])], ignore_index=True)
            agg_number_z = pd.concat([agg_number_z, pd.DataFrame([[t, 0, z]], columns=['Time', 'Number', 'z_sect'])], ignore_index=True)
    for i in range(len(df)):
        agg_number_r.loc[(agg_number_r['Time'] == df['t'][i]) & (agg_number_r['r_sect'] == df['r_sect'][i]), 'Number'] += 1
        agg_number_z.loc[(agg_number_z['Time'] == df['t'][i]) & (agg_number_z['z_sect'] == df['z_sect'][i]), 'Number'] += 1
    agg_number_r.loc[agg_number_r['r_sect'] == 'Intermediate', 'Number'] = agg_number_r.loc[agg_number_r['r_sect'] == 'Intermediate', 'Number'] / 3
    agg_number_r.loc[agg_number_r['r_sect'] == 'External', 'Number'] = agg_number_r.loc[agg_number_r['r_sect'] == 'External', 'Number'] / 5
    sns.lineplot(ax=axs[0], data=agg_number_r, x='Time', y='Number', hue='r_sect')
    axs[0].set_title('$r$ sections')
    sns.lineplot(ax=axs[1], data=agg_number_z, x='Time', y='Number', hue='z_sect')
    axs[1].set_title('$z$ sections')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        _ = ax.set_ylabel('Agglomerate density [a.u.]')

    if save:
        fig.savefig(f'Figures/{exp}.png', dpi=300, bbox_inches='tight')
    return None