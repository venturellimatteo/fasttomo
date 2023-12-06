import numpy as np    
from numpy.lib.format import open_memmap                           
from skimage.measure import label, regionprops, marching_cubes
from skimage.morphology import erosion, dilation, ball    
from tqdm import tqdm                               
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   
import os



# functions returning the lists of the experiments names and related features
def exp_list():
    return ['P28A_FT_H_Exp1', 'P28A_FT_H_Exp2', 'P28A_FT_H_Exp3_3', 'P28A_FT_H_Exp4_2', 'P28B_ISC_FT_H_Exp2', 'VCT5_FT_N_Exp1', 
            'VCT5_FT_N_Exp3', 'VCT5_FT_N_Exp4', 'VCT5_FT_N_Exp5', 'VCT5A_FT_H_Exp2', 'VCT5A_FT_H_Exp5']

def bad_exp_list():
    return ['P28B_ISC_FT_H_Exp3','P28B_ISC_FT_H_Exp4','P28B_ISC_FT_H_Exp4_2','P28B_ISC_FT_H_Exp5','VCT5A_FT_H_Exp1','VCT5A_FT_H_Exp4']

def exp_start_TR_list():
    return [2, 2, 2, 6, 11, 4, 5, 4, 4, 1, 4]

def labels_to_remove_list():
    return [[525], [], [370, 390], [], [], [], [], [], [], [], []]

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
    return 

# function returning the area associated to the biggest agglomerate in the sequence
def find_biggest_area(sequence, threshold):
    sequence_mask = segment3D(sequence, threshold)
    areas = [rp.area for rp in regionprops(sequence_mask)]
    return np.max(areas)/sequence_mask.shape[0]

def remove_small_agglomerates(hypervolume_mask, smallest_volume):
    max_label = np.max(hypervolume_mask)
    bincount = np.zeros(max_label+1)
    for time in range(hypervolume_mask.shape[0]):
        b = np.bincount(hypervolume_mask[time].flatten())
        bincount[:len(b)] += b
    lookup_table = np.where(bincount < smallest_volume, 0, np.arange(len(bincount)))
    for time in range(hypervolume_mask.shape[0]):
        hypervolume_mask[time] = np.take(lookup_table, hypervolume_mask[time])
    return 

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
    return 

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
    for time in time_index:
        hypervolume_mask[time] = np.take(lookup_table, hypervolume_mask[time])
    return 

# function returning the 3D segmentation map given the 3D volume and the threshold
def segment3D(volume, threshold, filtering3D=True, smallest_3Dvolume=50):
    mask = np.greater(volume, threshold)
    mask = label(np.logical_and(mask, dilation(erosion(mask, ball(1)), ball(4))))
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
            step = step/2 if not add else step  # step is halved every time the direction of the step is changed
            threshold += step
            add = True
        elif current_area < target - delta:     # if the area is smaller than target, the threshold is decreased in order to increase the area
            step = step/2 if add else step              
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
    ordered_labels = np.array([rp.label for rp in rps])[np.argsort([rp.area for rp in rps])]
    propagation_map = dict()

    for previous_mask_label in ordered_labels:
        propagation_map = update_map(current_mask, previous_mask, previous_mask_label, propagation_map)

    # START OF NEW CODE
    lookup_table = np.arange(np.max(current_mask)+1)
    for current_slice_label, previous_mask_label in propagation_map.items():
        lookup_table[current_slice_label] = previous_mask_label[0]
    if forward:
        new_labels = np.unique(current_mask[current_mask > max_label])
        lookup_table[new_labels] = np.arange(len(new_labels)) + max_label + 1
    current_mask = np.take(lookup_table, current_mask)
    # END OF NEW CODE
    # for current_slice_label, previous_mask_label in propagation_map.items():
    #     current_mask[current_mask == current_slice_label] = previous_mask_label[0]
    # if forward:
    #     new_labels = np.unique(current_mask[current_mask > max_label])
    #     lookup_table = np.arange(np.max(new_labels)+1)
    #     lookup_table[new_labels] = np.arange(len(new_labels)) + max_label + 1
    #     current_mask = np.take(lookup_table, current_mask)
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
    threshold = find_threshold(previous_volume, target=(6500 if exp=='VCT5A_FT_H_Exp2' else 6800))

    # segmenting the first volume
    update_pb(progress_bar, 'Segmenting first volume')
    previous_mask = segment3D(previous_volume, threshold, filtering3D, smallest_3Dvolume)

    # reassigning the labels after the filtering
    update_pb(progress_bar, 'Reassigning labels')
    rps = regionprops(previous_mask)
    old_labels = np.array([rp.label for rp in rps])[np.argsort([rp.area for rp in rps])][::-1]
    lookup_table = np.arange(np.max(old_labels)+1)
    lookup_table[old_labels] = np.arange(len(old_labels))+1
    previous_mask[:] = np.take(lookup_table, previous_mask)

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
    remove_isolated_agglomerates(hypervolume_mask)
    update_pb(progress_bar, 'Small agglomerates removal')
    remove_small_agglomerates(hypervolume_mask, smallest_4Dvolume)
    # remove_pre_TR_agglomerates(hypervolume_mask, time_index, exp)
    update_pb(progress_bar, 'Labels renaming')
    rename_labels(hypervolume_mask, time_index)
    progress_bar.close()
    return None



# function used to manually remove the agglomerates that appear before the thermal runaway and the agglomerates due to artifacts
def manual_filtering(hypervolume_mask, exp, offset=0):
    progress_bar = tqdm(total=2, desc=f'{exp} manual filtering', position=offset, leave=False)
    progress_bar.set_postfix_str('Finding max label')
    max_label = np.max(hypervolume_mask)
    start_TR = exp_start_TR_list()[exp_list().index(exp)]
    labels_to_remove = np.union1d(np.setdiff1d(np.unique(hypervolume_mask[:start_TR]), np.array([0, 1])),
                                  np.array(labels_to_remove_list()[exp_list().index(exp)])).astype(np.ushort)
    lookup_table = np.arange(max_label+1)
    lookup_table[labels_to_remove] = 0
    update_pb(progress_bar, 'Removing labels')
    for time in range(hypervolume_mask.shape[0]):
        hypervolume_mask[time] = np.take(lookup_table, hypervolume_mask[time])
    #hypervolume_mask[:] = np.take(lookup_table, hypervolume_mask)
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
        vx, vy, vxy, vz, v, dVdt = 0, 0, 0, 0, 0, V/t_ratio
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
    for time in tqdm(range(n_time_instants), desc=f'{exp} dataframe computation', position=offset, leave=False):
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



# function returning the dataframe containing the motion properties of the agglomerates and the related dataframes
def load_dfs(exp, OS):
    df = pd.read_csv(os.path.join(OS_path(exp, OS), 'motion_properties.csv'))

    # creating 'total' dataframe
    grouped_df = df.groupby('t').size().reset_index(name='N')
    df_tot = pd.merge(df[['t']], grouped_df, on='t', how='left').drop_duplicates()
    df_tot['V'] = df.groupby('t')['V'].sum().values
    df_tot['V/N'] = df_tot['V'] / df_tot['N'] # np.maximum(df_tot['N'].values, 1)
    df_tot['dVdt'] = df.groupby('t')['dVdt'].sum().values
    df_tot.fillna(0, inplace=True)
    df_tot.reset_index(drop=True, inplace=True)

    # creating 'r' dataframe
    r_sect_list = ['Core', 'Intermediate', 'External']
    df_r = pd.DataFrame(columns=['t', 'r_sect', 'V/N'])
    df_r['t'] = np.repeat(df['t'].unique(), len(r_sect_list))
    df_r['r_sect'] = np.tile(r_sect_list, len(df['t'].unique()))
    grouped_N = df.groupby(['t', 'r_sect']).size().reset_index(name='N')
    grouped_V = df.groupby(['t', 'r_sect'])['V'].sum().reset_index(name='V')
    grouped_dVdt = df.groupby(['t', 'r_sect'])['dVdt'].sum().reset_index(name='dVdt')
    df_r = pd.merge(df_r, grouped_N, on=['t', 'r_sect'], how='left')
    df_r = pd.merge(df_r, grouped_V, on=['t', 'r_sect'], how='left')
    df_r = pd.merge(df_r, grouped_dVdt, on=['t', 'r_sect'], how='left')
    df_r.fillna(0, inplace=True)
    df_r['V/N'] = df_r['V'] / np.maximum(df_r['N'].values, 1)
    df_r.reset_index(drop=True, inplace=True)

    # creating 'z' dataframe
    z_sect_list = ['Bottom', 'Middle', 'Top']
    df_z = pd.DataFrame(columns=['t', 'z_sect', 'V/N'])
    df_z['t'] = np.repeat(df['t'].unique(), len(z_sect_list))
    df_z['z_sect'] = np.tile(z_sect_list, len(df['t'].unique()))
    grouped_N = df.groupby(['t', 'z_sect']).size().reset_index(name='N')
    grouped_V = df.groupby(['t', 'z_sect'])['V'].sum().reset_index(name='V')
    grouped_dVdt = df.groupby(['t', 'z_sect'])['dVdt'].sum().reset_index(name='dVdt')
    df_z = pd.merge(df_z, grouped_N, on=['t', 'z_sect'], how='left')
    df_z = pd.merge(df_z, grouped_V, on=['t', 'z_sect'], how='left')
    df_z = pd.merge(df_z, grouped_dVdt, on=['t', 'z_sect'], how='left')
    df_z.fillna(0, inplace=True)
    df_z['V/N'] = df_z['V'] / np.maximum(df_z['N'].values, 1)
    df_z.reset_index(drop=True, inplace=True)

    return df, df_tot, df_r, df_z, r_sect_list, z_sect_list



# function used to plot the data contained in the dataframes
def plot_data(exp, OS, offset=0, save=True):

    progress_bar = tqdm(total=5, desc=f'{exp} drawing plots', position=offset, leave=False)
    length, heigth = 5, 3.5
    fig = plt.figure(figsize=(3*length, 5*heigth), dpi=150)
    subfigs = fig.subfigures(5, 1, hspace=0.3)
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette(sns.color_palette(['#3cb44b', '#bfef45']))
    palette2 = ['#e6194B', '#f58231', '#ffe119'] # ['#bce4b5', '#56b567', '#05712f'] #
    palette3 = ['#000075', '#4363d8', '#42d4f4'] # ['#fdc692', '#f67824', '#ad3803'] # 

    df, df_tot, df_r, df_z, r_sect_list, z_sect_list = load_dfs(exp, OS)
    time_axis = np.arange(len(np.unique(df['t'])))/20

    # Agglomerates total volume vs time
    subfigs[0].suptitle('Agglomerates total volume vs time', y=1.1, fontsize=14)
    axs = subfigs[0].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df_tot, x='t', y='V')
    sns.lineplot(ax=plt.twinx(ax=axs[0]), data=df_tot, x='t', y='N', color='#bfef45')
    axs[0].set_title('Whole battery')
    axs[0].legend(handles=[Line2D([], [], marker='_', color='#3cb44b', label='Volume [$mm^3$]'), Line2D([], [], marker='_', color='#bfef45', label='Number of agglomerates')], loc='upper left')
    sns.lineplot(ax=axs[1], data=df_r, x='t', y='V', hue='r_sect', hue_order=r_sect_list, palette=palette2)
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper right')
    sns.lineplot(ax=axs[2], data=df_z, x='t', y='V', hue='z_sect', hue_order=z_sect_list, palette=palette3)
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper right')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Volume [$mm^3$]')
    progress_bar.update()

    # Agglomerates mean volume vs time
    subfigs[1].suptitle('Agglomerates average volume vs time', y=1.1, fontsize=14)
    axs = subfigs[1].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df_tot, x='t', y='V/N')
    axs[0].set_title('Whole battery')
    sns.lineplot(ax=axs[1], data=df_r, x='t', y='V/N', hue='r_sect', hue_order=r_sect_list, palette=palette2)
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper right')
    sns.lineplot(ax=axs[2], data=df_z, x='t', y='V/N', hue='z_sect', hue_order=z_sect_list, palette=palette3)
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper right')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Volume [$mm^3$]')
    progress_bar.update()

    # Agglomerates total volume expansion rate vs time
    subfigs[2].suptitle('Agglomerates total volume expansion rate vs time', y=1.1, fontsize=14)
    axs = subfigs[2].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df_tot, x='t', y='dVdt')
    axs[0].set_title('Whole battery')
    sns.lineplot(ax=axs[1], data=df_r, x='t', y='dVdt', hue='r_sect', hue_order=r_sect_list, palette=palette2)
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper right')
    sns.lineplot(ax=axs[2], data=df_z, x='t', y='dVdt', hue='z_sect', hue_order=z_sect_list, palette=palette3)
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper right')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Volume expansion rate [$mm^3/s$]')
    progress_bar.update()

    # Agglomerates speed vs time
    subfigs[3].suptitle('Agglomerates speed vs time', y=1.1, fontsize=14)
    axs = subfigs[3].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df, x='t', y='v')
    axs[0].set_title('Modulus')
    sns.lineplot(ax=axs[1], data=df, x='t', y='vxy', color='#f58231')
    axs[1].set_title('$xy$ component')
    sns.lineplot(ax=axs[2], data=df, x='t', y='vz', color='#4363d8')
    axs[2].set_title('$z$ component')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Speed [$mm/s$]')
    progress_bar.update()

    # Agglomerates density vs time
    battery_volume = np.pi * (0.5 * find_diameter(exp))**2 * 0.012

    subfigs[4].suptitle('Agglomerates density vs time', y=1.1, fontsize=14)
    axs = subfigs[4].subplots(1, 3, sharey=True)
    df_tot['N'] = df_tot['N'] / (battery_volume)
    sns.lineplot(ax=axs[0], data=df_tot, x='t', y='N')
    axs[0].set_title('Whole battery')
    df_r.loc[df_r['r_sect'] == 'Core', 'N'] = df_r.loc[df_r['r_sect'] == 'Core', 'N'] / (battery_volume/9)
    df_r.loc[df_r['r_sect'] == 'Intermediate', 'N'] = df_r.loc[df_r['r_sect'] == 'Intermediate', 'N'] / (battery_volume*3/9)
    df_r.loc[df_r['r_sect'] == 'External', 'N'] = df_r.loc[df_r['r_sect'] == 'External', 'N'] / (battery_volume*5/9)
    sns.lineplot(ax=axs[1], data=df_r, x='t', y='N', hue='r_sect', hue_order=r_sect_list, palette=palette2)
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper left')
    df_z['N'] = df_z['N'] / (battery_volume*3/9)
    sns.lineplot(ax=axs[2], data=df_z, x='t', y='N', hue='z_sect', hue_order=z_sect_list, palette=palette3)
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper left')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        _ = ax.set_ylabel('Agglomerate density [cm$^{-3}$]')
    progress_bar.update()

    if save:
        fig.savefig(os.path.join(OS_path(exp, OS), 'motion_properties.png'), dpi=300, bbox_inches='tight')

    progress_bar.close()
    return None