import numpy as np                                  # type: ignore
from skimage.measure import label, regionprops      # type: ignore
from skimage.io import imread                       # type: ignore
from skimage.filters import threshold_yen           # type: ignore
from dataclasses import dataclass
import glob
import os



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





def image_path(exp, time, slice, win=False):
    if win:
        path = 'Z:/Reconstructions/' + exp
    else:
        path = '../MasterThesisData/' + exp
    folder_name = 'entry' + str(time).zfill(4) + '_no_extpag_db0100_vol'
    image_name = 'entry' + str(time).zfill(4) + '_no_extpag_db0100_vol_' + str(slice).zfill(6) + '.tiff'
    return os.path.join(path, folder_name, image_name)



def read_sequence(exp, time=0, slice=0, start_time=0, end_time=220, first_slice=20, last_slice=260, volume=True, win=False):

    if volume:
        test_image = imread(image_path(exp, time, first_slice, win))
        sequence = np.zeros((last_slice-first_slice, test_image.shape[0], test_image.shape[1]))
        for slice in range(first_slice, last_slice):
            image = imread(image_path(exp, time, slice, win))
            sequence[slice-first_slice,:,:] = image
    else:
        test_image = imread(image_path(exp, start_time, slice, win))
        sequence = np.zeros((end_time-start_time, test_image.shape[0], test_image.shape[1]))
        for time in range(start_time, end_time):
            image = imread(image_path(exp, time, slice, win))
            sequence[time-start_time,:,:] = image

    return sequence



def mask(image, threshold):
    return np.vectorize(label, signature='(n,m)->(n,m)')(image > threshold)



def find_biggest_area(sequence, threshold):

    area = np.zeros(sequence.shape[0])
    for i in range(sequence.shape[0]):
        image = sequence[i,:,:]
        mask_labeled = mask(image, threshold)
        rps = regionprops(mask_labeled)
        areas = [r.area for r in rps]
        area[i] = np.max(areas)

    return np.mean(area)



def find_threshold(sequence, threshold=0, step=1, target=5000, delta=50):

    flag = False
    add = False
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

    return threshold


# the problem is that the label propagation is overriden because of the order in which the labels appear
# one solution would be to do the label update from the smallest area to the largest one
# this problem is solved but still there's not consistent labeling everywhere
def propagate_labels(previous_mask, current_mask, forward=True):
    if forward:
        current_mask[current_mask > 0] = current_mask[current_mask > 0] + np.max(previous_mask)
    # here I order np.unique(previous_mask) from the smallest to the largest area
    ordered_labels = np.argsort([r.area for r in regionprops(previous_mask)])
    for previous_slice_label in ordered_labels:
        if previous_slice_label == 0:
            continue
        bincount = np.bincount(current_mask[previous_mask == previous_slice_label])
        if len(bincount) <= 1:
            continue
        bincount[0] = 0
        current_slice_label = np.argmax(bincount)
        current_mask[current_mask == current_slice_label] = previous_slice_label
    return current_mask



def remove_small_agglomerates(sequence_mask, smallest_volume):

    return sequence_mask



def segment(sequence, threshold, smallest_volume=10):

    sequence_mask = np.zeros_like(sequence).astype(int)
    sequence_mask[0,:,:] = mask(sequence[0,:,:], threshold)

    # masking of current slice and forward propagation from the first slice
    for i in range(1, sequence.shape[0]):
        sequence_mask[i,:,:] = mask(sequence[i,:,:], threshold)
        sequence_mask[i,:,:] = propagate_labels(sequence_mask[i-1,:,:], sequence_mask[i,:,:], forward=True)

    # backward propagation from the last slice
    '''for i in range(sequence_mask.shape[0]-1, 0, -1):
        sequence_mask[i-1,:,:] = propagate_labels(sequence_mask[i,:,:], sequence_mask[i-1,:,:], forward=False)'''
    
    # removal of the agglomerates with volume smaller than smallest_volume
    sequence_mask = remove_small_agglomerates(sequence_mask, smallest_volume)
    return sequence_mask



# the biggest agglomerate has to be removed since it is the external shell
def explore_volume(exp, start_time, end_time, first_slice, last_slice, time_steps_number, step, win):
    
    time_steps = np.arange(start_time, min(start_time+step*time_steps_number, end_time), time_steps_number, dtype=int)
    temp_area = np.zeros((len(time_steps), last_slice-first_slice))
    temp_number = np.zeros_like(temp_area)

    for t, time in enumerate(time_steps):
        sequence = read_sequence(exp, time=time, first_slice=first_slice, last_slice=last_slice, volume=True, win=win)
        segmented_image = (np.zeros_like(sequence)).astype(int)

        for z in range(sequence.shape[0]):
            segmented_image[z,:,:] = segment(sequence[z,:,:])
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



def rotate180(image):
    rotated_image = np.zeros_like(image)
    M = image.shape[0]
    N = image.shape[1]
    for i in range(M):
        for j in range(N):
            rotated_image[i,j] = image[M-1-i, N-1-j]
    return rotated_image
            


def explore_slice(exp, start_time, end_time, first_slice, last_slice, volumes_number, win):

    slices = np.linspace(first_slice, last_slice, volumes_number, dtype=int)
    temp_area = np.zeros((len(slices), end_time-start_time)) 
    temp_number = np.zeros_like(temp_area)

    for z, slice in enumerate(slices):
        sequence = read_sequence(exp, slice=slice, start_time=start_time, end_time=end_time,  volume=False, win=win)
        for i in range(0, sequence.shape[0], 2):
            sequence[i,:,:] = rotate180(sequence[i,:,:])
        segmented_image = (np.zeros_like(sequence)).astype(int)

        for t in range(sequence.shape[0]):
            segmented_image[t,:,:] = segment(sequence[t,:,:])
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