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



def read_sequence(exp, time=0, slice=0, first_time=0, last_time=220, first_slice=20, last_slice=260, volume=True, win=False):

    if volume:
        test_image = imread(image_path(exp, time, first_slice, win))
        sequence = np.zeros((last_slice-first_slice, test_image.shape[0], test_image.shape[1]))
        for slice in range(first_slice, last_slice):
            image = imread(image_path(exp, time, slice, win))
            sequence[slice-first_slice,:,:] = image
    else:
        test_image = imread(image_path(exp, first_time, slice, win))
        sequence = np.zeros((last_time-first_time, test_image.shape[0], test_image.shape[1]))
        for time in range(first_time, last_time):
            image = imread(image_path(exp, time, slice, win))
            sequence[time-first_time,:,:] = image

    return sequence



def isAgglomerate(rps, i, smallest_area, eccentricity, volume=False):
    if rps[i].area < smallest_area:
        return False
    if not volume:
        if rps[i].eccentricity > eccentricity:
            return False
    return True



def segment(image, n_agglomerates=50, smallest_area=5, eccentricity=0.99):
    
    image = (image - np.min(image))/(np.max(image) - np.min(image))

    threshold = threshold_yen(image)
    mask = image > threshold
    mask_labeled = np.vectorize(label, signature='(n,m)->(n,m)')(mask)

    rps = regionprops(mask_labeled)
    areas = [r.area for r in rps]
    idxs = np.argsort(areas)[::-1]
    new_mask = np.zeros_like(mask_labeled)
    
    for j, i in enumerate(idxs[:n_agglomerates]):
        if isAgglomerate(rps, i, smallest_area, eccentricity):
            new_mask[tuple(rps[i].coords.T)] = j + 1
    return new_mask



def propagate_labels(mask, start=12, stop=0):
    for slice in range(start, mask.shape[0]-stop):
        previous_slice = mask[slice-1,:,:]
        current_slice = mask[slice,:,:]
        current_slice[current_slice > 0] = current_slice[current_slice > 0] + np.max(previous_slice)
        flag = False

        for previous_slice_label in np.unique(previous_slice):
            if previous_slice_label == 0:
                continue
            previous_slice_region = previous_slice == previous_slice_label
            overlap = current_slice * previous_slice_region
            unique_labels = np.unique(overlap)
            
            for _, current_slice_label in enumerate(unique_labels):
                if current_slice_label == 0:
                    continue
                temp = np.array([[previous_slice_label, current_slice_label, len(overlap[overlap == current_slice_label])]]).T
                if not flag:
                    mapping = temp
                    flag = True
                else:
                    mapping = np.append(mapping, temp, axis=1)
        
        for current_slice_label in np.unique(mapping[1,:]):
            temp = mapping[:,mapping[1,:] == current_slice_label]
            previous_slice_label = temp[0, np.argmax(temp[2,:])]
            current_slice[current_slice == current_slice_label] = previous_slice_label
        
        mask[slice,:,:] = current_slice
            
    return mask


# the biggest agglomerate has to be removed since it is the external shell
def explore_volume(exp, start_time, end_time, first_slice, last_slice, time_steps_number, step, win):
    
    time_steps = np.arange(start_time, min(start_time+step*time_steps_number, 220), time_steps_number, dtype=int)
    temp_area = np.zeros((len(time_steps), last_slice-first_slice))
    temp_number = np.zeros_like(temp_area)

    for t, time in enumerate(time_steps):
        sequence = read_sequence(exp, time=time, first_slice=first_slice, last_slice=last_slice,  volume=True, win=win)
        segmented_image = (np.zeros_like(sequence)).astype(int)

        for z in range(sequence.shape[0]):
            segmented_image[z,:,:] = segment(sequence[z,:,:])
        new_segmented_image = propagate_labels(segmented_image)

        for z in range(sequence.shape[0]):
            rps = regionprops(new_segmented_image[z,:,:])
            areas = [r.area for r in rps]
            areas.pop(np.max(areas))
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
            rotated_image[i, N-1-j] = image[M-1-i, j]
            


def explore_slice(exp, start_time, end_time, first_slice, last_slice, volumes_number, win):

    slices = np.linspace(first_slice, last_slice, volumes_number, dtype=int)
    temp_area = np.zeros((len(slices), end_time-start_time)) 
    temp_number = np.zeros_like(temp_area)

    for z, slice in enumerate(slices):
        sequence = read_sequence(exp, slice=slice, first_slice=first_slice, last_slice=last_slice,  volume=True, win=win)
        for i in range(0, sequence.shape[0], 2):
            sequence[i,:,:] = rotate180(sequence[i,:,:])
        segmented_image = (np.zeros_like(sequence)).astype(int)

        for t in range(start_time, end_time):
            segmented_image[t,:,:] = segment(sequence[t,:,:])
        new_segmented_image = propagate_labels(segmented_image)

        for t in range(start_time, end_time):
            rps = regionprops(new_segmented_image[t,:,:])
            areas = [r.area for r in rps]
            areas.pop(np.max(areas))
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