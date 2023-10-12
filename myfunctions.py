import numpy as np                                  # type: ignore
from skimage.measure import label, regionprops      # type: ignore
from skimage.io import imread                       # type: ignore
from skimage.filters import threshold_yen           # type: ignore
import glob
from dataclasses import dataclass



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
    return [112, 99, 90, 90108, 127, 130, 114, 99, 105, 104, 115, 155, 70, 54, 7, 71, 52, 4, 66, 66]



def read_sequence(exp, time=0, slice=0, volume=True, win=False, first_time=0, last_time=220, first_slice=20, last_slice=260):
    if win:
        if volume:
            path = 'Z:/Reconstructions/' + exp + '/entry' + str(time).zfill(4) + '_no_extpag_db0100_vol/'
        else:
            path = 'Z:/Reconstructions/' + exp + '/slice ' + str(slice) + '/'
    else:
        if volume:
            path = '../MasterThesisData/' + exp + '/entry' + str(time).zfill(4) + '_no_extpag_db0100_vol/'
        else:
            path = '../MasterThesisData/' + exp + '/slice ' + str(slice) + '/'
        
    image = imread(path+'entry' + str(time).zfill(4) + '_no_extpag_db0100_vol_' + str(slice).zfill(6) + '.tiff')

    if volume:
        sequence = np.zeros((last_slice-first_slice, image.shape[0], image.shape[1]))
        for i in range(first_slice, last_slice):
            image = imread(path+'entry'+str(time).zfill(4)+'_no_extpag_db0100_vol_'+str(i).zfill(6)+'.tiff')
            sequence[i-first_slice,:,:] = image
    else:
        sequence = np.zeros((last_time-first_time, image.shape[0], image.shape[1]))
        for i in range(first_time, last_time):
            image = imread(path+'entry'+str(i).zfill(4)+'_no_extpag_db0100_vol_'+str(slice).zfill(6)+'.tiff')
            sequence[i-first_time,:,:] = image

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



def explore_volume(exp, start_time, time_steps_number, first_slice, last_slice, step):
    
    time_steps = np.arange(start_time, min(start_time+step*time_steps_number, 220), time_steps_number, dtype=int)
    temp_area = np.zeros((len(time_steps), last_slice-first_slice))
    temp_number = np.zeroslike(temp_area)

    for time in time_steps:
        sequence = read_sequence(exp, time=time, volume=True, first_slice=first_slice, last_slice=last_slice)
        segmented_image = (np.zeros_like(sequence)).astype(int)

        for z in range(sequence.shape[0]):
            segmented_image[z,:,:] = segment(sequence[z,:,:])
        new_segmented_image = propagate_labels(segmented_image)

        for z in range(sequence.shape[0]):
            rps = regionprops(new_segmented_image[z,:,:])
            areas = [r.area for r in rps]
            temp_area[time-time_steps[0], z] = np.mean(areas)
            temp_number[time-time_steps[0], z] = len(areas)

    volume_area = feature(np.mean(temp_area, axis=1), np.mean(temp_area), np.std(temp_area))
    volume_number = feature(np.mean(temp_number, axis=1), np.mean(temp_number), np.std(temp_number))

    return volume_number, volume_area




def explore_slice(exp, start_time, volumes_number, first_slice, last_slice):

    return slice_number, slice_area, slice_stability_time



def explore_experiment(exp, time_steps_number=5, volumes_number=5, first_slice=20, last_slice=260, step=5):

    start_time = exp_start_time()[exp_list().index(exp)]

    volume_number, volume_area = explore_volume(exp, start_time, time_steps_number, first_slice, last_slice)
    slice_number, slice_area, slice_stability_time = explore_slice(exp, start_time, volumes_number, first_slice, last_slice)
    
    return experiment(exp, volume_number, volume_area, slice_number, slice_area, slice_stability_time)