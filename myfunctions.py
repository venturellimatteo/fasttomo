import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from skimage.filters import threshold_yen
import glob



def read_sequence(exp, time=0, slice=0, volume=True, win=False):
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
    files_number = len(glob.glob1(path,"*.tiff"))
    sequence = np.zeros((files_number,image.shape[0],image.shape[1]))

    for i in range(files_number):
        if volume:
            image = imread(path+'entry'+str(time).zfill(4)+'_no_extpag_db0100_vol_'+str(i).zfill(6)+'.tiff')
        else:
            image = imread(path+'entry'+str(i).zfill(4)+'_no_extpag_db0100_vol_'+str(slice).zfill(6)+'.tiff')
        sequence[i,:,:] = image
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