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

def isAgglomerate(rps, smallest_area, area_ratio, axis_ratio):
    if rps.area < smallest_area:
        return False
    if rps.area_convex > area_ratio*rps.area_filled:
            return False
    if rps.major_axis_length > axis_ratio*rps.minor_axis_length:
        return False
    return True


def segment(image, threshold, n_agglomerates=50, smallest_area=10, area_ratio=2, axis_ratio=4, all=True):

    mask = image > threshold
    mask_labeled = np.vectorize(label, signature='(n,m)->(n,m)')(mask)

    rps = regionprops(mask_labeled)
    areas = [r.area for r in rps]
    idxs = np.argsort(areas)[::-1]
    new_mask = np.zeros_like(mask_labeled)
    
    if all:
        n_agglomerates = len(idxs)
        
    for j, i in enumerate(idxs[:n_agglomerates]):
        # if isAgglomerate(rps[i], smallest_area, area_ratio, axis_ratio):
        new_mask[tuple(rps[i].coords.T)] = j + 1
    return new_mask, areas[idxs[0]]



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