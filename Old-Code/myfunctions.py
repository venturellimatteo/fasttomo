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


# def old_propagate_labels(previous_mask, current_mask, forward=True, propagation_threshold=10, verbose=False, leave=False):
#     if forward:
#         max_label = np.max(previous_mask)
#         current_mask[current_mask > 0] += max_label
#     unique_labels, label_counts = np.unique(previous_mask, return_counts=True)
#     ordered_labels = unique_labels[np.argsort(label_counts)]
#     for previous_slice_label in iterator(ordered_labels, verbose=verbose, desc='Label propagation', leave=leave):
#         if previous_slice_label == 0:   # the background is not considered
#             continue
#         bincount = np.bincount(current_mask[previous_mask == previous_slice_label])
#         if len(bincount) <= 1:  # if the agglomerate is not present in the current mask (i.e. bincount contains only background), the propagation is skipped
#             continue
#         bincount[0] = 0     # the background is not considered
#         current_slice_label = np.argmax(bincount)
#         current_mask[current_mask == current_slice_label] = previous_slice_label
#         for current_slice_label in np.where(bincount > propagation_threshold)[0]:
#             current_mask[current_mask == current_slice_label] = previous_slice_label
#     if forward:
#         new_labels = np.unique(current_mask[current_mask > np.max(previous_mask)])
#         for i, new_label in enumerate(new_labels):
#             current_mask[current_mask == new_label] = max_label + i + 1
#     return current_mask