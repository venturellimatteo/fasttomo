import myfunctions as mf
from skimage.io import imshow
import time as clock
import numpy as np
from tqdm import tqdm

def propagate_labels(previous_mask, current_mask, forward=True, biggest=False, propagation_threshold=10, verbose=False):
    print('Propagating labels profiling...')
    if forward:
        t1 = clock.perf_counter(), clock.process_time()
        max_label = np.max(previous_mask)
        current_mask[current_mask > 0] += max_label
        t2 = clock.perf_counter(), clock.process_time()
        print(f'Increasing labels:\nReal time: {t2[0] - t1[0]:.2f} s, CPU time: {t2[1] - t1[1]:.2f} s')
    t1 = clock.perf_counter(), clock.process_time()
    unique_labels, label_counts = np.unique(previous_mask, return_counts=True)
    t2 = clock.perf_counter(), clock.process_time()
    print(f'Computing and counting unique labels:\nReal time: {t2[0] - t1[0]:.2f} s, CPU time: {t2[1] - t1[1]:.2f} s')
    t1 = clock.perf_counter(), clock.process_time()
    ordered_labels = unique_labels[np.argsort(label_counts)]
    t2 = clock.perf_counter(), clock.process_time()
    print(f'Ordering labels:\nReal time: {t2[0] - t1[0]:.2f} s, CPU time: {t2[1] - t1[1]:.2f} s')
    for previous_slice_label in tqdm(ordered_labels):
        if previous_slice_label == 0:   # the background is not considered
            continue
        bincount = np.bincount(current_mask[previous_mask == previous_slice_label])
        if len(bincount) <= 1:      # if the agglomerate is not present in the current mask (i.e. bincount contains only background), the propagation is skipped
            continue
        bincount[0] = 0     # the background is not considered
        current_slice_label = np.argmax(bincount)
        current_mask[current_mask == current_slice_label] = previous_slice_label
        if not biggest:
            for current_slice_label in np.where(bincount > propagation_threshold)[0]:
                current_mask[current_mask == current_slice_label] = previous_slice_label
    if forward:
        new_labels = np.unique(current_mask[current_mask > np.max(previous_mask)])
        label_mapping = {new_label: max_label + i + 1 for i, new_label in enumerate(new_labels)}
        current_mask = np.vectorize(label_mapping.get)(current_mask, current_mask)
    return current_mask



OS = 'Linux'
smallest_3Dvolume = 25
exp = mf.exp_list()[0]
start_time = mf.exp_start_time()[mf.exp_list().index(exp)] + 20
skip180 = True
filtering3D = True

print('Loading previous volume...')
previous_volume = mf.load_volume(exp=exp, time=start_time, isImage=True, OS=OS)
threshold = mf.find_threshold(previous_volume)
print('Segmenting previous volume...')
tic = clock.time()
previous_mask = mf.segment3D(previous_volume, threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
toc = clock.time()
print(f'Previous volume segmented in {toc-tic:.2f} s')
print('Saving previous mask...')
mf.save_volume(volume=previous_mask, exp=exp, time=0, OS=OS)
print('Loading current volume...')
current_volume = mf.load_volume(exp=exp, time=start_time+2, isImage=True, OS=OS)
print('Segmenting current volume...')
tic = clock.time()
current_mask = mf.segment3D(current_volume, threshold, smallest_volume=smallest_3Dvolume, filtering=filtering3D)
toc = clock.time()
print(f'Current volume segmented in {toc-tic:.2f} s')
current_mask = propagate_labels(previous_mask, current_mask, forward=True)
print('Saving current mask...')
mf.save_volume(volume=current_mask, exp=exp, time=start_time+2, OS=OS)