import numpy as np
from numpy.lib.format import open_memmap
import myfunctions as mf
import os
from tqdm import tqdm

def OS_path(exp, OS, isrec=False):
    if OS=='Windows':
        if isrec:
            return 'Z:/rot_datasets/selected_vol/' + exp
        else:
            return 'Z:/rot_datasets/' + exp
    elif OS=='MacOS':
        return '../../MasterThesisData/' + exp
    elif OS=='Linux':
        if isrec:
            return '/data/projects/whaitiri/Data/Data_Processing_July2022/rot_datasets/selected_vol/' + exp
        else:
            return '/data/projects/whaitiri/Data/Data_Processing_July2022/rot_datasets/' + exp
    elif OS=='Tyrex':
        return 'U:/whaitiri/Data/Data_Processing_July2022/rot_datasets/' + exp
    else:
        raise ValueError('OS not recognized')

def volume_path(exp, time, isrec, isImage=True, OS='Windows'):
    flag = mf.exp_flag()[mf.exp_list().index(exp)]
    vol = '0050' if flag else '0100'
    folder_name = 'entry' + str(time).zfill(4) + '_no_extpag_db' + vol + '_vol'
    volume_name = 'volume_v2.npy' if isImage else 'segmented.npy'
    return os.path.join(OS_path(exp, OS, isrec), folder_name, volume_name)

exp = mf.exp_list()[2]
start_time = mf.exp_start_time()[mf.exp_list().index(exp)]
end_time = 220
skip180=True
OS = 'Linux'
rec = range(96,123+1)

time_steps = range(start_time, end_time+1, 2) if skip180 else range(start_time, end_time+1)
shape = (len(time_steps), 270, 500, 500)
hypervolume = open_memmap(os.path.join(OS_path(exp, OS), 'hypervolume.npy'), dtype=np.half, mode='w+', shape=shape)
for t, time in tqdm(enumerate(time_steps), desc='Loading hypervolume memmap', total=len(time_steps)):
    volume = open_memmap(volume_path(exp=exp, time=time, OS=OS, isImage=True, isrec=(time in rec)), mode='r')
    hypervolume[t,:,:,:] = volume[10:,208:708,244:744]