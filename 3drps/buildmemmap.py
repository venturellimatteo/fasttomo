import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import os

def OS_path(exp, OS, isrec=False):
    if OS=='Windows':
        if isrec:
            return 'Z:/rot_datasets/selected_vol/' + exp
        else:
            return 'Z:/rot_datasets/' + exp
    elif OS=='Linux':
        if isrec:
            return '/data/projects/whaitiri/Data/Data_Processing_July2022/rot_datasets/selected_vol/' + exp
        else:
            return '/data/projects/whaitiri/Data/Data_Processing_July2022/rot_datasets/' + exp
    else:
        raise ValueError('OS not recognized')

def volume_path(exp, time, isrec, isImage=True, OS='Windows'):
    flag = exp_flag[exp_list.index(exp)]
    vol = '0050' if flag else '0100'
    folder_name = 'entry' + str(time).zfill(4) + '_no_extpag_db' + vol + '_vol'
    volume_name = 'volume_v2.npy' if isImage else 'segmented.npy'
    return os.path.join(OS_path(exp, OS, isrec), folder_name, volume_name)


if __name__ == '__main__':
    exp_list = ['P28A_ISC_FT_H_Exp5', 'VCT5A_FT_H_Exp3']

    exp_start_time = [114, 4]

    exp_flag = [False, True]

    exp_rec = [range(114,222), range(0,222)]

    for exp in exp_list:

        start_time = exp_start_time[exp_list.index(exp)]
        end_time = 220
        skip180=True
        OS = 'Linux'
        rec = exp_rec[exp_list.index(exp)]
        time_steps = range(start_time, end_time+1, 2) if skip180 else range(start_time, end_time+1)
        shape = (len(time_steps), 260, 700, 700)
        if not os.path.exists(os.path.join('/data/projects/whaitiri/Data/Data_Processing_July2022','hypervolumes', exp)):
            os.mkdir(os.path.join('/data/projects/whaitiri/Data/Data_Processing_July2022','hypervolumes', exp))
        hypervolume = open_memmap(os.path.join('/data/projects/whaitiri/Data/Data_Processing_July2022','hypervolumes', exp, 'hypervolume.npy'), 
                                  dtype=np.half, mode='w+', shape=shape)

        for t, time in tqdm(enumerate(time_steps), desc=f'Loading {exp} memmap', total=len(time_steps)):
            volume = open_memmap(volume_path(exp=exp, time=time, OS=OS, isImage=True, isrec=(time in rec)), mode='r')
            hypervolume[t,:,:,:] = volume[15:275,108:808,144:844]