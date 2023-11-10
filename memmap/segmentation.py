import myfunctions as mf
from numpy.lib.format import open_memmap
import os

exp_list = ['P28A_FT_H_Exp1', 'P28A_FT_H_Exp2', 'P28A_FT_H_Exp3_3',
            'P28A_FT_N_Exp1', 'P28A_FT_N_Exp4',
            'VCT5_FT_N_Exp3', 'VCT5_FT_N_Exp4', 'VCT5_FT_N_Exp5', 
            'VCT5A_FT_H_Exp2', 'VCT5A_FT_H_Exp5']
segment = True
filtering = True
motion = False
OS = 'MacOS'

for exp in exp_list[2:]:
    if segment:
        hypervolume_mask = mf.segment4D(exp=exp, OS=OS)
    else:
        hypervolume_mask = open_memmap(os.path.join(mf.OS_path(exp, OS), 'hypervolume_mask.npy'), mode='r')
    if filtering:
        mf.filtering4D(hypervolume_mask)
    if motion:
        df = mf.motion_df(hypervolume_mask, exp=exp)
        print('Saving properties...')
        df.to_csv(os.path.join(mf.OS_path(exp, OS), 'motion_properties.csv'), index=False)
        print('Done!')