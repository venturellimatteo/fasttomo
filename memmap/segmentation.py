import myfunctions as mf
from numpy.lib.format import open_memmap
import os

segment = True
filtering = True
motion = False
exp = mf.exp_list()[2]
OS = 'MacOS'

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