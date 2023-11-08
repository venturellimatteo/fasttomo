import myfunctions as mf
from numpy.lib.format import open_memmap
import os

segment = False
filtering = False
motion = True
exp = mf.exp_list()[1]
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


# if motion:
#     position, volume, speed, volume_exp_rate, avg_volume, agg_number = mf.motion_matrix(hypervolume_mask, exp=exp)
#     if not os.path.exists(os.path.join(mf.OS_path(exp, OS), 'motion_properties')):
#         os.makedirs(os.path.join(mf.OS_path(exp, OS), 'motion_properties'))
#     print('Saving properties...')
#     save(os.path.join(mf.OS_path(exp, OS), 'motion_properties', 'position.npy'), position)
#     save(os.path.join(mf.OS_path(exp, OS), 'motion_properties', 'volume.npy'), volume)
#     save(os.path.join(mf.OS_path(exp, OS), 'motion_properties', 'speed.npy'), speed)
#     save(os.path.join(mf.OS_path(exp, OS), 'motion_properties', 'volume_exp_rate.npy'), volume_exp_rate)
#     save(os.path.join(mf.OS_path(exp, OS), 'motion_properties', 'avg_volume.npy'), avg_volume)
#     save(os.path.join(mf.OS_path(exp, OS), 'motion_properties', 'agg_number.npy'), agg_number)
#     print('Done!')