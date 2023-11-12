import myfunctions as mf
import concurrent.futures as cf
from numpy.lib.format import open_memmap
import os

def pipeline(exp, segment, filtering, motion, OS, offset):
    if segment:
        hypervolume_mask = mf.segment4D(exp=exp, OS=OS, offset=offset)
    else:
        hypervolume_mask = open_memmap(os.path.join(mf.OS_path(exp, OS), 'hypervolume_mask.npy'), mode='r')
    if filtering:
        mf.filtering4D(hypervolume_mask=hypervolume_mask, exp=exp, offset=offset)
    if motion:
        df = mf.motion_df(hypervolume_mask, exp=exp, offset=offset)
        df.to_csv(os.path.join(mf.OS_path(exp, OS), 'motion_properties.csv'), index=False)
    return None


if __name__ == '__main__':

    exp_list = ['P28A_FT_H_Exp1', 'P28A_FT_H_Exp2', 'P28A_FT_H_Exp3_3', 'VCT5_FT_N_Exp3', 'VCT5_FT_N_Exp4', 'VCT5_FT_N_Exp5',
                'VCT5A_FT_H_Exp2', 'VCT5A_FT_H_Exp5']
    segment = True
    filtering = True
    motion = True
    OS = 'MacOS'

    processes = []

    with cf.ProcessPoolExecutor() as executor:
        for offset, exp in enumerate(exp_list[5:7]):
            executor.submit(pipeline, exp, segment, filtering, motion, OS, offset)

    print('All done!')