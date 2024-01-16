import myfunctions as mf
import concurrent.futures as cf
from numpy.lib.format import open_memmap
import os

def pipeline(exp, segment, filtering, motion, graphs, OS, offset):
    if segment:
        hypervolume_mask = mf.segment4D(exp=exp, OS=OS, offset=offset)
    else:
        hypervolume_mask = open_memmap(os.path.join(mf.OS_path(exp, OS), 'hypervolume_mask.npy'), mode='r+')
    if filtering:
        mf.filtering4D(hypervolume_mask=hypervolume_mask, exp=exp, offset=offset)
    if manual_filtering:
        mf.manual_filtering(hypervolume_mask=hypervolume_mask, exp=exp, offset=offset)
    if motion:
        df = mf.motion_df(hypervolume_mask, exp=exp, offset=offset)
        df.to_csv(os.path.join(mf.OS_path(exp, OS), 'motion_properties.csv'), index=False)
    if graphs:
        mf.plot_data(exp, OS, offset, save=True)
    return None


if __name__ == '__main__':

    exp_list = mf.exp_list()
    segment = False
    filtering = False
    manual_filtering = False
    motion = False
    graphs = True
    OS = 'MacOS_SSD'

    # with cf.ProcessPoolExecutor() as executor:
    #     for offset, exp in enumerate(exp_list):
    #         executor.submit(pipeline, exp, segment, filtering, motion, graphs, OS, offset)

    # pipeline(exp=exp_list[9], segment=segment, filtering=filtering, motion=motion, graphs=graphs, OS=OS, offset=0)

    for exp in exp_list:
        pipeline(exp=exp, segment=segment, filtering=filtering, motion=motion, graphs=graphs, OS=OS, offset=0)

    print('\nAll done!')