import myfunctions as mf
from numpy.lib.format import open_memmap
import os

# hypervolume_mask = mf.segment4D(exp=mf.exp_list()[0], OS='Linux')

OS = 'Linux'
exp=mf.exp_list()[0]

hypervolume_mask = open_memmap(os.path.join(mf.OS_path(exp, OS), 'hypervolume_mask.npy'))

mf.filtering4D(hypervolume_mask)