import myfunctions as mf

hypervolume_mask = mf.segment4D(exp=mf.exp_list()[2], OS='Linux')

mf.filtering4D(hypervolume_mask)