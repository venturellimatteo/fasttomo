import myfunctions as mf

hypervolume_mask = mf.segment4D(exp=mf.exp_list()[0], OS='Linux')
mf.filtering4D(hypervolume_mask)