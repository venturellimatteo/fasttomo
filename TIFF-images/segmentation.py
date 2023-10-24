import numpy as np                  # type: ignore    
import myfunctions as mf            # type: ignore

if __name__ == '__main__':
    for exp in mf.exp_list():
        sequence = mf.read_4Dsequence(exp, OS='Tyrex')
        threshold = mf.find_threshold(sequence)
        segmented_sequence = mf.segment4D(sequence, threshold, save=True, OS='Tyrex')
    print('Done!')