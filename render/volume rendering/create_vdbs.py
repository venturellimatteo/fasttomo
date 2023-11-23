import pyopenvdb as vdb                     # type: ignore
import numpy as np                          # type: ignore
from numpy.lib.format import open_memmap    # type: ignore
from tqdm import tqdm
import os

# function that saves the vdb file related to a given label given the map and the path
def save_vdb(temp, obj_path):
    grid = vdb.Int32Grid()
    grid.copyFromArray(temp.astype(int))
    grid.transform  = vdb.createLinearTransform(voxelSize=0.01)
    grid.gridClass = vdb.GridClass.FOG_VOLUME
    grid.name = 'density'
    vdb.write(obj_path, grid)
    return None

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None

if __name__ == "__main__":
    parent_dir = '/Volumes/T7/Thesis'
    exp_list = ['P28A_FT_H_Exp1','P28A_FT_H_Exp2','P28A_FT_H_Exp3_3', 'P28A_FT_H_Exp4_2', 'P28B_ISC_FT_H_Exp2','VCT5_FT_N_Exp1',
                'VCT5_FT_N_Exp3','VCT5_FT_N_Exp4','VCT5_FT_N_Exp5','VCT5A_FT_H_Exp2','VCT5A_FT_H_Exp5']
    exp_list = ['P28A_FT_H_Exp1']
    
    for exp in exp_list:
        # defining and creating the folders where the vdb objects will be saved
        exp_dir = os.path.join(parent_dir, exp)
        vdb_dir = os.path.join(exp_dir, 'vdbs')
        create_folder(vdb_dir)
        # opening the 4D segmentation map as a memmap to save space on the RAM
        mask = open_memmap(os.path.join(exp_dir, 'hypervolume_mask.npy'), mode='r')
        # defining the number of time steps contained in the 4D volume
        time_steps = mask.shape[0]

        for t in tqdm(range(time_steps), desc=f'Creating {exp} vdbs', leave=False):
            # defining the vdbs folder path associated to this time step
            vdb_time_dir = os.path.join(vdb_dir, str(t).zfill(3))
            # creating the folder if it doesn't exist already
            create_folder(vdb_time_dir)
            # creating a numpy array containing the labels present in the current volume
            labels = np.unique(mask[t])
            
            for label in labels:
                # we do not create any objects associated with the background
                if label==0:
                    continue
                # creating a map containing all zeros except where the specific label is present (value=1000)
                temp = np.where(mask[t]==label, 1000, 0)
                # defining the name and the path of the vdb objects that are being saved
                obj_path = os.path.join(vdb_time_dir, str(label) + '.vdb')
                # constructing the vdb grid for the label and saving the vdb object in the folder associated to that time instant
                save_vdb(temp, obj_path)
    print('Done!')