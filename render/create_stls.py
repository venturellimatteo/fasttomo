import numpy as np    
from numpy.lib.format import open_memmap                           
from skimage.measure import marching_cubes
from stl import mesh
from tqdm import tqdm                               
import os

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':

    OS = 'MacOS_SSD'
    exp_list = ['P28A_FT_H_Exp1', 'P28A_FT_H_Exp2', 'P28A_FT_H_Exp3_3', 'P28A_FT_H_Exp4_2', 'P28B_ISC_FT_H_Exp2', 'VCT5_FT_N_Exp1',
                'VCT5_FT_N_Exp3', 'VCT5_FT_N_Exp4', 'VCT5_FT_N_Exp5', 'VCT5A_FT_H_Exp2', 'VCT5A_FT_H_Exp5']

    for exp in exp_list:
        parent_dir = '/Volumes/T7/Thesis/' + exp
        hypervolume_mask = open_memmap(os.path.join(parent_dir, 'hypervolume_mask.npy'), mode='r')
        stl_dir = os.path.join(parent_dir, 'stls')
        create_folder(stl_dir)
        for time in tqdm(range(46, hypervolume_mask.shape[0]), desc=exp, leave=False):
            time_dir = os.path.join(stl_dir, str(time).zfill(3))
            create_folder(time_dir)
            verts, faces, normals, values = marching_cubes(hypervolume_mask[time], 0)
            verts = (0.004 * verts * np.array([1, 1, -1])) + np.array([-1, -1, 0.5])
            values = values.astype(np.ushort)
            for label in np.unique(values):
                label_faces = faces[np.where(values[faces[:,0]] == label)]
                stl_mesh = mesh.Mesh(np.zeros(label_faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, face in enumerate(label_faces):
                    for j in range(3):
                        stl_mesh.vectors[i][j] = verts[face[j]]
                stl_mesh.save(os.path.join(time_dir, str(label).zfill(5) + '.stl'))
            
