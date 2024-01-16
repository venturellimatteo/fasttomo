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
    exp_list = ['VCT5_FT_N_Exp5', 'VCT5A_FT_H_Exp5']

    for exp in exp_list:
        parent_dir = '/Volumes/T7/Thesis/' + exp
        hypervolume_mask = open_memmap(os.path.join(parent_dir, 'hypervolume_mask.npy'), mode='r')
        stl_dir = os.path.join(parent_dir, 'stls')
        create_folder(stl_dir)
        for time in tqdm(range(hypervolume_mask.shape[0]), desc=exp, leave=False):
            s = hypervolume_mask[time].shape
            mask = np.zeros((s[2], s[1], s[0]+2), dtype=np.ushort)
            mask[:, :, 1:-1] = np.swapaxes(hypervolume_mask[time], 0, 2)
            time_dir = os.path.join(stl_dir, str(time).zfill(3))
            create_folder(time_dir)
            verts, faces, normals, values = marching_cubes(mask, 0)
            verts = (0.004 * verts * np.array([1, 1, -1])) + np.array([-1, -1, 0.5])
            values = values.astype(np.ushort)
            for label in np.unique(values):
                label_faces = faces[np.where(values[faces[:,0]] == label)]
                stl_mesh = mesh.Mesh(np.zeros(label_faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, face in enumerate(label_faces):
                    for j in range(3):
                        stl_mesh.vectors[i][j] = verts[face[j]]
                stl_mesh.save(os.path.join(time_dir, str(label).zfill(5) + '.stl'))
            
