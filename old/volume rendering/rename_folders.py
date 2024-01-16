import os
from tqdm import tqdm

if __name__ == "__main__":
    parent_dir = '/Volumes/T7/Thesis'
    exp_list = ['P28A_FT_H_Exp3_3','P28B_ISC_FT_H_Exp2','VCT5_FT_N_Exp1',
                'VCT5_FT_N_Exp3','VCT5_FT_N_Exp4','VCT5_FT_N_Exp5','VCT5A_FT_H_Exp2','VCT5A_FT_H_Exp5']
    exp_list = ['P28A_FT_H_Exp4_2']
    for exp in tqdm(exp_list, desc='Renaming folders'):
        vdb_dir = os.path.join(parent_dir, exp, 'vdbs')
        for time in os.listdir(vdb_dir):
            if time == '.DS_Store':
                continue
            old_time_dir = os.path.join(vdb_dir, time)
            time_dir = os.path.join(vdb_dir, (time.split('-')[1]).zfill(3))
            os.rename(old_time_dir, time_dir)
            for obj in os.listdir(time_dir):
                if obj == '.DS_Store':
                    continue
                obj_path = os.path.join(time_dir, obj)
                label = obj.split('.')[0]
                new_obj = label.zfill(5) + '.vdb'
                os.rename(obj_path, os.path.join(time_dir, new_obj))
        render_dir = os.path.join(parent_dir, exp, 'renders')
        for view in ['side view', 'top view']:
            view_dir = os.path.join(render_dir, view)
            for time in os.listdir(view_dir):
                if time == '.DS_Store':
                    continue
                old_image = os.path.join(view_dir, time)
                t = time.split('-')[1]
                t = t.split('.')[0]
                new_image = os.path.join(view_dir, t.zfill(3) + '.png')
                os.rename(old_image, new_image)
                

    