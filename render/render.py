import bpy                                  # type: ignore
import numpy as np                          # type: ignore
from tqdm import tqdm
import os

# function that returns a color tuple given the label and the maximum value of the labels
def color(label, color_palette):
    l = len(color_palette)
    R, G, B = color_palette[label%l]
    return (R, G, B, 1)
    
# function that modifies the properties of the object associated to a specific label
def modify_properties(so, label, color_palette):
    bpy.ops.object.modifier_add(type='SMOOTH')
    bpy.context.object.modifiers["Smooth"].factor = 1
    bpy.context.object.modifiers["Smooth"].iterations = 10

    return None

# function that modifies the properties of the rendering engine
def modify_engine():
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.time_limit = 30
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_x = 2160
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.5

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None



if __name__ == "__main__":
    parent_dir = '/Volumes/T7/Thesis'
    exp_list = ['P28A_FT_H_Exp1', 'P28A_FT_H_Exp2', 'P28A_FT_H_Exp3_3', 'P28A_FT_H_Exp4_2', 'P28B_ISC_FT_H_Exp2', 'VCT5_FT_N_Exp1',
                'VCT5_FT_N_Exp3', 'VCT5_FT_N_Exp4', 'VCT5_FT_N_Exp5', 'VCT5A_FT_H_Exp2', 'VCT5A_FT_H_Exp5']
    color_palette = np.load(os.path.join(parent_dir, 'Render/palette.npy'))
    # modify rendering engine
    modify_engine()
    
    for exp in exp_list:
        # defining and creating the folders where the vdb objects will be saved
        exp_dir = os.path.join(parent_dir, exp)
        stl_dir = os.path.join(exp_dir, 'stls')
        time_list = os.listdir(stl_dir)

        render_dir = os.path.join(exp_dir, 'renders')
        create_folder(render_dir)

        side_dir = os.path.join(render_dir, 'side view')
        create_folder(side_dir)

        top_dir = os.path.join(render_dir, 'top view')
        create_folder(top_dir)
        
        iterable = time_list
        progress = 1
        
        print(f'Render {exp} started')        

        for t in iterable:
            time_dir = os.path.join(stl_dir, t)
            obj_names_list = os.listdir(time_dir)
            
            for obj_name in obj_names_list:
                label = int(obj_name.split('.')[0])
                obj_path = os.path.join(time_dir, obj_name)
                bpy.ops.import_mesh.stl(filepath=obj_path)
                # selecting the new object
                so = bpy.context.active_object
                # modifying the properties of the new object
                modify_properties(so, label, color_palette)

            # creating the light and modifying its intensity
            # bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(10, 0, 10), scale=(1, 1, 1))
            # bpy.context.active_object.data.energy = 1000

            # ---------------------- TOP VIEW ---------------------- #

            bpy.context.scene.render.resolution_y = 2160
            bpy.ops.object.camera_add(enter_editmode=False, location=(0, 0, 1))
            bpy.context.active_object.data.type = 'ORTHO'
            bpy.context.object.data.ortho_scale = 2
            bpy.context.scene.camera = bpy.context.active_object
            bpy.context.scene.render.filepath = os.path.join(top_dir, t + '.png')
            bpy.ops.render.render(write_still=True)
            print(f'Render {exp}: {progress}/{2*len(iterable)}')
            progress += 1
            bpy.ops.object.delete()

            # ---------------------- SIDE VIEW ---------------------- #
            bpy.context.scene.render.resolution_y = 1440
            bpy.ops.object.camera_add(enter_editmode=False, location=(0, -1, 0), rotation=(np.pi/2, 0, 0))
            bpy.context.active_object.data.type = 'ORTHO'
            bpy.context.object.data.ortho_scale = 2
            bpy.context.scene.camera = bpy.context.active_object
            bpy.context.scene.render.filepath = os.path.join(side_dir, t + '.png')
            bpy.ops.render.render(write_still=True)
            print(f'Render {exp}: {progress}/{2*len(iterable)}')
            progress += 1
            
            # removing objects for each label related to current time step        
            for i in range(len(bpy.data.objects)):
                bpy.data.objects[i].select_set(True)
            bpy.ops.object.delete()
        
            for m in bpy.data.materials:
                bpy.data.materials.remove(m)