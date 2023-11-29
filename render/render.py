import bpy                                  # type: ignore
import numpy as np                          # type: ignore
import os

def exp_list():
    return ['P28A_FT_H_Exp1', 'P28A_FT_H_Exp2', 'P28A_FT_H_Exp3_3', 'P28A_FT_H_Exp4_2', 'P28B_ISC_FT_H_Exp2', 'VCT5_FT_N_Exp1',
            'VCT5_FT_N_Exp3', 'VCT5_FT_N_Exp4', 'VCT5_FT_N_Exp5', 'VCT5A_FT_H_Exp2', 'VCT5A_FT_H_Exp5']

def color_palette():
    return np.array([[0.90, 0.01, 0.01], [0.96, 0.33, 0.01], [1.00, 0.90, 0.01], [0.02, 0.70, 0.04],
                     [0.08, 0.76, 0.83], [0.03, 0.11, 0.64], [0.00, 0.00, 0.35],
                     [0.27, 0.01, 0.34], [0.94, 0.20, 0.74], [0.25, 0.25, 0.25]])

# function that returns a color tuple given the label and the maximum value of the labels
def modify_properties(so, label, palette):
    mod = so.modifiers.new(name='Smooth', type='SMOOTH')
    mod.factor = 1
    mod.iterations = 5
    if label==1:
        so.data.materials.append(bpy.data.materials['ShellMaterial'])
    else:
        so.data.materials.append(bpy.data.materials[f'Material{label%len(palette)}'])
    return None
    
# function that modifies the properties of the object associated to a specific label
def create_materials(palette):
    mat = bpy.data.materials
    for i in range(len(palette)):
        R, G, B = palette[i]
        mat.new(name=f'Material{i}')
        mat[f'Material{i}'].use_nodes = True
        mat[f'Material{i}'].node_tree.nodes['Principled BSDF'].inputs[0].default_value = (R, G, B, 1)
    mat.new(name='ShellMaterial')
    mat['ShellMaterial'].use_nodes = True
    mat['ShellMaterial'].node_tree.nodes['Principled BSDF'].inputs[0].default_value = (0, 0, 0, 1)
    mat['ShellMaterial'].node_tree.nodes['Principled BSDF'].inputs[21].default_value = 0.1

# function that modifies the properties of the rendering engine
def modify_engine():
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.time_limit = 15
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_x = 2160
    bpy.context.scene.render.image_settings.file_format = 'PNG'

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None



if __name__ == "__main__":
    parent_dir = '/Volumes/T7/Thesis'
    palette = color_palette()
    modify_engine()
    create_materials(palette)
    experiments = exp_list()
    
    for exp in experiments:
        
        # ---------------------- CREATING FOLDERS ---------------------- #

        exp_dir = os.path.join(parent_dir, exp)
        stl_dir = os.path.join(exp_dir, 'stls')
        time_list = os.listdir(stl_dir)
        render_dir = os.path.join(exp_dir, 'renders')
        create_folder(render_dir)
        side_dir = os.path.join(render_dir, 'side view')
        create_folder(side_dir)
        top_dir = os.path.join(render_dir, 'top view')
        create_folder(top_dir)
        persp_dir = os.path.join(render_dir, 'perspective view')
        create_folder(persp_dir)
        
        iterable = time_list
        progress = 1
        
        print(f'Render {exp} started')        

        for t in iterable:
            if t == '.DS_Store':
                continue
            time_dir = os.path.join(stl_dir, t)
            obj_names_list = os.listdir(time_dir)

            # ---------------------- LOADING MESHES ---------------------- #
            
            for obj_name in obj_names_list:
                label = int(obj_name.split('.')[0])
                obj_path = os.path.join(time_dir, obj_name)
                bpy.ops.import_mesh.stl(filepath=obj_path)
                so = bpy.context.active_object
                modify_properties(so, label, palette)


            # ---------------------- TOP VIEW ---------------------- #

            bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(2, 0, 2), scale=(1, 1, 1))
            bpy.context.active_object.data.energy = 500
            bpy.context.object.data.shadow_soft_size = 2
            bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(-1, -1, 1), scale=(1, 1, 1))
            bpy.context.active_object.data.energy = 750
            bpy.context.object.data.shadow_soft_size = 2
            bpy.context.scene.render.resolution_y = 2160
            bpy.ops.object.camera_add(enter_editmode=False, location=(0, 0, 1))
            bpy.context.active_object.data.type = 'ORTHO'
            bpy.context.object.data.ortho_scale = 2
            bpy.context.scene.camera = bpy.context.active_object
            bpy.context.scene.render.filepath = os.path.join(top_dir, t + '.png')
            bpy.ops.render.render(write_still=True)
            bpy.ops.object.delete()

            # ---------------------- SIDE VIEW ---------------------- #
            bpy.context.scene.render.resolution_y = 1440
            bpy.ops.object.camera_add(enter_editmode=False, location=(0, -1, 0), rotation=(np.pi/2, 0, 0))
            bpy.context.active_object.data.type = 'ORTHO'
            bpy.context.object.data.ortho_scale = 2
            bpy.context.scene.camera = bpy.context.active_object
            bpy.context.scene.render.filepath = os.path.join(side_dir, t + '.png')
            bpy.ops.render.render(write_still=True)
            bpy.ops.object.delete()

            # ---------------------- PERSPECTIVE VIEW ---------------------- #

            bpy.context.scene.render.resolution_y = 2160
            bpy.ops.object.camera_add(enter_editmode=False, location=(1.8, -1.8, 2.4), rotation=(np.pi/4, 0, np.pi/4))
            bpy.context.scene.camera = bpy.context.active_object
            bpy.context.scene.render.filepath = os.path.join(persp_dir, t + '.png')
            bpy.ops.render.render(write_still=True)
            bpy.ops.object.delete()

            # ---------------------- CLEANING ---------------------- #
            
            # removing objects for each label related to current time step        
            for i in range(len(bpy.data.objects)):
                bpy.data.objects[i].select_set(True)
            bpy.ops.object.delete()

            # for m in bpy.data.materials:
            #     bpy.data.materials.remove(m)