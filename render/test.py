import bpy                                  # type: ignore
import numpy as np                          # type: ignore
from tqdm import tqdm
import os

# function that returns a color tuple given the label and the maximum value of the labels
def modify_properties(so, label, color_palette):
    mod = so.modifiers.new(type='SMOOTH')
    mod.factor = 2
    mod.iterations = 10
    if label==1:
        so.data.materials.append(bpy.data.materials['ShellMaterial'])
    else:
        so.data.materials.append(bpy.data.materials[f'Material{label%len(color_palette)}'])
    return None
    
# function that modifies the properties of the object associated to a specific label
def create_materials(color_palette):
    mat = bpy.data.materials
    for i in range(len(color_palette)):
        R, G, B = color_palette[i]
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
    exp = 'P28A_FT_H_Exp1'
    t = 20
    color_palette = np.load(os.path.join(parent_dir, 'Render/palette.npy'))
    # modify rendering engine
    modify_engine()
    create_materials(color_palette)

    exp_dir = os.path.join(parent_dir, exp)
    stl_dir = os.path.join(exp_dir, 'stls')
    time_list = os.listdir(stl_dir)

    render_dir = os.path.join(exp_dir, 'renders')
    create_folder(render_dir)

    side_dir = os.path.join(render_dir, 'side view')
    create_folder(side_dir)

    top_dir = os.path.join(render_dir, 'top view')
    create_folder(top_dir)

    print(f'Render {exp} started')        

    time_dir = os.path.join(stl_dir, t)
    obj_names_list = os.listdir(time_dir)
    
    for obj_name in obj_names_list:
        label = int(obj_name.split('.')[0])
        obj_path = os.path.join(time_dir, obj_name)
        bpy.ops.import_mesh.stl(filepath=obj_path) # selecting the new object
        so = bpy.context.active_object
        # modifying the properties of the new object
        modify_properties(so, label, color_palette)

    # creating the light and modifying its intensity
    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(2, 0, 2), scale=(1, 1, 1))
    bpy.context.active_object.data.energy = 1000

    # ---------------------- TOP VIEW ---------------------- #

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
    
    # removing objects for each label related to current time step        
    for i in range(len(bpy.data.objects)):
        bpy.data.objects[i].select_set(True)
    bpy.ops.object.delete()