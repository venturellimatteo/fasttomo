import bpy                                  # type: ignore
import numpy as np                          # type: ignore
import os

# function that modifies the properties of the rendering engine
def modify_engine():
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.time_limit = 240
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_x = 2160
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    return

# function that returns a color tuple given the label and the maximum value of the labels
def modify_properties(o, label):
    mod = o.modifiers.new(name='Smooth', type='SMOOTH')
    mod.factor = 1
    mod.iterations = 5
    if label==1:
        o.data.materials.append(bpy.data.materials['ShellMaterial'])
        o.scale[0] = 1.02
        o.scale[1] = 1.02
    else:
        o.data.materials.append(bpy.data.materials[f'JellyrollMaterial'])
    return None
    
# function that modifies the properties of the object associated to a specific label
def create_materials():
    mat = bpy.data.materials
    mat.new(name='ShellMaterial')
    mat['ShellMaterial'].use_nodes = True
    mat['ShellMaterial'].node_tree.nodes['Principled BSDF'].inputs[0].default_value = (0, 0, 0, 1)
    mat['ShellMaterial'].node_tree.nodes['Principled BSDF'].inputs[21].default_value = 0.75
    mat.new(name='JellyrollMaterial')
    mat['JellyrollMaterial'].use_nodes = True
    mat['JellyrollMaterial'].node_tree.nodes['Principled BSDF'].inputs[0].default_value = (1, 1, 1, 1)
    mat['JellyrollMaterial'].node_tree.nodes['Principled BSDF'].inputs[21].default_value = 0.05
    return

def create_folders(path, exp):
    exp_path = os.path.join(path, exp)
    stl_path = os.path.join(exp_path, 'sidewall_stls')
    render_path = os.path.join(exp_path, 'sidewall_renders')
    persp_path = os.path.join(render_path, 'perspective view')
    if not os.path.exists(persp_path):
        os.makedirs(persp_path)
    return stl_path, persp_path

def add_lights():
    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(2, 0, 2), scale=(1, 1, 1))
    bpy.context.active_object.data.energy = 500
    bpy.context.object.data.shadow_soft_size = 2
    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(-1, -1, 1), scale=(1, 1, 1))
    bpy.context.active_object.data.energy = 750
    bpy.context.object.data.shadow_soft_size = 2
    return

def persp_view_render(persp_path, time):
    bpy.context.scene.render.resolution_y = 2160
    bpy.ops.object.camera_add(enter_editmode=False, location=(2.4, -2.4, 2.8), rotation=(np.pi/4, 0, np.pi/4))
    bpy.context.scene.camera = bpy.context.active_object
    bpy.context.scene.render.filepath = os.path.join(persp_path, time + '.png')
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    path = '/Volumes/T7/Thesis'
    modify_engine()
    create_materials()
    exp = 'VCT5_FT_N_Exp1'
    stl_path, persp_path = create_folders(path, exp)
    for time in os.listdir(stl_path):
        if time in ['.DS_Store']:
            continue
        time_path = os.path.join(stl_path, time)
        for obj in os.listdir(time_path):
            if obj == '.DS_Store':
                continue
            bpy.ops.import_mesh.stl(filepath=os.path.join(time_path, obj))
            o = bpy.context.active_object
            label = int(o.name)
            modify_properties(o, label)
        add_lights()
        persp_view_render(persp_path, time)
        for i in range(len(bpy.data.objects)):
            bpy.data.objects[i].select_set(True)
        bpy.ops.object.delete()