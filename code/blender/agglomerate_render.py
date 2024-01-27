import bpy # type: ignore
import numpy as np
import sys
import os

def color_palette():
    return np.array([[0.90, 0.01, 0.01], [0.96, 0.33, 0.01], [1.00, 0.90, 0.01], [0.02, 0.70, 0.04],
                     [0.08, 0.76, 0.83], [0.03, 0.11, 0.64], [0.00, 0.00, 0.35],
                     [0.27, 0.01, 0.34], [0.94, 0.20, 0.74], [0.25, 0.25, 0.25]])

# function that modifies the properties of the rendering engine
def modify_engine():
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.time_limit = 15
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_x = 2160
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.cycles.use_denoising = False
    return

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
    return

def create_folders(path, exp):
    exp_path = os.path.join(path, exp)
    stl_path = os.path.join(exp_path, 'stls')
    render_path = os.path.join(exp_path, 'renders')
    side_path = os.path.join(render_path, 'side view')
    if not os.path.exists(side_path):
        os.makedirs(side_path)
    top_path = os.path.join(render_path, 'top view')
    if not os.path.exists(top_path):
        os.makedirs(top_path)
    persp_path = os.path.join(render_path, 'perspective view')
    if not os.path.exists(persp_path):
        os.makedirs(persp_path)
    return stl_path, side_path, top_path, persp_path

def import_object(obj_name, time_path):
    label = int(obj_name.split('.')[0])
    obj_path = os.path.join(time_path, obj_name)
    bpy.ops.import_mesh.stl(filepath=obj_path)
    so = bpy.context.active_object
    modify_properties(so, label, palette)

def add_lights():
    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(2, 0, 2), scale=(1, 1, 1))
    bpy.context.active_object.data.energy = 500
    bpy.context.object.data.shadow_soft_size = 2
    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(-1, -1, 1), scale=(1, 1, 1))
    bpy.context.active_object.data.energy = 750
    bpy.context.object.data.shadow_soft_size = 2
    return

def top_view_render(top_path, t):
    bpy.context.scene.render.resolution_y = 2160
    bpy.ops.object.camera_add(enter_editmode=False, location=(0, 0, 1))
    bpy.context.active_object.data.type = 'ORTHO'
    bpy.context.object.data.ortho_scale = 2
    bpy.context.scene.camera = bpy.context.active_object
    bpy.context.scene.render.filepath = os.path.join(top_path, t + '.png')
    bpy.ops.render.render(write_still=True)
    bpy.ops.object.delete()
    return

def side_view_render(side_path, t):
    bpy.context.scene.render.resolution_y = 1440
    bpy.ops.object.camera_add(enter_editmode=False, location=(0, -1, 0), rotation=(np.pi/2, 0, 0))
    bpy.context.active_object.data.type = 'ORTHO'
    bpy.context.object.data.ortho_scale = 2
    bpy.context.scene.camera = bpy.context.active_object
    bpy.context.scene.render.filepath = os.path.join(side_path, t + '.png')
    bpy.ops.render.render(write_still=True)
    bpy.ops.object.delete()
    return

def persp_view_render(persp_path, t):
    bpy.context.scene.render.resolution_y = 2160
    bpy.ops.object.camera_add(enter_editmode=False, location=(1.8, -1.8, 2.4), rotation=(np.pi/4, 0, np.pi/4))
    bpy.context.scene.camera = bpy.context.active_object
    bpy.context.scene.render.filepath = os.path.join(persp_path, t + '.png')
    bpy.ops.render.render(write_still=True)
    bpy.ops.object.delete()



if __name__ == "__main__":
    path = '/Volumes/T7/Thesis'
    palette = color_palette()
    modify_engine()
    create_materials(palette)
    exp = sys.argv[1]
    stl_path, side_path, top_path, persp_path = create_folders(path, exp)
    iterable = os.listdir(stl_path)
    for t in iterable:
        if t == '.DS_Store':
            continue
        time_path = os.path.join(stl_path, t)
        for obj_name in os.listdir(time_path):
            import_object(obj_name, time_path)
        add_lights()
        top_view_render(top_path, t)
        side_view_render(side_path, t)
        persp_view_render(persp_path, t)
        for i in range(len(bpy.data.objects)):
            bpy.data.objects[i].select_set(True)
        bpy.ops.object.delete()