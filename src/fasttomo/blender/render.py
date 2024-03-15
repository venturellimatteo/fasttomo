"""

This module contains the functions to render the agglomerates and
jellyroll in ``Blender``. The module is called from the ``Data.render``
method in the ``fasttomo`` package to run within ``Blender``. 
The script imports the STL files for each time instant for a specific
experiment and renders either the agglomerates or the jellyroll 
(to show the sidewall rupture) in three different views: side, top
and perspective. 

"""

import bpy  # type: ignore
import numpy as np
import sys
import os


def color_palette():
    """

    Returns the color palette used for the different labels.

    Returns
    -------
    palette : np.array
        10x3 array of RGB values for highly distinguishable colors.

    """
    return np.array(
        [
            [0.90, 0.01, 0.01],
            [0.96, 0.33, 0.01],
            [1.00, 0.90, 0.01],
            [0.02, 0.70, 0.04],
            [0.08, 0.76, 0.83],
            [0.03, 0.11, 0.64],
            [0.00, 0.00, 0.35],
            [0.27, 0.01, 0.34],
            [0.94, 0.20, 0.74],
            [0.25, 0.25, 0.25],
        ]
    )


# function that modifies the properties of the rendering engine
def modify_engine(isJellyroll):
    """

    Modifies the properties of the rendering engine.

    Parameters
    ----------
    isJellyroll : bool
        Whether agglomerates or jellyroll are being rendered.
        Default is ``False``.

    """
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.time_limit = 240 if isJellyroll else 15
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.render.resolution_x = 2160
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.cycles.use_denoising = False
    return


# function that returns a color tuple given the label and the maximum value of the labels
def modify_properties(obj, label, isJellyroll):
    """

    Modifies the properties of the object associated to a
    specific label. A smoothing modifier is added to the
    object and the material is assigned based on the label.

    Parameters
    ----------
    isJellyroll : bool
        Whether agglomerates or jellyroll are being rendered.
        Default is ``False``.

    """
    mod = obj.modifiers.new(name="Smooth", type="SMOOTH")
    mod.factor = 1
    mod.iterations = 5
    if label == 1:
        obj.data.materials.append(bpy.data.materials["ShellMaterial"])
        if isJellyroll:
            obj.scale[0] = obj.scale[1] = 1.02
        return
    if isJellyroll:
        obj.data.materials.append(bpy.data.materials[f"JellyrollMaterial"])
        return
    obj.data.materials.append(
        bpy.data.materials[f"Material{label%len(color_palette())}"]
    )
    return


# function that modifies the properties of the object associated to a specific label
def create_materials(isJellyroll):
    """

    Creates the materials for the different labels. A black
    material is assigned to the battery casing, a grey material
    to the jellyroll and different colors for the agglomerates.

    Parameters
    ----------
    isJellyroll : bool
        Whether agglomerates or jellyroll are being rendered.
        Default is ``False``.

    """
    mat = bpy.data.materials
    mat.new(name="ShellMaterial")
    mat["ShellMaterial"].use_nodes = True
    mat["ShellMaterial"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
        0,
        0,
        0,
        1,
    )
    mat["ShellMaterial"].node_tree.nodes["Principled BSDF"].inputs[21].default_value = (
        0.75 if isJellyroll else 0.1
    )
    if isJellyroll:
        mat.new(name="JellyrollMaterial")
        mat["JellyrollMaterial"].use_nodes = True
        mat["JellyrollMaterial"].node_tree.nodes["Principled BSDF"].inputs[
            0
        ].default_value = (1, 1, 1, 1)
        mat["JellyrollMaterial"].node_tree.nodes["Principled BSDF"].inputs[
            21
        ].default_value = 0.05
        return
    palette = color_palette()
    for i in range(len(palette)):
        R, G, B = palette[i]
        mat.new(name=f"Material{i}")
        mat[f"Material{i}"].use_nodes = True
        mat[f"Material{i}"].node_tree.nodes["Principled BSDF"].inputs[
            0
        ].default_value = (R, G, B, 1)
    return


def create_folders(path, isJellyroll):
    """

    Returns the path for the STL files. Creates and returns the
    folder to store the renders for the side, top and perspective
    views.

    Parameters
    ----------
    path : str
        Path to the experiment folder.
    isJellyroll : bool
        Whether agglomerates or jellyroll are being rendered.
        Default is ``False``.

    Returns
    -------
    stl_path : str
        Path to the STL files.
    side_path : str
        Path to the side view renders.
    top_path : str
        Path to the top view renders.
    persp_path : str
        Path to the perspective view renders.

    """
    path_modifier = "" if not isJellyroll else "sidewall_"
    stl_path = os.path.join(path, path_modifier + "stls")
    render_path = os.path.join(path, path_modifier + "renders")
    side_path = os.path.join(render_path, "side view")
    if not os.path.exists(side_path):
        os.makedirs(side_path)
    top_path = os.path.join(render_path, "top view")
    if not os.path.exists(top_path):
        os.makedirs(top_path)
    persp_path = os.path.join(render_path, "perspective view")
    if not os.path.exists(persp_path):
        os.makedirs(persp_path)
    return stl_path, side_path, top_path, persp_path


def import_object(obj_name, time_path, isJellyroll):
    """

    Imports the STL file for a specific label, creates the associated
    ``Blender`` object and modifies its properties.

    Parameters
    ----------
    obj_name : str
        Name of the STL file.
    time_path : str
        Path to the folder containing the STL files for a specific
        time instant.
    isJellyroll : bool
        Whether agglomerates or jellyroll are being rendered.
        Default is ``False``.

    """
    bpy.ops.import_mesh.stl(filepath=os.path.join(time_path, obj_name))
    obj = bpy.context.active_object
    label = int(obj.name)
    modify_properties(obj, label, isJellyroll)
    return


def add_lights():
    """

    Adds two point lights to the scene.

    """
    bpy.ops.object.light_add(
        type="POINT", radius=1, align="WORLD", location=(2, 0, 2), scale=(1, 1, 1)
    )
    bpy.context.active_object.data.energy = 500
    bpy.context.object.data.shadow_soft_size = 2
    bpy.ops.object.light_add(
        type="POINT", radius=1, align="WORLD", location=(-1, -1, 1), scale=(1, 1, 1)
    )
    bpy.context.active_object.data.energy = 750
    bpy.context.object.data.shadow_soft_size = 2
    return


def top_view_render(top_path, time):
    """

    Renders the top view of the scene in orthographic mode.

    Parameters
    ----------
    top_path : str
        Path to the folder containing the top view renders.
    time : str
        Time instant for the render.

    """
    bpy.context.scene.render.resolution_y = 2160
    bpy.ops.object.camera_add(
        enter_editmode=False, location=(0, 0, 1)
    )  # 0,2 0,2 1 for P42A_ISC_FT_H_Exp2, 0, 0, 1 for others
    bpy.context.active_object.data.type = "ORTHO"
    bpy.context.object.data.ortho_scale = 2  # 2.5 for P42A_ISC_FT_H_Exp2, 2 for others
    bpy.context.scene.camera = bpy.context.active_object
    bpy.context.scene.render.filepath = os.path.join(top_path, time + ".png")
    bpy.ops.render.render(write_still=True)
    bpy.ops.object.delete()
    return


def side_view_render(side_path, time):
    """

    Renders the side view of the scene in orthographic mode.

    Parameters
    ----------
    side_path : str
        Path to the folder containing the side view renders.
    time : str
        Time instant for the render.

    """
    bpy.context.scene.render.resolution_y = 1440
    bpy.ops.object.camera_add(
        enter_editmode=False,
        location=(0, -1, 0),
        rotation=(
            np.pi / 2,
            0,
            0,
        ),  # (0.2, -1.5, -0.02) if P42A_ISC_FT_H_Exp2, else (0, -1, 0)
    )
    bpy.context.active_object.data.type = "ORTHO"
    bpy.context.object.data.ortho_scale = 2  # 2.5 if P42A_ISC_FT_H_Exp2, else 2
    bpy.context.scene.camera = bpy.context.active_object
    bpy.context.scene.render.filepath = os.path.join(side_path, time + ".png")
    bpy.ops.render.render(write_still=True)
    bpy.ops.object.delete()
    return


def persp_view_render(persp_path, time, isJellyroll):
    """

    Renders a perspective view of the scene.

    Parameters
    ----------
    persp_path : str
        Path to the folder containing the perspective view renders.
    time : str
        Time instant for the render.
    isJellyroll : bool
        Whether agglomerates or jellyroll are being rendered.
        Default is ``False``.

    """
    bpy.context.scene.render.resolution_y = 2160
    location = (
        (2.4, -2.4, 3) if isJellyroll else (1.8, -1.8, 2.4)
    )  # location = (2.4, -2.4, 2.8) if isJellyroll else (1.8, -1.8, 2.4)
    bpy.ops.object.camera_add(
        enter_editmode=False, location=location, rotation=(np.pi / 4, 0, np.pi / 4)
    )
    bpy.context.scene.camera = bpy.context.active_object
    bpy.context.scene.render.filepath = os.path.join(persp_path, time + ".png")
    bpy.ops.render.render(write_still=True)
    return


if __name__ == "__main__":

    path, rupture = sys.argv[-2:]
    isJellyroll = True if rupture == "True" else False
    modify_engine(isJellyroll)
    create_materials(isJellyroll)
    stl_path, side_path, top_path, persp_path = create_folders(path, isJellyroll)
    for time in os.listdir(stl_path):
        if time == ".DS_Store":
            continue
        time_path = os.path.join(stl_path, time)
        for obj_name in os.listdir(time_path):
            if obj_name == ".DS_Store":
                continue
            import_object(obj_name, time_path, isJellyroll)
        add_lights()
        if not isJellyroll:
            top_view_render(top_path, time)
            side_view_render(side_path, time)
        persp_view_render(persp_path, time, isJellyroll)
        for i in range(len(bpy.data.objects)):
            bpy.data.objects[i].select_set(True)
        bpy.ops.object.delete()
