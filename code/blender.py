import subprocess
import os

blender_executable_path = '/Applications/Blender.app/Contents/MacOS/Blender'

parent_path = '/Users/matteoventurelli/Documents/VS Code/MasterThesis/code/blender_scripts'
blender_file_path = os.path.join(parent_path, 'surface_rendering.blend')
script_path = os.path.join(parent_path, 'agglomerate_render.py')

command = [blender_executable_path, blender_file_path, '--background', '--python', script_path]

subprocess.run(command)