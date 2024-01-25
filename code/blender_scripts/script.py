import subprocess

# Path to Blender executable - change this to the correct path for your system
blender_executable_path = "/Applications/Blender.app/Contents/MacOS/Blender"

# Path to your Blender file
blender_file_path = "/Users/matteoventurelli/Documents/VS Code/MasterThesis/code/blender_scripts/surface_rendering.blend"

# Path to your Blender Python script
script_path = "/Users/matteoventurelli/Documents/VS Code/MasterThesis/code/blender_scripts/agglomerate_render.py"

# Command to run Blender in background mode with your script and Blender file
command = [blender_executable_path, blender_file_path, "--background", "--python", script_path]

# Run the command
subprocess.run(command)