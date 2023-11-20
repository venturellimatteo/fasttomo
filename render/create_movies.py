import cv2
import os


# Set the frames per second (fps)
fps = 8
exp_list = ['P28A_FT_H_Exp1','P28A_FT_H_Exp2','P28A_FT_H_Exp3_3', 'P28A_FT_H_Exp4_2', 'P28B_ISC_FT_H_Exp2','VCT5_FT_N_Exp1',
            'VCT5_FT_N_Exp3','VCT5_FT_N_Exp4','VCT5_FT_N_Exp5','VCT5A_FT_H_Exp2','VCT5A_FT_H_Exp5']
exp_list = ['P28A_FT_H_Exp1']

# Path to the directory containing your frames
for exp in exp_list:
    parent_dir = f'/Volumes/T7/Thesis/{exp}/renders'
    for view in ['top view', 'side view']:
        frames_dir = os.path.join(parent_dir, view)
        # Get the list of frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        # Set the video resolution based on the first frame
        frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        height, width, layers = frame.shape
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try other codecs like 'XVID'
        video = cv2.VideoWriter(os.path.join(parent_dir, view + '.mp4'), fourcc, fps, (width, height))
        # Iterate through the frames and write to the video file
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            video.write(frame)
        # Release the VideoWriter object
        video.release()
        print(f'{exp} {view} done!')
print('Done!')