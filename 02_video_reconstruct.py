import cv2
import os
import glob
import re

def create_video_from_frames(frame_list, output_path, fps, width, height):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for compatibility, change to 'avc1' for H.264
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video.isOpened():
        raise ValueError("Video writer could not be opened. Check codec and file path.")

    # Read each file, add it to the video
    for frame_filename in frame_list:
        frame = cv2.imread(frame_filename)
        if frame is None:
            print(f"Skipping file, could not read: {frame_filename}")
            continue
        video.write(frame)  # Add frame to video

    # Release everything when done
    video.release()

# Parameters
frame_folder = './output'  # Update this to the actual folder path
output_folder = './video_output_depth_estimated'
fps = 60  # Frames per second of the resulting videos
frame_pattern = os.path.join(frame_folder, 'myvid_*-frame-*-dpt_beit_large_512.png')

# Collect all frames and sort them
frames = glob.glob(frame_pattern)
if not frames:
    raise ValueError("No frames found. Check the frame pattern and folder path.")

# Group frames by video ID
grouped_frames = {}
for frame in frames:
    match = re.search(r'myvid_(\d+)-', frame)
    if match:
        video_id = match.group(1)
        if video_id not in grouped_frames:
            grouped_frames[video_id] = []
        grouped_frames[video_id].append(frame)

# Create separate videos for each video ID
for video_id, frame_list in grouped_frames.items():
    frame_list.sort(key=lambda x: int(re.search(r'frame-(\d+)-', x).group(1)))  # Sort by frame number
    first_frame = cv2.imread(frame_list[0])
    height, width, layers = first_frame.shape
    output_video_path = os.path.join(output_folder, f'video_{video_id}.mp4')
    create_video_from_frames(frame_list, output_video_path, fps, width, height)

print("Video reconstruction is complete for all videos!")