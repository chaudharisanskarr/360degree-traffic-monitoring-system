import cv2
import os
import glob
import re

def enhance_and_clean_image(frame):
    # Apply Gaussian blur
    frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Apply bilateral filter to smooth the image but keep edges sharp
    frame_filtered = cv2.bilateralFilter(frame_blurred, 9, 75, 75)
    
    # Upscale image for better resolution - modify the dst_size as per your need
    height, width = frame.shape[:2]
    dst_size = (width * 2, height * 2)  # scaling factor of 2
    frame_upscaled = cv2.resize(frame_filtered, dst_size, interpolation=cv2.INTER_CUBIC)
    
    return frame_upscaled

def create_video_from_frames(frame_list, output_path, fps, width, height, enhance=True):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height * 2) if enhance else (width, height))

    if not video.isOpened():
        raise ValueError("Video writer could not be opened. Check codec and file path.")

    for frame_filename in frame_list:
        frame = cv2.imread(frame_filename)
        if frame is None:
            print(f"Skipping file, could not read: {frame_filename}")
            continue
        if enhance:
            frame = enhance_and_clean_image(frame)
        video.write(frame)

    video.release()

# Assuming you have already sorted and grouped your frames appropriately
frame_folder = './output'
output_folder = './video_output_depth_estimated'
fps = 60

frames = glob.glob(os.path.join(frame_folder, 'myvid_*-frame-*-dpt_beit_large_512.png'))
if not frames:
    raise ValueError("No frames found. Check the frame pattern and folder path.")

grouped_frames = {}
for frame in frames:
    match = re.search(r'myvid_(\d+)-', frame)
    if match:
        video_id = match.group(1)
        if video_id not in grouped_frames:
            grouped_frames[video_id] = []
        grouped_frames[video_id].append(frame)

for video_id, frame_list in grouped_frames.items():
    frame_list.sort(key=lambda x: int(re.search(r'frame-(\d+)-', x).group(1)))
    first_frame = cv2.imread(frame_list[0])
    height, width, layers = first_frame.shape
    output_video_path = os.path.join(output_folder, f'video_{video_id}.mp4')
    create_video_from_frames(frame_list, output_video_path, fps, width, height)

print("Video reconstruction and enhancement complete for all videos!")