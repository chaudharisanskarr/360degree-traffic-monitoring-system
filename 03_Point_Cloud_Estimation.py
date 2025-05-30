import cv2
import open3d as o3d
import numpy as np
import os

def process_video_rgb_depth_to_pointcloud(video_path_rgb, video_path_depth, output_directory, intrinsic_matrix, depth_scale=1000.0):
    """
    Convert RGB and Depth videos to a series of point clouds with color.

    Parameters:
    - video_path_rgb: path to the input RGB video file.
    - video_path_depth: path to the input depth video file.
    - output_directory: directory to save the output point clouds.
    - intrinsic_matrix: camera intrinsic parameters as an instance of o3d.camera.PinholeCameraIntrinsic.
    - depth_scale: scale factor to convert depth values to meters.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Open the video files
    cap_rgb = cv2.VideoCapture(video_path_rgb)
    cap_depth = cv2.VideoCapture(video_path_depth)
    frame_idx = 0

    while True:
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_depth, frame_depth = cap_depth.read()
        if not ret_rgb or not ret_depth:
            break

        # Prepare the RGB image
        rgb_image = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        rgb_image_o3d = o3d.geometry.Image(rgb_image)

        # Prepare the depth image
        depth = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2GRAY)
        depth = depth.astype(np.float32) / depth_scale  # Convert to meters
        depth_image_o3d = o3d.geometry.Image(depth)

        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image_o3d, depth_image_o3d, depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
        
        # Create a point cloud from the RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_matrix)
        
        # Save the point cloud
        output_file = os.path.join(output_directory, f"frame_{frame_idx:04d}.pcd")
        o3d.io.write_point_cloud(output_file, pcd)
        frame_idx += 1

    cap_rgb.release()
    cap_depth.release()

def process_multiple_videos(video_paths_rgb, video_paths_depth, output_root_directory, intrinsic_matrix):
    """
    Process multiple RGB and Depth videos into point clouds.

    Parameters:
    - video_paths_rgb: list of paths to the RGB video files.
    - video_paths_depth: list of paths to the depth video files.
    - output_root_directory: root directory to save the output point clouds for each video.
    - intrinsic_matrix: camera intrinsic parameters.
    """
    for video_path_rgb, video_path_depth in zip(video_paths_rgb, video_paths_depth):
        output_directory = os.path.join(output_root_directory, os.path.splitext(os.path.basename(video_path_rgb))[0])
        process_video_rgb_depth_to_pointcloud(video_path_rgb, video_path_depth, output_directory, intrinsic_matrix)

# Camera intrinsic parameters example (Adjust these based on your camera setup)
width = 848
height = 478
fx = 425  # Focal length x, example value
fy = 425  # Focal length y, example value
cx = width / 2  # Principal point x
cy = height / 2  # Principal point y
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Example call for multiple videos
video_paths_rgb = ["./input/video_1.mp4", "./input/video_2.mp4","./input/video_3.mp4"]
video_paths_depth = ["./video_output_depth_estimated/video_1.mp4","./video_output_depth_estimated/video_2.mp4","./video_output_depth_estimated/video_3.mp4"]
process_multiple_videos(video_paths_rgb, video_paths_depth, "./video_output_point_cloud", intrinsic)
