import open3d as o3d
import os

# List of point cloud files
file_paths = [
    "./video_output_point_cloud/video_1/frame_0000.pcd"
    # "./video_output_point_cloud/myvideo_2/frame_138.pcd",
    # "./video_output_point_cloud/myvideo_2/frame_139.pcd"
    # Add more file paths as needed
]

# Initialize an empty list to hold point clouds
point_clouds = []

for file_path in file_paths:
    if os.path.exists(file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.is_empty():
            print(f"Warning: The point cloud {file_path} is empty.")
        else:
            point_clouds.append(pcd)
            print(f"Loaded point cloud from {file_path} with {len(pcd.points)} points.")
    else:
        print(f"Error: File not found {file_path}")

# Visualize all loaded point clouds if there are any
if point_clouds:
    o3d.visualization.draw_geometries(point_clouds)
else:
    print("No valid point clouds to display.")