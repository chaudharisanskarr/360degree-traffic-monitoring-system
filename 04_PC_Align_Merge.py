import os
import open3d as o3d
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def load_and_preprocess_point_cloud(file_path, voxel_size):
    print(f"Loading and preprocessing point cloud: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_down.points)

def align_point_clouds(points_list):
    print("Aligning point clouds...")
    point_clouds = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)) for points in points_list]
    reference = point_clouds[0]
    for pcd in point_clouds[1:]:
        init_transformation = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd, reference, max_correspondence_distance=0.05,
            init=init_transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        pcd.transform(reg_p2p.transformation)
        reference += pcd
    reference = reference.voxel_down_sample(voxel_size=0.05)  # Downsample merged point cloud
    return reference

def poisson_mesh_reconstruction(pcd):
    print("Performing Poisson mesh reconstruction...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh

def load_point_clouds(folder_path, voxel_size):
    print(f"Loading point clouds from folder: {folder_path}")
    pcd_files = [f for f in os.listdir(folder_path) if f.endswith('.pcd')]
    pcd_paths = [os.path.join(folder_path, f) for f in pcd_files]

    with ProcessPoolExecutor() as executor:
        points_list = list(executor.map(load_and_preprocess_point_cloud, pcd_paths, [voxel_size] * len(pcd_paths)))

    if len(points_list) > 1:
        combined_pcd = align_point_clouds(points_list)
    else:
        combined_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_list[0]))
    return combined_pcd

def process_video_point_clouds(base_folder, voxel_size):
    subfolders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    for folder in subfolders:
        print(f"Processing folder: {folder}")
        combined_pcd = load_point_clouds(folder, voxel_size)
        mesh = poisson_mesh_reconstruction(combined_pcd)
        mesh.paint_uniform_color([0.5, 0.5, 0.7])  # Set mesh color for better visualization

        print(f"Visualizing combined point cloud and mesh for folder: {folder}")
        o3d.visualization.draw_geometries([combined_pcd, mesh], window_name='Point Cloud and Mesh Visualization', width=800, height=600)

        # Optional: Save the mesh
        output_path = os.path.join(folder, "output_mesh.obj")
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Mesh saved to {output_path}")

def main():
    base_folder = './video_output_point_cloud'
    voxel_size = 0.05  # example voxel size
    process_video_point_clouds(base_folder, voxel_size)

if __name__ == '__main__':
    main()