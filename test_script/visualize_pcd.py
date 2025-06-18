import open3d as o3d
import numpy as np
import argparse

def visualize_pcd(pcd_file):
    """
    Visualize a PCD file using Open3D
    """
    try:
        # Read the point cloud
        print(f"Loading point cloud from: {pcd_file}")
        pcd = o3d.io.read_point_cloud(pcd_file)
        
        if not pcd.has_points():
            print("Error: The point cloud is empty or couldn't be loaded.")
            return
        
        # Print some basic information
        print(f"Point cloud loaded successfully!")
        print(f"Number of points: {len(pcd.points)}")
        print(f"Available attributes: {'Colors' if pcd.has_colors() else 'No colors'}, {'Normals' if pcd.has_normals() else 'No normals'}")
        
        # Basic preprocessing (optional)
        print("Performing basic preprocessing...")
        # Remove NaN points if any
        pcd = pcd.remove_non_finite_points()
        # Downsample if the point cloud is too large
        if len(pcd.points) > 1000000:
            print("Downsampling large point cloud...")
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        # Visualize the point cloud
        print("Visualizing point cloud...")
        o3d.visualization.draw_geometries([pcd],
                                          window_name="PCD Viewer",
                                          width=1024,
                                          height=768,
                                          point_show_normal=False)
        
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Visualize PCD (Point Cloud Data) files')
    parser.add_argument('pcd_file', type=str, help='Path to the PCD file to visualize')
    args = parser.parse_args()
    
    visualize_pcd(args.pcd_file)

if __name__ == "__main__":
    main()