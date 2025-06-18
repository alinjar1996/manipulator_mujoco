import numpy as np
import mujoco
import open3d as o3d
import os
import time
from typing import Tuple, Optional

class MuJoCoPointCloudGenerator:
    """
    Point cloud generator for MuJoCo simulations with camera rendering
    """
    
    def __init__(self, 
                 model: mujoco.MjModel, 
                 cam_name: str = "camera1",
                 height: int = 480, 
                 width: int = 640,
                 output_dir: str = "pcd_data"):
        """
        Initialize the point cloud generator
        
        Args:
            model: MuJoCo model
            cam_name: Name of the camera in the MuJoCo model
            height: Image height in pixels
            width: Image width in pixels
            output_dir: Directory to save point cloud files
        """
        self.model = model
        self.cam_name = cam_name
        self.height = height
        self.width = width
        self.output_dir = output_dir
        
        # Get camera ID
        try:
            self.cam_id = model.camera(cam_name).id
        except:
            raise ValueError(f"Camera '{cam_name}' not found in model")
        
        # Initialize renderer
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        
        # Calculate camera intrinsics
        fovy = model.cam_fovy[self.cam_id]
        self.f = height / (2 * np.tan(np.deg2rad(fovy / 2)))
        self.cx, self.cy = width / 2, height / 2
        
        # Create meshgrids for projection
        self.i, self.j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Initialized point cloud generator with camera '{cam_name}'")
        print(f"Image resolution: {width}x{height}")
        print(f"Focal length: {self.f:.2f}")
    
    def generate_point_cloud(self, data: mujoco.MjData, 
                           max_depth: float = 10.0,
                           downsample_factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate point cloud from current scene
        
        Args:
            data: MuJoCo data
            max_depth: Maximum depth to include in point cloud
            downsample_factor: Factor to downsample the point cloud (1 = no downsampling)
            
        Returns:
            points: 3D points in world coordinates (N, 3)
            colors: RGB colors (N, 3)
        """
        # Update scene and render
        mujoco.mj_forward(self.model, data)
        self.renderer.update_scene(data, camera=self.cam_name)
        
        # Render RGB and depth
        rgb = self.renderer.render()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        
        # Get camera pose
        cam_pos = data.cam_xpos[self.cam_id]
        cam_mat = data.cam_xmat[self.cam_id].reshape(3, 3)
        
        # Project to 3D camera coordinates
        z = depth
        x = (self.i - self.cx) * z / self.f
        y = (self.j - self.cy) * z / self.f
        points_cam = np.stack((x, -y, -z), axis=-1).reshape(-1, 3)
        
        # Transform to world coordinates
        points_world = (cam_mat @ points_cam.T).T + cam_pos
        
        # Flatten RGB
        rgb_flat = rgb.reshape(-1, 3)
        
        # Filter valid points
        valid_mask = ~(np.isnan(points_world).any(axis=1) | np.isinf(points_world).any(axis=1))
        valid_mask &= np.abs(points_world[:, 2]) < max_depth
        valid_mask &= z.flatten() > 0.01  # Remove very close points
        
        points = points_world[valid_mask]
        colors = rgb_flat[valid_mask].astype(np.uint8)
        
        # Downsample if requested
        if downsample_factor > 1:
            indices = np.arange(0, len(points), downsample_factor)
            points = points[indices]
            colors = colors[indices]
        
        return points, colors
    
    def save_pcd_ascii(self, points: np.ndarray, colors: np.ndarray, 
                      filename: str) -> str:
        """
        Save point cloud as ASCII PCD file
        
        Args:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3)
            filename: Output filename
            
        Returns:
            Full path to saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Combine points and colors
        pcd_data = np.hstack((points, colors))
        
        with open(output_path, "w") as f:
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z r g b\n")
            f.write("SIZE 4 4 4 1 1 1\n")
            f.write("TYPE F F F U U U\n")
            f.write("COUNT 1 1 1 1 1 1\n")
            f.write(f"WIDTH {pcd_data.shape[0]}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {pcd_data.shape[0]}\n")
            f.write("DATA ascii\n")
            for row in pcd_data:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {int(row[3])} {int(row[4])} {int(row[5])}\n")
        
        return output_path
    
    def save_pcd_binary(self, points: np.ndarray, colors: np.ndarray, 
                       filename: str) -> str:
        """
        Save point cloud using Open3D (binary format)
        
        Args:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3)
            filename: Output filename
            
        Returns:
            Full path to saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Create Open3D point cloud
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        # Save
        o3d.io.write_point_cloud(output_path, pcd_o3d)
        
        return output_path
    
    def visualize_point_cloud(self, points: np.ndarray, colors: np.ndarray):
        """
        Visualize point cloud using Open3D
        
        Args:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3)
        """
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        o3d.visualization.draw_geometries([pcd_o3d],
                                          zoom=0.7,
                                          front=[0.0, 0.0, 1.0],
                                          lookat=[0.0, 0.0, 0.0],
                                          up=[0.0, 1.0, 0.0])
    
    def capture_sequence(self, data: mujoco.MjData, 
                        num_frames: int = 10,
                        interval: float = 0.1,
                        prefix: str = "frame") -> list:
        """
        Capture a sequence of point clouds
        
        Args:
            data: MuJoCo data
            num_frames: Number of frames to capture
            interval: Time interval between captures
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        for i in range(num_frames):
            # Generate point cloud
            points, colors = self.generate_point_cloud(data)
            
            # Save
            filename = f"{prefix}_{i:04d}.pcd"
            filepath = self.save_pcd_binary(points, colors, filename)
            saved_files.append(filepath)
            
            print(f"Captured frame {i+1}/{num_frames}: {len(points)} points")
            
            # Wait
            time.sleep(interval)
        
        return saved_files


def integrate_with_cem_planner(xml_path: str, cam_name: str = "camera1"):
    """
    Integration example with CEM planner
    """
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Initialize point cloud generator
    pcd_gen = MuJoCoPointCloudGenerator(model, cam_name=cam_name)
    
    # Set initial robot configuration (example)
    if model.nq >= 6:
        data.qpos[:6] = [1.5, -1.8, 1.75, -1.25, -1.6, 0]
    
    # Generate initial point cloud
    points, colors = pcd_gen.generate_point_cloud(data)
    
    print(f"Generated point cloud with {len(points)} points")
    
    # Save point cloud
    saved_path = pcd_gen.save_pcd_ascii(points, colors, "initial_scene.pcd")
    print(f"Saved point cloud to: {saved_path}")
    
    # Visualize (uncomment to show)
    # pcd_gen.visualize_point_cloud(points, colors)
    
    return pcd_gen, points, colors


def demo_point_cloud_generation():
    """
    Demo function to test point cloud generation
    """
    # Example usage - replace with your actual scene.xml path
    xml_path = "./sampling_based_planner/ur5e_hande_mjx/scene.xml"
    
    try:
        # Check if file exists
        if not os.path.exists(xml_path):
            print(f"Warning: {xml_path} not found. Using default MuJoCo scene.")
            # Create a simple test scene
            xml_content = """
            <mujoco>
                <worldbody>
                    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                    <body pos="0 0 1">
                        <joint type="free"/>
                        <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
                    </body>
                    <camera name="camera1" pos="2 2 2" xyaxes="-1 1 0 0 0 1"/>
                </worldbody>
            </mujoco>
            """
            # Save temporary XML
            temp_xml = "temp_scene.xml"
            with open(temp_xml, "w") as f:
                f.write(xml_content)
            xml_path = temp_xml
        
        # Load model and create data
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        pcd_gen, points, colors = integrate_with_cem_planner(xml_path, "camera1")
        
        # # Initialize point cloud generator
        #pcd_gen = MuJoCoPointCloudGenerator(model, cam_name="camera1")
        
        # # Generate point cloud
        start_time = time.time()
        #points, colors = pcd_gen.generate_point_cloud(data)
        generation_time = time.time() - start_time
        
        print(f"Point cloud generation took {generation_time:.3f} seconds")
        print(f"Generated {len(points)} points")
        
        # Save in both formats
        ascii_path = pcd_gen.save_pcd_ascii(points, colors, "demo_ascii.pcd")
        binary_path = pcd_gen.save_pcd_binary(points, colors, "demo_binary.pcd")
        
        print(f"Saved ASCII PCD: {ascii_path}")
        print(f"Saved binary PCD: {binary_path}")
        
        # Show statistics
        print(f"Point cloud bounds:")
        print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        # Visualize (comment out if running headless)
        try:
            pcd_gen.visualize_point_cloud(points, colors)
        except Exception as e:
            print(f"Visualization failed (likely running headless): {e}")
        
        # Clean up temporary file
        if xml_path == "temp_scene.xml":
            os.remove(xml_path)
            
    except Exception as e:
        print(f"Error in demo: {e}")
        return None
    
    return pcd_gen


if __name__ == "__main__":
    demo_point_cloud_generation()