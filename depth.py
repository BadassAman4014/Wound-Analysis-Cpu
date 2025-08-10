import gradio as gr
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import tempfile
import cv2
import plotly.graph_objects as go
import plotly.express as px
from scipy import ndimage
import os

class DepthVisualizer:
    def __init__(self):
        """Initialize the depth visualizer with default parameters"""
        self.default_focal_length = 470.4
        self.default_pixel_spacing = 0.5  # mm per pixel
        self.default_max_depth = 15.0  # mm
        
    def load_heatmap_image(self, image_path):
        """
        Load and preprocess heatmap image
        
        Args:
            image_path: Path to the heatmap image
            
        Returns:
            numpy array: Processed heatmap image
        """
        if image_path is None:
            return None
            
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Handle PIL Image or numpy array
            if hasattr(image_path, 'convert'):
                image = np.array(image_path)
            else:
                image = image_path
                
        return image
    
    def convert_heatmap_to_depth(self, heatmap_image, color_scheme='red_blue', invert_depth=False):
        """
        Convert heatmap image to depth map based on color intensity
        
        Args:
            heatmap_image: Input heatmap image
            color_scheme: Color scheme interpretation ('red_blue', 'grayscale', 'custom')
            invert_depth: Whether to invert the depth values
            
        Returns:
            numpy array: Depth map
        """
        if heatmap_image is None:
            return None
            
        # Convert to grayscale if RGB
        if len(heatmap_image.shape) == 3:
            if color_scheme == 'red_blue':
                # Extract red channel for red-blue heatmaps (red = high, blue = low)
                depth_map = heatmap_image[:, :, 0].astype(np.float32)
            elif color_scheme == 'grayscale':
                # Use grayscale intensity
                depth_map = np.mean(heatmap_image, axis=2).astype(np.float32)
            elif color_scheme == 'custom':
                # Custom color mapping - use HSV to extract value
                hsv = cv2.cvtColor(heatmap_image, cv2.COLOR_RGB2HSV)
                depth_map = hsv[:, :, 2].astype(np.float32)
            else:
                # Default to red channel
                depth_map = heatmap_image[:, :, 0].astype(np.float32)
        else:
            depth_map = heatmap_image.astype(np.float32)
        
        # Normalize depth map
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        # Invert if requested
        if invert_depth:
            depth_map = 1.0 - depth_map
            
        return depth_map
    
    def create_point_cloud_from_depth(self, depth_map, image=None, focal_length=470.4, 
                                    pixel_spacing=0.5, max_points=50000, scale_factor=1.0):
        """
        Create 3D point cloud from depth map
        
        Args:
            depth_map: Depth map (normalized 0-1)
            image: Original image for colors (optional)
            focal_length: Camera focal length in pixels
            pixel_spacing: Physical pixel spacing in mm
            max_points: Maximum number of points to include
            scale_factor: Scale factor for depth values
            
        Returns:
            open3d.geometry.PointCloud: 3D point cloud
        """
        if depth_map is None:
            return None
            
        h, w = depth_map.shape
        
        # Calculate step size to achieve desired number of points
        step = max(1, int(np.sqrt(h * w / max_points)))
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h:step, 0:w:step]
        
        # Convert to camera coordinates
        x_cam = (x_coords - w / 2) / focal_length
        y_cam = (y_coords - h / 2) / focal_length
        
        # Get depth values
        depth_values = depth_map[::step, ::step] * scale_factor
        
        # Calculate 3D points
        x_3d = x_cam * depth_values * pixel_spacing
        y_3d = y_cam * depth_values * pixel_spacing
        z_3d = depth_values * pixel_spacing
        
        # Flatten arrays
        points = np.stack([x_3d.flatten(), y_3d.flatten(), z_3d.flatten()], axis=1)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add colors if image is provided
        if image is not None:
            image_colors = image[::step, ::step, :]
            colors = image_colors.reshape(-1, 3) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Use depth-based coloring
            depth_colors = plt.cm.viridis(depth_values.flatten())[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(depth_colors)
        
        return pcd
    
    def create_mesh_from_point_cloud(self, pcd, depth=8, width=0.01):
        """
        Create mesh from point cloud using Poisson reconstruction
        
        Args:
            pcd: Point cloud
            depth: Poisson reconstruction depth
            width: Filter width for outlier removal
            
        Returns:
            open3d.geometry.TriangleMesh: 3D mesh
        """
        if pcd is None or len(pcd.points) == 0:
            return None
            
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=30)
        
        # Create mesh using Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        
        # Remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        return mesh
    
    def create_plotly_visualization(self, depth_map, image=None, max_points=20000):
        """
        Create interactive 3D visualization using Plotly
        
        Args:
            depth_map: Depth map
            image: Original image for colors
            max_points: Maximum number of points to display
            
        Returns:
            plotly.graph_objects.Figure: 3D scatter plot
        """
        if depth_map is None:
            return None
            
        h, w = depth_map.shape
        
        # Downsample for performance
        step = max(1, int(np.sqrt(h * w / max_points)))
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h:step, 0:w:step]
        
        # Convert to camera coordinates
        focal_length = 470.4
        x_cam = (x_coords - w / 2) / focal_length
        y_cam = (y_coords - h / 2) / focal_length
        
        # Get depth values
        depth_values = depth_map[::step, ::step]
        
        # Calculate 3D points
        x_3d = x_cam * depth_values
        y_3d = y_cam * depth_values
        z_3d = depth_values
        
        # Flatten arrays
        x_flat = x_3d.flatten()
        y_flat = y_3d.flatten()
        z_flat = z_3d.flatten()
        
        # Get colors
        if image is not None:
            image_colors = image[::step, ::step, :]
            colors_flat = image_colors.reshape(-1, 3)
        else:
            # Use depth-based coloring
            colors_flat = plt.cm.viridis(depth_values.flatten())[:, :3] * 255
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            mode='markers',
            marker=dict(
                size=2,
                color=colors_flat,
                opacity=0.8
            ),
            hovertemplate='<b>Position:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>' +
                         '<b>Depth:</b> %{z:.2f}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title="3D Depth Visualization",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def analyze_depth_statistics(self, depth_map, pixel_spacing=0.5):
        """
        Analyze depth statistics from depth map
        
        Args:
            depth_map: Depth map
            pixel_spacing: Physical pixel spacing in mm
            
        Returns:
            dict: Depth statistics
        """
        if depth_map is None:
            return {}
            
        # Convert to physical units
        depth_mm = depth_map * pixel_spacing * 10  # Convert to mm
        
        # Calculate statistics
        stats = {
            'min_depth_mm': float(np.min(depth_mm)),
            'max_depth_mm': float(np.max(depth_mm)),
            'mean_depth_mm': float(np.mean(depth_mm)),
            'std_depth_mm': float(np.std(depth_mm)),
            'median_depth_mm': float(np.median(depth_mm)),
            'depth_range_mm': float(np.max(depth_mm) - np.min(depth_mm)),
            'pixel_count': int(depth_map.size),
            'area_cm2': float(depth_map.size * (pixel_spacing / 10) ** 2)
        }
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            stats[f'p{p}_depth_mm'] = float(np.percentile(depth_mm, p))
        
        return stats
    
    def create_depth_analysis_report(self, stats):
        """
        Create HTML report from depth statistics
        
        Args:
            stats: Depth statistics dictionary
            
        Returns:
            str: HTML formatted report
        """
        if not stats:
            return "<p>No depth data available for analysis.</p>"
        
        report = f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; color: white;">
            <h3 style="color: #4CAF50; margin-top: 0;">📊 Depth Analysis Report</h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <div>
                    <h4 style="color: #2196F3;">📏 Basic Measurements</h4>
                    <p><strong>Min Depth:</strong> {stats['min_depth_mm']:.2f} mm</p>
                    <p><strong>Max Depth:</strong> {stats['max_depth_mm']:.2f} mm</p>
                    <p><strong>Mean Depth:</strong> {stats['mean_depth_mm']:.2f} mm</p>
                    <p><strong>Depth Range:</strong> {stats['depth_range_mm']:.2f} mm</p>
                    <p><strong>Standard Deviation:</strong> {stats['std_depth_mm']:.2f} mm</p>
                </div>
                
                <div>
                    <h4 style="color: #FF9800;">📈 Statistical Analysis</h4>
                    <p><strong>Median Depth:</strong> {stats['median_depth_mm']:.2f} mm</p>
                    <p><strong>10th Percentile:</strong> {stats['p10_depth_mm']:.2f} mm</p>
                    <p><strong>25th Percentile:</strong> {stats['p25_depth_mm']:.2f} mm</p>
                    <p><strong>75th Percentile:</strong> {stats['p75_depth_mm']:.2f} mm</p>
                    <p><strong>90th Percentile:</strong> {stats['p90_depth_mm']:.2f} mm</p>
                </div>
            </div>
            
            <div style="background-color: #2c2c2c; padding: 15px; border-radius: 8px;">
                <h4 style="color: #9C27B0; margin-top: 0;">🔍 Additional Information</h4>
                <p><strong>Total Pixels:</strong> {stats['pixel_count']:,}</p>
                <p><strong>Estimated Area:</strong> {stats['area_cm2']:.2f} cm²</p>
                <p><strong>Depth Variation:</strong> {stats['std_depth_mm']/stats['mean_depth_mm']*100:.1f}% (coefficient of variation)</p>
            </div>
        </div>
        """
        
        return report

def process_heatmap_image(image, color_scheme, invert_depth, focal_length, 
                         pixel_spacing, max_points, scale_factor, create_mesh):
    """
    Main processing function for heatmap to 3D visualization
    
    Args:
        image: Input heatmap image
        color_scheme: Color scheme interpretation
        invert_depth: Whether to invert depth values
        focal_length: Camera focal length
        pixel_spacing: Physical pixel spacing
        max_points: Maximum number of points
        scale_factor: Depth scale factor
        create_mesh: Whether to create mesh
        
    Returns:
        tuple: (depth_map, point_cloud_file, mesh_file, plotly_fig, stats_report)
    """
    visualizer = DepthVisualizer()
    
    # Load and process image
    heatmap = visualizer.load_heatmap_image(image)
    if heatmap is None:
        return None, None, None, None, "<p>Error: No image provided.</p>"
    
    # Convert to depth map
    depth_map = visualizer.convert_heatmap_to_depth(heatmap, color_scheme, invert_depth)
    if depth_map is None:
        return None, None, None, None, "<p>Error: Failed to convert heatmap to depth map.</p>"
    
    # Create point cloud
    pcd = visualizer.create_point_cloud_from_depth(
        depth_map, heatmap, focal_length, pixel_spacing, max_points, scale_factor
    )
    
    # Save point cloud
    point_cloud_file = None
    if pcd is not None:
        tmp_pcd = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
        o3d.io.write_point_cloud(tmp_pcd.name, pcd)
        point_cloud_file = tmp_pcd.name
    
    # Create mesh if requested
    mesh_file = None
    if create_mesh and pcd is not None:
        mesh = visualizer.create_mesh_from_point_cloud(pcd)
        if mesh is not None:
            tmp_mesh = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
            o3d.io.write_triangle_mesh(tmp_mesh.name, mesh)
            mesh_file = tmp_mesh.name
    
    # Create Plotly visualization
    plotly_fig = visualizer.create_plotly_visualization(depth_map, heatmap, max_points)
    
    # Analyze statistics
    stats = visualizer.analyze_depth_statistics(depth_map, pixel_spacing)
    stats_report = visualizer.create_depth_analysis_report(stats)
    
    return depth_map, point_cloud_file, mesh_file, plotly_fig, stats_report

# Create Gradio interface
def create_gradio_interface():
    """Create the Gradio interface for 3D depth visualization"""
    
    # Custom CSS for dark theme
    css = """
    .gradio-container {
        font-family: 'Segoe UI', sans-serif;
        background-color: #121212;
        color: #ffffff;
        padding: 20px;
    }
    .gr-button {
        background-color: #2c3e50;
        color: white;
        border-radius: 10px;
    }
    .gr-button:hover {
        background-color: #34495e;
    }
    h1 {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 2rem 0;
        color: #ffffff;
    }
    h2 {
        color: #ffffff;
        text-align: center;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(css=css, title="3D Depth Visualization") as demo:
        gr.HTML("<h1>🔄 Heatmap to 3D Depth Visualization</h1>")
        gr.Markdown("### Convert heatmap images to 3D visualizations using Open3D and Plotly")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input Settings")
                
                # Image input
                image_input = gr.Image(
                    label="Upload Heatmap Image",
                    type="pil",
                    height=300
                )
                
                # Processing parameters
                color_scheme = gr.Dropdown(
                    choices=["red_blue", "grayscale", "custom"],
                    value="red_blue",
                    label="Color Scheme Interpretation",
                    info="How to interpret colors in the heatmap"
                )
                
                invert_depth = gr.Checkbox(
                    label="Invert Depth Values",
                    value=False,
                    info="Invert the depth interpretation (useful for inverted heatmaps)"
                )
                
                focal_length = gr.Slider(
                    minimum=100,
                    maximum=1000,
                    value=470.4,
                    step=10,
                    label="Focal Length (pixels)",
                    info="Camera focal length for 3D projection"
                )
                
                pixel_spacing = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.5,
                    step=0.1,
                    label="Pixel Spacing (mm/pixel)",
                    info="Physical pixel spacing for real-world measurements"
                )
                
                max_points = gr.Slider(
                    minimum=1000,
                    maximum=100000,
                    value=20000,
                    step=1000,
                    label="Maximum Points",
                    info="Maximum number of points in 3D visualization"
                )
                
                scale_factor = gr.Slider(
                    minimum=0.1,
                    maximum=10.0,
                    value=1.0,
                    step=0.1,
                    label="Depth Scale Factor",
                    info="Scale factor for depth values"
                )
                
                create_mesh = gr.Checkbox(
                    label="Create 3D Mesh",
                    value=False,
                    info="Generate 3D mesh from point cloud (may take longer)"
                )
                
                process_btn = gr.Button(
                    "🔄 Process Heatmap",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                
                # Depth map output
                depth_output = gr.Image(
                    label="Generated Depth Map",
                    height=300
                )
                
                # 3D visualization
                plotly_output = gr.Plot(
                    label="3D Interactive Visualization"
                )
                
                # Statistics report
                stats_output = gr.HTML(
                    label="Depth Analysis Report"
                )
                
                # File downloads
                gr.Markdown("### 💾 Download Files")
                point_cloud_download = gr.File(
                    label="Point Cloud (.ply)"
                )
                mesh_download = gr.File(
                    label="3D Mesh (.ply)"
                )
        
        # Process button event
        process_btn.click(
            fn=process_heatmap_image,
            inputs=[
                image_input, color_scheme, invert_depth, focal_length,
                pixel_spacing, max_points, scale_factor, create_mesh
            ],
            outputs=[
                depth_output, point_cloud_download, mesh_download,
                plotly_output, stats_output
            ]
        )
        
        # Auto-process on image upload
        image_input.change(
            fn=lambda img: process_heatmap_image(
                img, "red_blue", False, 470.4, 0.5, 20000, 1.0, False
            ),
            inputs=[image_input],
            outputs=[
                depth_output, point_cloud_download, mesh_download,
                plotly_output, stats_output
            ]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )
