# 🔄 Heatmap to 3D Depth Visualization

A powerful tool for converting heatmap images into 3D visualizations using Open3D and Plotly. This module is specifically designed for medical imaging, wound analysis, and any application requiring depth visualization from 2D heatmaps.

## 🚀 Features

- **Heatmap to Depth Conversion**: Convert color-coded heatmaps to depth maps
- **3D Point Cloud Generation**: Create detailed 3D point clouds from depth data
- **Mesh Reconstruction**: Generate 3D meshes using Poisson surface reconstruction
- **Interactive 3D Visualization**: Plotly-based interactive 3D plots
- **Depth Analysis**: Comprehensive statistical analysis of depth data
- **Gradio Interface**: User-friendly web interface for easy interaction
- **Multiple Color Schemes**: Support for red-blue, grayscale, and custom color mappings

## 📋 Requirements

All dependencies are already included in the main `requirements.txt` file:

- `gradio` - Web interface
- `open3d` - 3D point cloud and mesh processing
- `plotly` - Interactive 3D visualization
- `matplotlib` - Image processing and colormaps
- `numpy` - Numerical computations
- `opencv-python` - Image processing
- `scipy` - Scientific computing

## 🛠️ Installation

1. Ensure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

2. The module is ready to use!

## 📖 Usage

### 1. Running the Gradio Interface

Launch the web interface:
```bash
python depth.py
```

The interface will be available at `http://localhost:7861`

### 2. Using the Python API

```python
from depth import DepthVisualizer

# Create visualizer instance
visualizer = DepthVisualizer()

# Load heatmap image
heatmap = visualizer.load_heatmap_image("your_heatmap.png")

# Convert to depth map
depth_map = visualizer.convert_heatmap_to_depth(
    heatmap, 
    color_scheme='red_blue', 
    invert_depth=False
)

# Create 3D point cloud
pcd = visualizer.create_point_cloud_from_depth(
    depth_map, 
    heatmap, 
    max_points=20000
)

# Generate 3D mesh
mesh = visualizer.create_mesh_from_point_cloud(pcd)

# Create interactive visualization
plotly_fig = visualizer.create_plotly_visualization(depth_map, heatmap)

# Analyze statistics
stats = visualizer.analyze_depth_statistics(depth_map)
report = visualizer.create_depth_analysis_report(stats)
```

### 3. Testing the Module

Run the test script to verify functionality:
```bash
python test_depth.py
```

This will generate sample files and test all features.

## 🎯 Interface Features

### Input Settings

- **Upload Heatmap Image**: Upload your heatmap image (PNG, JPG, etc.)
- **Color Scheme Interpretation**: Choose how to interpret colors
  - `red_blue`: Red = high depth, Blue = low depth
  - `grayscale`: Intensity-based depth
  - `custom`: HSV value-based depth
- **Invert Depth Values**: Toggle to invert depth interpretation
- **Focal Length**: Camera focal length for 3D projection (pixels)
- **Pixel Spacing**: Physical pixel spacing (mm/pixel)
- **Maximum Points**: Number of points in 3D visualization
- **Depth Scale Factor**: Scale factor for depth values
- **Create 3D Mesh**: Option to generate 3D mesh (may take longer)

### Output Results

- **Generated Depth Map**: Visual representation of the depth map
- **3D Interactive Visualization**: Plotly-based 3D scatter plot
- **Depth Analysis Report**: Comprehensive statistical analysis
- **Download Files**: 
  - Point Cloud (.ply) - 3D point cloud file
  - 3D Mesh (.ply) - 3D mesh file (if enabled)

## 🔧 Technical Details

### Depth Conversion Methods

1. **Red-Blue Scheme**: Extracts red channel intensity (red = high depth)
2. **Grayscale**: Uses average RGB values
3. **Custom**: Uses HSV value channel for depth interpretation

### 3D Projection

The module uses camera projection to convert 2D depth maps to 3D coordinates:

```
X_3D = (x_pixel - width/2) / focal_length * depth * pixel_spacing
Y_3D = (y_pixel - height/2) / focal_length * depth * pixel_spacing
Z_3D = depth * pixel_spacing
```

### Mesh Reconstruction

Uses Open3D's Poisson surface reconstruction with:
- Normal estimation using KD-tree search
- Consistent normal orientation
- Density-based vertex filtering

## 📊 Analysis Features

### Statistical Measurements

- **Basic Measurements**: Min, max, mean, range, standard deviation
- **Percentiles**: 10th, 25th, 50th (median), 75th, 90th percentiles
- **Area Calculation**: Estimated physical area in cm²
- **Depth Variation**: Coefficient of variation

### Medical Applications

This module is particularly useful for:
- **Wound Analysis**: Depth measurement and visualization
- **Medical Imaging**: 3D reconstruction from 2D scans
- **Research**: Depth analysis and visualization
- **Education**: Interactive 3D demonstrations

## 🎨 Customization

### Color Schemes

You can add custom color schemes by modifying the `convert_heatmap_to_depth` method:

```python
def convert_heatmap_to_depth(self, heatmap_image, color_scheme='custom', invert_depth=False):
    # Add your custom color interpretation logic here
    if color_scheme == 'your_custom_scheme':
        # Your custom implementation
        pass
```

### Visualization Parameters

Adjust visualization parameters in the `create_plotly_visualization` method:
- Point size and opacity
- Camera position and orientation
- Color mapping
- Hover information

## 🔍 Troubleshooting

### Common Issues

1. **No depth map generated**: Check if the heatmap has sufficient color variation
2. **Poor 3D visualization**: Try adjusting focal length or pixel spacing
3. **Mesh creation fails**: Reduce point cloud density or adjust Poisson parameters
4. **Slow performance**: Reduce maximum points or disable mesh creation

### Performance Tips

- Use smaller images for faster processing
- Reduce maximum points for real-time visualization
- Disable mesh creation for faster results
- Use appropriate focal length for your camera setup

## 📁 File Formats

### Input Formats
- PNG, JPG, JPEG, TIFF, BMP
- RGB or grayscale images
- Any size (larger images may be slower)

### Output Formats
- **Point Cloud**: PLY format (compatible with most 3D software)
- **Mesh**: PLY format with vertices and faces
- **Visualization**: HTML file with interactive Plotly plot
- **Report**: HTML file with statistical analysis

## 🤝 Integration

This module can be easily integrated with the main wound analysis system:

```python
# In your main application
from depth import DepthVisualizer

# Use with existing wound analysis pipeline
def analyze_wound_depth(wound_image, heatmap_image):
    visualizer = DepthVisualizer()
    depth_map = visualizer.convert_heatmap_to_depth(heatmap_image)
    # Continue with your analysis...
```

## 📈 Future Enhancements

- Support for more color schemes
- Advanced mesh smoothing options
- Real-time processing capabilities
- Integration with more 3D file formats
- Advanced statistical analysis features

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Run the test script to verify installation
3. Review the example usage in `test_depth.py`

---

**Note**: This module is designed for educational and research purposes. For medical applications, ensure proper validation and calibration of depth measurements.
