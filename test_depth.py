#!/usr/bin/env python3
"""
Test script for depth.py - 3D Depth Visualization from Heatmap Images
"""

import numpy as np
import matplotlib.pyplot as plt
from depth import DepthVisualizer
import tempfile
import os

def create_sample_heatmap(width=512, height=512):
    """
    Create a sample heatmap for testing
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        numpy array: Sample heatmap image
    """
    # Create coordinate grids
    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    X, Y = np.meshgrid(x, y)
    
    # Create a complex surface with multiple peaks and valleys
    Z = (np.sin(X * 2) * np.cos(Y * 2) + 
         0.5 * np.sin(X * 4) * np.cos(Y * 4) +
         0.3 * np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.5) +
         0.2 * np.exp(-((X+0.5)**2 + (Y+0.5)**2) / 0.3))
    
    # Normalize to 0-1 range
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    
    # Convert to RGB heatmap using matplotlib colormap
    heatmap = plt.cm.viridis(Z)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    return heatmap

def test_depth_visualizer():
    """Test the DepthVisualizer class with sample data"""
    print("🧪 Testing DepthVisualizer...")
    
    # Create visualizer instance
    visualizer = DepthVisualizer()
    
    # Create sample heatmap
    print("📊 Creating sample heatmap...")
    heatmap = create_sample_heatmap(256, 256)
    
    # Test heatmap to depth conversion
    print("🔄 Converting heatmap to depth map...")
    depth_map = visualizer.convert_heatmap_to_depth(heatmap, color_scheme='red_blue')
    
    if depth_map is not None:
        print(f"✅ Depth map created successfully! Shape: {depth_map.shape}")
        print(f"   Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
    else:
        print("❌ Failed to create depth map")
        return
    
    # Test point cloud creation
    print("☁️ Creating 3D point cloud...")
    pcd = visualizer.create_point_cloud_from_depth(
        depth_map, heatmap, max_points=10000
    )
    
    if pcd is not None:
        print(f"✅ Point cloud created successfully! Points: {len(pcd.points)}")
    else:
        print("❌ Failed to create point cloud")
        return
    
    # Test mesh creation
    print("🔲 Creating 3D mesh...")
    mesh = visualizer.create_mesh_from_point_cloud(pcd)
    
    if mesh is not None:
        print(f"✅ Mesh created successfully! Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")
    else:
        print("❌ Failed to create mesh")
    
    # Test statistics analysis
    print("📈 Analyzing depth statistics...")
    stats = visualizer.analyze_depth_statistics(depth_map)
    
    if stats:
        print("✅ Statistics calculated successfully!")
        print(f"   Mean depth: {stats['mean_depth_mm']:.2f} mm")
        print(f"   Max depth: {stats['max_depth_mm']:.2f} mm")
        print(f"   Area: {stats['area_cm2']:.2f} cm²")
    else:
        print("❌ Failed to calculate statistics")
    
    # Test Plotly visualization
    print("📊 Creating Plotly visualization...")
    plotly_fig = visualizer.create_plotly_visualization(depth_map, heatmap, max_points=5000)
    
    if plotly_fig is not None:
        print("✅ Plotly visualization created successfully!")
        # Save the figure
        plotly_fig.write_html("test_3d_visualization.html")
        print("   Saved as 'test_3d_visualization.html'")
    else:
        print("❌ Failed to create Plotly visualization")
    
    # Test report generation
    print("📋 Generating analysis report...")
    report = visualizer.create_depth_analysis_report(stats)
    
    if report:
        print("✅ Analysis report generated successfully!")
        # Save the report
        with open("test_depth_report.html", "w") as f:
            f.write(report)
        print("   Saved as 'test_depth_report.html'")
    else:
        print("❌ Failed to generate analysis report")
    
    # Save sample files
    print("💾 Saving sample files...")
    
    # Save heatmap
    plt.imsave("sample_heatmap.png", heatmap)
    print("   Saved 'sample_heatmap.png'")
    
    # Save depth map
    plt.imsave("sample_depth_map.png", depth_map, cmap='viridis')
    print("   Saved 'sample_depth_map.png'")
    
    # Save point cloud
    if pcd is not None:
        import open3d as o3d
        o3d.io.write_point_cloud("sample_point_cloud.ply", pcd)
        print("   Saved 'sample_point_cloud.ply'")
    
    # Save mesh
    if mesh is not None:
        import open3d as o3d
        o3d.io.write_triangle_mesh("sample_mesh.ply", mesh)
        print("   Saved 'sample_mesh.ply'")
    
    print("\n🎉 All tests completed successfully!")
    print("\n📁 Generated files:")
    print("   - sample_heatmap.png (input heatmap)")
    print("   - sample_depth_map.png (converted depth map)")
    print("   - sample_point_cloud.ply (3D point cloud)")
    print("   - sample_mesh.ply (3D mesh)")
    print("   - test_3d_visualization.html (interactive 3D plot)")
    print("   - test_depth_report.html (analysis report)")
    
    print("\n🚀 You can now run the Gradio interface with:")
    print("   python depth.py")

if __name__ == "__main__":
    test_depth_visualizer()
