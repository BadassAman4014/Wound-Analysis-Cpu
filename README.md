# 🩹 Wound Analysis & Depth Estimation System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/aman4014/Wound-Analysis-Cpu)

A comprehensive AI-powered wound analysis system that combines computer vision, deep learning, and medical imaging to provide detailed wound classification, depth estimation, and severity assessment.

## 🌟 What This Project Does

This system helps medical professionals and researchers analyze wound images by providing:

- **🔍 Wound Classification**: Identifies wound types (Abrasions, Burns, Lacerations, Punctures, Ulcers) with AI-powered reasoning
- **📊 Depth Estimation**: Creates 3D depth maps and point clouds from 2D images
- **🩺 Severity Analysis**: Provides comprehensive medical assessment including depth measurements, area calculations, and AI-powered severity classification
- **📈 3D Visualization**: Generates interactive 3D models for better understanding of wound geometry

## 🚀 Try It Live

**Experience the full system right now!** Click the badge above to access our live demo on Hugging Face Spaces.

## 🎯 Key Features

### 1. **Smart Wound Classification**
- Uses Google's Gemini AI for intelligent wound type identification
- Provides detailed medical reasoning for each classification
- Supports multiple wound types: Abrasions, Burns, Lacerations, Punctures, Ulcers

### 2. **Advanced Depth Estimation**
- Leverages DepthAnythingV2 model for accurate depth mapping
- Generates both grayscale and colored depth maps
- Creates downloadable 3D point clouds (.ply format)
- Interactive 3D visualizations with proper camera projection

### 3. **Comprehensive Severity Analysis**
- **Medical-grade depth measurements** with millimeter precision
- **Area calculations** in square centimeters
- **Volume estimation** for wound cavities
- **AI-powered severity classification** (Superficial to Very Severe)
- **Tissue involvement analysis** (superficial, partial thickness, full thickness, deep)

### 4. **Professional Medical Assessment**
- Combines visual analysis with quantitative measurements
- Provides treatment recommendations based on wound characteristics
- Identifies potential risk factors and complications
- Offers clinical significance and prognosis insights

## 🛠️ Technical Architecture

### **AI Models Used**
- **Google Gemini 2.5 Pro**: For wound classification and medical reasoning
- **DepthAnythingV2**: For depth estimation from single images
- **Custom DeepLabV3+**: For wound segmentation and mask generation
- **TensorFlow/Keras**: For model inference and processing

### **Key Technologies**
- **Gradio**: Beautiful web interface
- **Open3D**: 3D point cloud processing
- **OpenCV**: Image processing and computer vision
- **Plotly**: Interactive 3D visualizations
- **Matplotlib**: Scientific plotting and colormaps

## 📋 Prerequisites

Before running this project, make sure you have:

- **Python 3.8+** installed
- **8GB+ RAM** (recommended for 3D processing)
- **GPU support** (optional, but recommended for faster processing)
- **Google API Key** (for Gemini AI features)

## 🚀 Quick Start

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/Wound-Analysis-Cpu.git
cd Wound-Analysis-Cpu
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Set Up API Keys**
Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. **Run the Application**
```bash
python app.py
```

The application will be available at `http://localhost:7860`

## 📖 How to Use

### **Step 1: Wound Classification**
1. Upload a wound image
2. The AI will automatically classify the wound type
3. Review the detailed reasoning and medical assessment
4. Click "Pass Image to Depth Analysis" to continue

### **Step 2: Depth Estimation**
1. Load your image (or use the one from Step 1)
2. Adjust focal length parameters if needed
3. Click "Compute Depth" to generate depth maps
4. Download grayscale depth maps, raw 16-bit files, or 3D point clouds
5. Explore the interactive 3D visualization

### **Step 3: Severity Analysis**
1. Load the depth map from Step 2
2. The system automatically generates a wound mask
3. Adjust pixel spacing and depth calibration as needed
4. Click "Analyze Severity" for comprehensive medical assessment
5. Review detailed measurements and AI-powered analysis

## 📊 Understanding the Results

### **Wound Classification Results**
- **Classification**: The identified wound type
- **Reasoning**: Detailed explanation of why this classification was made
- **Characteristics**: Key visual features observed
- **Severity Indicators**: Potential risk factors

### **Depth Analysis Results**
- **Mean Depth**: Average wound depth in millimeters
- **Maximum Depth**: Deepest point measurement
- **Area Measurements**: Total wound area and tissue involvement breakdown
- **Volume Estimation**: Approximate wound cavity volume
- **Quality Metrics**: Analysis confidence and data point count

### **Severity Assessment**
- **Severity Level**: From Superficial to Very Severe
- **Medical Assessment**: Comprehensive clinical analysis
- **Treatment Recommendations**: Specific care instructions
- **Risk Factors**: Potential complications to monitor

## 🔧 Configuration Options

### **Depth Estimation Settings**
- **Focal Length**: Adjust based on your camera (default: 470.4 pixels)
- **Point Count**: Control 3D visualization detail (1,000-300,000 points)
- **Download Formats**: Grayscale, raw 16-bit, or 3D point clouds

### **Severity Analysis Settings**
- **Pixel Spacing**: Camera calibration (default: 0.5 mm/pixel)
- **Depth Calibration**: Maximum expected wound depth (5-30mm)
- **Minimum Area**: Filter out noise in segmentation (default: 500 pixels)

## 🏥 Medical Applications

This system is designed for:

- **Healthcare Professionals**: Quick wound assessment and documentation
- **Research Institutions**: Wound healing studies and analysis
- **Medical Education**: Teaching wound classification and assessment
- **Telemedicine**: Remote wound evaluation and monitoring
- **Clinical Trials**: Standardized wound measurement and tracking

## ⚠️ Important Notes

### **Medical Disclaimer**
- This system is for **research and educational purposes only**
- **Not intended for clinical diagnosis or treatment decisions**
- Always consult qualified healthcare professionals for medical advice
- Results should be validated by medical professionals

### **Image Requirements**
- **High-quality images** work best (good lighting, clear focus)
- **Consistent camera distance** improves depth accuracy
- **Multiple angles** can provide better 3D reconstruction
- **Avoid shadows and reflections** for optimal analysis

### **Performance Considerations**
- **GPU acceleration** recommended for faster processing
- **Large images** may take longer to process
- **3D visualizations** require more computational resources
- **Internet connection** needed for AI model inference

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Issues**: Found a bug? Let us know!
2. **Feature Requests**: Have ideas for improvements?
3. **Code Contributions**: Submit pull requests
4. **Documentation**: Help improve our docs
5. **Testing**: Test with different wound types and images

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** for intelligent wound classification
- **DepthAnythingV2** team for depth estimation capabilities
- **Gradio** for the beautiful web interface
- **Open3D** for 3D point cloud processing
- **Medical professionals** who provided domain expertise

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Wound-Analysis-Cpu/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Wound-Analysis-Cpu/discussions)
- **Email**: [your-email@example.com]

## 🔄 Updates

Stay updated with the latest features and improvements:

- **Star the repository** to track updates
- **Watch for releases** for new features
- **Follow our blog** for detailed tutorials and case studies

---

**Made with ❤️ for the medical community**

*This project aims to advance wound care through AI-powered analysis and 3D visualization technology.*
