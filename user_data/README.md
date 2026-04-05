# 📁 User Data Folder

This folder is where you place your data for exoplanet prediction.

## 📋 What to Put Here:

### **1. Your Light Curve File**
- **Format**: `.tbl` file (Kepler format)
- **Columns**: TIME and PDCSAP_FLUX
- **Example**: `star_light_curve.tbl`

### **2. What the System Needs:**
- **Orbital Period**: Days (e.g., 5.678)
- **Epoch Period**: Days (e.g., 1234.567)

## 🔄 Process:
1. Place your `.tbl` file here
2. Run `python predict_exoplanet.py`
3. Enter period and epoch when prompted
4. Get prediction + visualization

## 📊 Output Files:
- `processing_visualization.png` - Shows processing steps
- `prediction_results.json` - Detailed results

## 🌟 Ready to predict exoplanets!
