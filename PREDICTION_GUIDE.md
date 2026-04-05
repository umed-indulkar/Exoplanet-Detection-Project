# 🌍 Exoplanet Detection Prediction System

## 📁 How to Use

### **Step 1: Prepare Your Data**
1. Place your light curve `.tbl` file in the `user_data/` folder
2. Know the orbital period and epoch of the transit event

### **Step 2: Run Prediction**
```bash
python predict_exoplanet.py
```

### **Step 3: Follow Prompts**
```
🌍 EXOPLANET DETECTION PREDICTION SYSTEM
==================================================

📝 INPUT REQUIRED:
Please place your .tbl file in the 'user_data/' folder
and provide the following information:

Found .tbl files: star_light_curve.tbl
Using: user_data/star_light_curve.tbl

Enter orbital period (days): 5.678
Enter epoch period (days): 1234.567
```

### **Step 4: Get Results**
```
🎯 PREDICTION RESULTS
==================================================
Input file: user_data/star_light_curve.tbl
Period: 5.678000 days
Epoch: 1234.567000 days
Prediction: 🌍 EXOPLANET
Confidence: 0.8234
Embedding Norm: 12.4567
```

## 🔧 What the System Does

### **1. Reads Your .tbl File**
- Extracts TIME and PDCSAP_FLUX columns
- Handles different column name formats
- Removes NaN values

### **2. Folds the Light Curve**
- Uses your provided period and epoch
- Creates phase-folded light curve
- Same method as training data

### **3. Bins to 500 Points**
- Exactly matches training preprocessing
- Handles empty bins with interpolation
- Creates uniform 500-point representation

### **4. Normalizes & Scales**
- Normalizes flux to median = 1.0
- Applies same scaler as training
- Perfect match to training data

### **5. Makes Prediction**
- Uses trained Siamese network
- Extracts 128-dimensional embedding
- Classifies as exoplanet/non-exoplanet

### **6. Provides Visualization**
- Shows all processing steps
- Saves to `user_data/processing_visualization.png`
- Helps you understand the transformation

## 📊 Output Files

After running, you'll get:
- `user_data/processing_visualization.png` - Processing steps
- `user_data/prediction_results.json` - Detailed results

## 🎯 Expected Performance

The system uses the same pipeline as training:
- **Training accuracy**: 81.73%
- **Expected prediction accuracy**: Similar range
- **Best for**: Clear transit signals

## ⚠️ Important Notes

1. **File Format**: Must be standard Kepler .tbl format
2. **Column Names**: TIME and PDCSAP_FLUX (or similar)
3. **Parameters**: Accurate period and epoch are crucial
4. **Quality**: Better data = better predictions

## 🔍 Troubleshooting

### **Common Issues:**
- **File not found**: Check file path in `user_data/`
- **Column error**: Verify .tbl has TIME and PDCSAP_FLUX columns
- **Bad prediction**: Check period and epoch accuracy

### **Model Loading Errors:**
- Make sure `models/cleaned_siamese.pth` exists
- Make sure `models/cleaned_scaler.pkl` exists
- Run training scripts if models are missing

## 🚀 Ready to Use!

The system recreates the exact training pipeline:
```
NASA FITS → .tbl file → Phase folding → 500 bins → Siamese network → Prediction
```

**Just provide your .tbl file + period + epoch, and get exoplanet prediction! 🌍**
