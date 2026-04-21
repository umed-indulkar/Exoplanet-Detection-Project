# 🤖 PROMPT FOR AI IMAGE GENERATION - EXOPLANET DETECTION PROJECT ARCHITECTURE

## 📋 INSTRUCTION FOR CHATGPT/DALL-E:
Create a detailed technical diagram/illustration for a PowerPoint presentation showing the Exoplanet Detection System architecture. This should be a clean, professional, educational infographic-style image suitable for academic presentation.

---

## 🎯 MAIN COMPONENTS TO ILLUSTRATE:

### **1. COMPLETE PROJECT PIPELINE (Top Section - Flow Diagram)**
Show a horizontal flow diagram from left to right:

**Stage 1: Data Acquisition**
- NASA Kepler Space Telescope (icon/image)
- Light curve collection (waveform icon)
- Raw FITS files (.tbl format)
- Star ID: KIC 2018261 (example)

**Stage 2: Data Preprocessing**
- TIME and PDCSAP_FLUX extraction
- Quarter information (multiple colored segments showing 18 quarters)
- NaN removal and cleaning
- Quarter column showing sensor changes (different colors)

**Stage 3: Phase Folding & Binning**
- Input: Raw light curve (irregular time series)
- Period and Epoch parameters (tce_period, tce_time0bk)
- Phase folding formula: ((TIME - EPOCH) / PERIOD) % 1.0
- Output: Phase-folded curve (0 to 1 phase)
- 500-bin processing (uniform bins)
- Visual show: Before (messy) → After (clean, periodic)

**Stage 4: Model Input Preparation**
- Normalization: flux / median(flux)
- StandardScaler transformation
- 500-dimensional vector input
- Shape: (1, 500) tensor

---

### **2. SIAMESE NEURAL NETWORK (Center - Large, Detailed)**
This is the HIGHEST ACCURACY MODEL (81.73%)

**Architecture Details:**
```
INPUT (500-dim vector)
    ↓
┌─────────────────────────┐
│ ENCODER NETWORK         │
│                         │
│ Linear: 500 → 256       │
│ BatchNorm1d(256)        │
│ ReLU                    │
│ Dropout(0.3)            │
│         ↓               │
│ Linear: 256 → 128       │
│ BatchNorm1d(128)        │
│ ReLU                    │
│ Dropout(0.2)            │
│         ↓               │
│ Linear: 128 → 128       │
│ BatchNorm1d(128)        │
│ (EMBEDDING VECTOR)      │
└─────────────────────────┘
    ↓
EMBEDDING VECTOR (128-dim)
    ↓
┌─────────────────────────┐
│ CONTRASTIVE LOSS        │
│ (During Training)       │
│                         │
│ Loss = (1-Y)*D² +      │
│        Y*max(0,m-D)²    │
│                         │
│ Y=1: Similar pairs      │
│ Y=0: Dissimilar pairs   │
│ D = Euclidean distance  │
│ m = Margin (typically 1)│
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ PREDICTION              │
│                         │
│ Embedding Norm =       │
│ ||vector||              │
│                         │
│ If norm > 0.2622:       │
│   → EXOPLANET           │
│ Else:                   │
│   → NON-EXOPLANET       │
└─────────────────────────┘
```

**Visual Elements:**
- Show two parallel encoder branches (Siamese = twin networks)
- One for "similar" pairs, one for "dissimilar" pairs
- Contrastive loss minimization visualization
- Embedding space visualization (128D → simplified 2D projection)
- Decision boundary at threshold 0.2622

**Key Features:**
- Training on 500-binned raw curves (NOT features)
- Contrastive learning for similarity
- 128-dimensional embedding space
- 81.73% accuracy (best performing)

---

### **3. OTHER MODELS (Right Section - Comparison)**

#### **Baseline Traditional ML Models:**
```
┌─────────────────────────┐
│ TRADITIONAL ML          │
│ (Feature-Based)         │
│                         │
│ Input: 958 features     │
│                         │
│ • Random Forest         │
│   - Ensemble trees        │
│   - Feature importance    │
│                         │
│ • XGBoost (75.53%)       │
│   - Gradient boosting     │
│   - Best baseline         │
│                         │
│ • Logistic Regression     │
│   - Linear classifier     │
│   - Baseline comparison   │
└─────────────────────────┘
```

#### **Neural Network Models:**
```
┌─────────────────────────┐
│ NEURAL NETWORKS         │
│ (Feature-Based)         │
│                         │
│ Input: 958 features       │
│                         │
│ • Feedforward NN (FFNN) │
│   - 3 layers:           │
│     958→128→64→1        │
│   - ReLU + Dropout      │
│   - Sigmoid output      │
│                         │
│ • 1D-CNN                │
│   - Conv1D layers       │
│   - MaxPool + BatchNorm │
│   - Feature extraction  │
│   - Classification head │
└─────────────────────────┘
```

---

### **4. DATA FLOW VISUALIZATION (Bottom)**

**Visual Comparison:**

```
RAW CURVES (Siamese)          vs          FEATURES (Others)
    ↓                                       ↓
┌─────────┐                             ┌─────────┐
│ 500 bins│                             │ 958     │
│ time    │                             │ stats   │
│ series  │                             │ features│
│         │                             │         │
│ Flux    │                             │ Mean    │
│ values  │                             │ Std     │
│ over    │                             │ Trend   │
│ phase   │                             │ Shape   │
│         │                             │ etc.    │
└─────────┘                             └─────────┘
    ↓                                       ↓
┌─────────┐                             ┌─────────┐
│ Siamese │                             │ RF/XGB  │
│ 81.73%  │                             │ 75.53%  │
│ BEST    │                             │ Good    │
└─────────┘                             └─────────┘
```

---

### **5. TRAINING PROCESS (Side Panel)**

**Siamese Training:**
- Contrastive Loss function
- Similar pairs (positive examples)
- Dissimilar pairs (negative examples)
- Distance minimization for similar, maximization for dissimilar
- Batch processing
- 500 epochs (example)

**Other Models Training:**
- Binary Cross-Entropy loss
- Standard supervised learning
- Train/test split
- StandardScaler preprocessing

---

## 🎨 VISUAL STYLE REQUIREMENTS:

### **Color Scheme:**
- **Primary:** Deep space blue (#1a237e)
- **Secondary:** Bright teal/cyan (#00bcd4) for neural network connections
- **Accent:** Gold/orange (#ff9800) for highest accuracy model (Siamese)
- **Background:** Dark space gradient (black to deep blue)
- **Text:** White and light blue for readability
- **Data flow:** Glowing cyan arrows

### **Layout:**
- **Top 20%:** Data Pipeline (horizontal flow)
- **Middle 40%:** Siamese Network (center, large, detailed)
- **Right 20%:** Other Models (comparison)
- **Bottom 20%:** Results/Performance metrics

### **Icons/Symbols:**
- Telescope/satellite icon for data acquisition
- Waveform/light curve icon
- Neural network nodes (circles with connections)
- Binary digits (0/1) for classification
- Percentage badges (81.73%, 75.53%, etc.)
- Stars and space background elements

### **Typography:**
- Bold, clear headers
- Monospace font for code/architecture details
- Readable sizes for presentation

---

## 📊 SPECIFIC METRICS TO DISPLAY:

**Performance Comparison:**
- 🥇 **Siamese Network: 81.73%** (Gold badge - HIGHEST)
- 🥈 **XGBoost: 75.53%** (Silver badge)
- 🥉 **Original Siamese: 75.06%** (Bronze badge)
- **Random Forest, Logistic Regression, FFNN, CNN** (listed below)

**Data Statistics:**
- 65,264 data points analyzed
- 18 quarters (sensor changes)
- 500 bins per light curve
- 128-dimensional embeddings
- ~4 years observation time

---

## 🎯 KEY MESSAGES TO CONVEY:

1. **Siamese is the BEST** (81.73% accuracy)
2. **Raw curves > Features** for this problem
3. **Contrastive learning** for similarity detection
4. **Complete pipeline** from NASA data to prediction
5. **Quarter handling** for sensor changes
6. **Multiple models** compared systematically

---

## 🖼️ IMAGE FORMAT:
- **Aspect Ratio:** 16:9 (standard presentation)
- **Resolution:** High quality (1920x1080 minimum)
- **Style:** Professional, educational, technical infographic
- **Background:** Space-themed (stars, nebula, subtle)
- **Format:** Clean, uncluttered, suitable for academic presentation

---

## ✨ FINAL OUTPUT:
Create a single comprehensive image that can be used as a standalone PPT slide explaining the entire Exoplanet Detection project, with the Siamese Neural Network as the hero/central focus, and clear visual hierarchy showing why it's the best performing model.

The image should make viewers immediately understand:
1. The data flow from telescope to prediction
2. Why Siamese network architecture is special
3. How it compares to traditional methods
4. The complete end-to-end pipeline

---

**KEYWORDS FOR IMAGE GENERATION:**
Siamese neural network, contrastive learning, deep learning, exoplanet detection, Kepler telescope, light curves, machine learning, data pipeline, neural network architecture, embedding space, similarity learning, astronomy, space technology, technical diagram, infographic, educational
