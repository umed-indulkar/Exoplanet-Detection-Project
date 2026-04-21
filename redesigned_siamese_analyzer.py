import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import json
from datetime import datetime

class RedesignedSiameseNetwork(nn.Module):
    """
    Redesigned Siamese Network with Classification Head
    
    Problem: Original model uses only embedding norm (magnitude) for classification
    Solution: Add Softmax classification head to use full 128D vector (direction + magnitude)
    
    Training: Hybrid loss (contrastive + cross-entropy)
    Testing: Softmax probabilities (not norm threshold)
    """
    
    def __init__(self, input_size=500, embedding_dim=128, dropout_rate=0.3):
        super(RedesignedSiameseNetwork, self).__init__()
        
        # Shared encoder (same as original Siamese)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # NEW: Classification head for full vector utilization
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(32, 2)  # 2 classes: Exoplanet, Non-Exoplanet
        )
        
    def get_embedding(self, x):
        """Extract 128D embedding vector"""
        return self.encoder(x)
    
    def forward_siamese(self, x1, x2):
        """Siamese forward pass for contrastive training"""
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        return distance, embedding1, embedding2
    
    def forward_classification(self, x):
        """Classification forward pass for inference"""
        embedding = self.encoder(x)
        logits = self.classification_head(embedding)
        probabilities = F.softmax(logits, dim=1)
        return logits, probabilities, embedding

class HybridLoss(nn.Module):
    """
    Hybrid loss combining contrastive learning and classification
    
    L_total = α * L_contrastive + β * L_classification
    
    Contrastive: Learns similarity relationships in embedding space
    Classification: Direct supervision for exoplanet detection
    """
    
    def __init__(self, margin=1.0, alpha=0.7, beta=0.3):
        super(HybridLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha  # Weight for contrastive loss
        self.beta = beta    # Weight for classification loss
        
    def forward(self, distances, pair_labels, classification_logits, true_labels):
        """
        Args:
            distances: Pairwise distances from Siamese network
            pair_labels: Pair labels (1=similar, 0=dissimilar)
            classification_logits: Classification logits
            true_labels: True class labels for each sample
        """
        
        # Contrastive loss (preserves geometric relationships)
        contrastive_loss = torch.mean(
            (pair_labels.float() * torch.pow(distances, 2) + 
             (1 - pair_labels.float()) * torch.pow(F.relu(self.margin - distances), 2))
        )
        
        # Classification loss (direct supervision)
        classification_loss = F.cross_entropy(classification_logits, true_labels)
        
        # Hybrid combination
        total_loss = self.alpha * contrastive_loss + self.beta * classification_loss
        
        return total_loss, contrastive_loss, classification_loss

class RedesignedExoplanetAnalyzer:
    """
    Redesigned exoplanet analyzer with improved Siamese model
    
    Key Improvements:
    1. Uses full 128D embedding vector (not just norm)
    2. Softmax probabilities instead of binary threshold
    3. Hybrid training (contrastive + classification)
    4. Better separation of false positives
    """
    
    def __init__(self):
        self.scaler = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
    def load_model(self, model_path="models/redesigned_siamese.pth", 
                  scaler_path="models/cleaned_scaler.pkl"):
        """Load redesigned model and scaler"""
        
        print("🔄 Loading Redesigned Siamese Model...")
        print("🎯 NEW: Uses full 128D vector + Softmax classification")
        
        # Load scaler
        import joblib
        self.scaler = joblib.load(scaler_path)
        print(f"   ✅ Scaler loaded: {scaler_path}")
        
        # Load model
        self.model = RedesignedSiameseNetwork().to(self.device)
        
        # Try to load saved weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"   ✅ Model loaded: {model_path}")
        else:
            print("   ⚠️ No saved model found, using random initialization")
            print("   📝 Note: Model should be trained before deployment")
        
        self.model.eval()
        
    def predict_with_full_vector(self, flux_processed):
        """
        Prediction using full 128D vector (not just norm)
        
        OLD: ||embedding|| > threshold → binary decision
        NEW: Softmax(full_embedding) → probability scores
        """
        
        with torch.no_grad():
            # Get full embedding and classification
            logits, probabilities, embedding = self.model.forward_classification(
                torch.FloatTensor(flux_processed).to(self.device)
            )
            
            # Extract probabilities
            exoplanet_prob = probabilities[0, 1].item()  # Class 1 = Exoplanet
            non_exoplanet_prob = probabilities[0, 0].item()  # Class 0 = Non-Exoplanet
            
            # Make prediction
            prediction = 1 if exoplanet_prob > 0.5 else 0
            confidence = max(exoplanet_prob, non_exoplanet_prob)
            
            # Also return embedding norm for comparison
            embedding_norm = torch.norm(embedding).item()
            
            return {
                'prediction': prediction,
                'exoplanet_probability': exoplanet_prob,
                'non_exoplanet_probability': non_exoplanet_prob,
                'confidence': confidence,
                'embedding_norm': embedding_norm,
                'embedding_vector': embedding.cpu().numpy()
            }
    
    def explain_improvement(self):
        """Explain the key improvements of the redesigned model"""
        
        print("\n" + "="*60)
        print("🔄 REDESIGNED SIAMESE MODEL - KEY IMPROVEMENTS")
        print("="*60)
        
        print("\n📊 PROBLEM WITH ORIGINAL MODEL:")
        print("   ❌ Used only embedding norm: ||v|| > threshold")
        print("   ❌ Lost directional information in 128D space")
        print("   ❌ Same norm = same prediction, regardless of direction")
        print("   ❌ High false positives for strong signals")
        
        print("\n✅ SOLUTION WITH REDESIGNED MODEL:")
        print("   ✅ Uses full 128D vector: Softmax(embedding)")
        print("   ✅ Preserves both magnitude AND direction")
        print("   ✅ Different vectors with same norm → different predictions")
        print("   ✅ Better separation of false positives")
        
        print("\n🎯 TRAINING IMPROVEMENT:")
        print("   🔄 Hybrid Loss: α × Contrastive + β × Classification")
        print("   📈 Preserves geometric relationships + direct supervision")
        
        print("\n🔬 INFERENCE IMPROVEMENT:")
        print("   📊 OLD: Binary threshold on magnitude")
        print("   📊 NEW: Probability scores from full vector")
        print("   🎯 More expressive decision boundary")
        print("   📈 Better calibrated confidence scores")
        
        print("\n" + "="*60)
    
    def process_light_curve(self, time, flux, period, epoch):
        """Process light curve into 500-binned format"""
        
        # Phase fold
        phase = ((time - epoch) / period) % 1.0
        sort_idx = np.argsort(phase)
        phase_sorted, flux_sorted = phase[sort_idx], flux[sort_idx]
        
        # Bin to 500 points
        bins = np.linspace(0, 1, 501)
        binned_flux = np.zeros(500)
        
        for i in range(500):
            mask = (phase_sorted >= bins[i]) & (phase_sorted < bins[i+1])
            if np.sum(mask) > 0:
                binned_flux[i] = np.median(flux_sorted[mask])
        
        # Handle empty bins
        empty_bins = binned_flux == 0
        if np.any(empty_bins):
            valid_indices = ~empty_bins
            if np.sum(valid_indices) > 1:
                bin_centers = (bins[:-1] + bins[1:]) / 2
                binned_flux[empty_bins] = np.interp(
                    bin_centers[empty_bins], 
                    bin_centers[valid_indices], 
                    binned_flux[valid_indices]
                )
        
        # Normalize and process
        normalized_flux = binned_flux / np.median(binned_flux)
        processed_flux = self.scaler.transform(normalized_flux.reshape(1, -1))
        
        return processed_flux
    
    def analyze_event_redesigned(self, time, flux, period, epoch, event_info):
        """Analyze single event with redesigned model"""
        
        # Process light curve
        processed_flux = self.process_light_curve(time, flux, period, epoch)
        
        # Predict with full vector utilization
        prediction_result = self.predict_with_full_vector(processed_flux)
        
        return {
            'planet_num': event_info['planet_num'],
            'period': period,
            'prediction': prediction_result['prediction'],
            'exoplanet_probability': prediction_result['exoplanet_probability'],
            'non_exoplanet_probability': prediction_result['non_exoplanet_probability'],
            'confidence': prediction_result['confidence'],
            'embedding_norm': prediction_result['embedding_norm'],
            'depth': event_info['depth'],
            'snr': event_info['snr'],
            'duration': event_info['duration'],
            'radius': event_info['radius']
        }
    
    def load_events_csv(self, csv_path):
        """Load TCE events from CSV file"""
        
        print(f"📖 Loading events from CSV: {csv_path}")
        
        try:
            events = pd.read_csv(csv_path, comment='#')
            events_list = []
            
            for _, row in events.iterrows():
                event = {
                    'planet_num': int(row['tce_plnt_num']),
                    'period': float(row['tce_period']),
                    'epoch': float(row['tce_time0bk']),
                    'depth': float(row.get('tce_depth', 0)),
                    'duration': float(row.get('tce_duration', 0)),
                    'snr': float(row.get('tce_model_snr', 0)),
                    'radius': float(row.get('tce_prad', 0))
                }
                events_list.append(event)
            
            print(f"   ✅ Loaded {len(events_list)} events")
            return events_list
            
        except Exception as e:
            print(f"   ❌ Error loading CSV: {e}")
            return None
    
    def read_tbl_file(self, tbl_path):
        """Read .tbl light curve file"""
        
        print(f"📖 Reading .tbl file: {tbl_path}")
        
        try:
            with open(tbl_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if '|set|' in line:
                    data_start = i + 2
                    break
            
            data = pd.read_csv(tbl_path, sep='\s+', skiprows=data_start, 
                             names=['QUARTER', 'TIME', 'PDCSAP_FLUX'], comment='#')
            
            time = data['TIME'].values
            flux = data['PDCSAP_FLUX'].values
            quarter = data['QUARTER'].values
            
            mask = ~np.isnan(time) & ~np.isnan(flux)
            time, flux, quarter = time[mask], flux[mask], quarter[mask]
            
            print(f"   ✅ Loaded {len(time)} data points")
            return time, flux, quarter
            
        except Exception as e:
            print(f"   ❌ Error reading .tbl file: {e}")
            return None, None, None
    
    def run_redesigned_analysis(self, tbl_path=None, csv_path=None):
        """Run complete redesigned analysis"""
        
        print("🌍 REDESIGNED EXOPLANET ANALYSIS SYSTEM")
        print("=" * 60)
        print("🔄 Siamese Network + Classification Head")
        print("🎯 Full 128D Vector Utilization (Not Just Norm)")
        
        # Explain improvements
        self.explain_improvement()
        
        # Load model
        try:
            self.load_model()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
        
        # Get files
        if not tbl_path or not csv_path:
            tbl_files = [f for f in os.listdir('user_data') if f.endswith('.tbl')]
            csv_files = [f for f in os.listdir('user_data') if f.endswith('.csv')]
            
            if not tbl_files or not csv_files:
                print("❌ No .tbl or .csv files found in user_data/")
                return
            
            tbl_path = f"user_data/{tbl_files[0]}"
            csv_path = f"user_data/{csv_files[0]}"
        
        # Read data
        time, flux, quarter = self.read_tbl_file(tbl_path)
        events = self.load_events_csv(csv_path)
        
        if time is None or flux is None or not events:
            print("❌ Failed to load data")
            return
        
        # Analyze each event
        print(f"\n🔍 Analyzing {len(events)} events with redesigned model...")
        print("📊 Using Softmax probabilities from full 128D embeddings")
        
        for event in events:
            result = self.analyze_event_redesigned(time, flux, event['period'], event['epoch'], event)
            self.results.append(result)
            
            pred_str = "🌍 EXOPLANET" if result['prediction'] == 1 else "⭐ NON-EXOPLANET"
            print(f"   Planet {result['planet_num']}: {pred_str}")
            print(f"      P(Exoplanet): {result['exoplanet_probability']:.3f}")
            print(f"      P(Non-Exoplanet): {result['non_exoplanet_probability']:.3f}")
            print(f"      Confidence: {result['confidence']:.3f}")
            print(f"      Embedding Norm: {result['embedding_norm']:.4f}")
        
        # Generate report
        self.generate_redesigned_report(tbl_path, csv_path)
        
        print(f"\n🎉 REDESIGNED ANALYSIS COMPLETE!")
        print(f"📄 Report saved: user_data/redesigned_exoplanet_analysis_report.md")
    
    def generate_redesigned_report(self, tbl_path, csv_path):
        """Generate redesigned analysis report"""
        
        star_id = tbl_path.split('/')[-1].replace('.tbl', '')
        
        # Calculate statistics
        total = len(self.results)
        exoplanets = sum(1 for r in self.results if r['prediction'] == 1)
        non_exoplanets = total - exoplanets
        
        # Probability analysis
        exoplanet_probs = [r['exoplanet_probability'] for r in self.results]
        avg_exoplanet_prob = np.mean(exoplanet_probs)
        
        report = f"""# 🌍 REDESIGNED SIAMESE EXOPLANET ANALYSIS REPORT
## Star KIC {star_id}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** Redesigned Siamese Network + Classification Head
**Key Innovation:** Full 128D Vector Utilization (Not Just Norm)

---

## 🎯 MODEL REDESIGN SUMMARY

### ❌ **Original Problem:**
- Used only embedding norm: ||v|| > threshold
- Lost directional information in 128D space
- Same norm = same prediction regardless of direction
- High false positives for strong signals

### ✅ **Redesigned Solution:**
- Uses full 128D vector: Softmax(embedding)
- Preserves both magnitude AND direction
- Different vectors with same norm → different predictions
- Better separation of false positives

---

## 📊 ANALYSIS RESULTS

### Summary Statistics:
- **Total Events:** {total}
- **Exoplanet Candidates:** {exoplanets} ({exoplanets/total*100:.1f}%)
- **Non-Exoplanets:** {non_exoplanets} ({non_exoplanets/total*100:.1f}%)
- **Average Exoplanet Probability:** {avg_exoplanet_prob:.3f}

### Detailed Results:

| Planet | Period (days) | Prediction | P(Exoplanet) | P(Non-Exoplanet) | Confidence | SNR | Depth (ppm) | Embedding Norm |
|--------|---------------|------------|--------------|------------------|------------|-----|-------------|----------------|
"""
        
        for r in self.results:
            pred_str = "🌍 EXOPLANET" if r['prediction'] == 1 else "⭐ NON-EXOPLANET"
            
            # Status assessment based on probability
            if r['exoplanet_probability'] > 0.8:
                status = "✅ Strong Candidate"
            elif r['exoplanet_probability'] > 0.6:
                status = "❓ Moderate Candidate"
            elif r['exoplanet_probability'] > 0.4:
                status = "❓ Weak Candidate"
            else:
                status = "⭐ Non-Exoplanet"
            
            report += f"| {r['planet_num']} | {r['period']:.3f} | {pred_str} | {r['exoplanet_probability']:.3f} | {r['non_exoplanet_probability']:.3f} | {r['confidence']:.3f} | {r['snr']:.1f} | {r['depth']:.1f} | {r['embedding_norm']:.4f} |\n"
        
        report += f"""

---

## 🔍 TECHNICAL ARCHITECTURE

### **Redesigned Model Structure:**
```
Input (500-dim light curve)
    ↓
Shared Encoder (500 → 256 → 128 → 128)
    ↓
128D Embedding Vector
    ↓
┌─────────────────────────┐
│ Classification Head     │
│                         │
│ Linear: 128 → 64        │
│ ReLU + Dropout          │
│ Linear: 64 → 32         │
│ ReLU + Dropout          │
│ Linear: 32 → 2          │
│ Softmax                 │
│                         │
│ Output:                 │
│ P(Exoplanet)            │
│ P(Non-Exoplanet)        │
└─────────────────────────┘
```

### **Hybrid Training Loss:**
```
L_total = α × L_contrastive + β × L_classification

Where:
- L_contrastive: Preserves geometric relationships
- L_classification: Direct supervision for detection
- α = 0.7, β = 0.3 (balanced weights)
```

### **Key Improvements:**
1. **Full Vector Utilization:** Uses complete 128D embedding
2. **Probability Outputs:** Softmax instead of binary threshold
3. **Better Separation:** Distinguishes similar-magnitude, different-direction vectors
4. **Calibrated Confidence:** Meaningful probability scores

---

## 📈 EXPECTED PERFORMANCE GAINS

### **Problem Solved:**
- **Before:** Embedding norm only → geometric information loss
- **After:** Full vector → geometric information preserved

### **Expected Benefits:**
- **Reduced False Positives:** Better separation of eclipsing binaries
- **Improved Precision:** Probability calibration
- **Better Recall:** Preserves learned relationships
- **Enhanced Interpretability:** Clear probability scores

---

## ⚠️ IMPORTANT NOTES

**Model Status:** This redesigned architecture addresses the fundamental limitation 
of the original Siamese approach. For production use:

1. **Training Required:** Model should be trained with hybrid loss
2. **Validation Needed:** Cross-validation on full dataset
3. **Calibration:** Probability threshold tuning
4. **Comparison:** ROC-AUC analysis vs original system

**Technical Innovation:** The redesign preserves the benefits of Siamese similarity 
learning while making the final decision boundary depend on both embedding 
direction and magnitude, rather than only radial distance from the origin.

---

*Report generated by Redesigned Exoplanet Analysis System*
"""
        
        # Save report
        report_file = f"user_data/redesigned_exoplanet_analysis_report_{star_id}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

def main():
    """Main function for redesigned analysis system"""
    analyzer = RedesignedExoplanetAnalyzer()
    analyzer.run_redesigned_analysis()

if __name__ == "__main__":
    main()
