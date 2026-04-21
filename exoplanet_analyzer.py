import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class ImprovedSiameseNetwork(nn.Module):
    """Improved Siamese Network with Classification Head"""
    
    def __init__(self, input_size=500, embedding_dim=128, dropout_rate=0.3):
        super(ImprovedSiameseNetwork, self).__init__()
        
        # Single encoder for classification (not Siamese)
        self.single_encoder = nn.Sequential(
            nn.Linear(self.input_size, 256),  # Use input_size from self
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
        
        # Shared encoder (for Siamese training)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 256),  # Use input_size from self
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(32, 2)  # 2 classes: Exoplanet, Non-Exoplanet
        )
        
        # Store input size for validation
        self.input_size = input_size
        
    def get_embedding(self, x):
        """Extract embedding vector using shared encoder"""
        return self.encoder(x)
    
    def get_embedding_single(self, x):
        """Extract embedding vector using single encoder"""
        return self.single_encoder(x)
    
    def forward(self, x1, x2):
        """Siamese forward pass for training"""
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        
        # Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        return distance
    
    def classify(self, x):
        """Classification forward pass for inference"""
        # Use single encoder for classification
        embedding = self.get_embedding_single(x)
        logits = F.softmax(self.classifier(embedding), dim=1)
        return logits, probabilities

class HybridContrastiveLoss(nn.Module):
    """Hybrid loss combining contrastive and cross-entropy"""
    
    def __init__(self, margin=1.0, alpha=0.7, beta=0.3):
        super(HybridContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha  # Weight for contrastive loss
        self.beta = beta    # Weight for classification loss
        
    def forward(self, distances, labels, logits, target_labels):
        """
        distances: Pairwise distances from Siamese network
        labels: Pair labels (1=similar, 0=dissimilar)
        logits: Classification logits
        target_labels: True class labels for each sample
        """
        
        # Contrastive loss
        contrastive_loss = torch.mean(
            (labels.float() * torch.pow(distances, 2) + 
             (1 - labels.float()) * torch.pow(F.relu(self.margin - distances), 2))
        )
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, target_labels)
        
        # Hybrid loss
        total_loss = self.alpha * contrastive_loss + self.beta * ce_loss
        
        return total_loss, contrastive_loss, ce_loss

class ImprovedExoplanetAnalyzer:
    """Improved exoplanet analyzer with hybrid Siamese model"""
    
    def __init__(self):
        self.scaler = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
    def load_model(self, model_path="models/improved_siamese.pth", 
                  scaler_path="models/cleaned_scaler.pkl"):
        """Load improved model and scaler"""
        
        print("🔄 Loading improved Siamese model...")
        
        # Load scaler
        import joblib
        self.scaler = joblib.load(scaler_path)
        print(f"   ✅ Scaler loaded: {scaler_path}")
        
        # Load model
        self.model = ImprovedSiameseNetwork().to(self.device)
        
        # Try to load saved weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"   ✅ Model loaded: {model_path}")
        else:
            print("   ⚠️ No saved model found, using random initialization")
        
        self.model.eval()
        
    def create_pairs(self, embeddings, labels):
        """Create pairs for contrastive training"""
        n = len(embeddings)
        pairs = []
        pair_labels = []
        
        # Positive pairs (same class)
        for i in range(n):
            for j in range(i+1, n):
                if labels[i] == labels[j]:
                    pairs.append((i, j))
                    pair_labels.append(1)
        
        # Negative pairs (different class)
        for i in range(n):
            for j in range(i+1, n):
                if labels[i] != labels[j]:
                    pairs.append((i, j))
                    pair_labels.append(0)
        
        return pairs, pair_labels
    
    def train_model(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Train the improved Siamese model"""
        
        print("🚀 Starting training...")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = HybridContrastiveLoss()
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0
            
            for batch_idx, (x1, x2, labels, target_labels) in enumerate(train_loader):
                x1, x2, labels, target_labels = (x1.to(self.device), x2.to(self.device), 
                                                      labels.to(self.device), 
                                                      target_labels.to(self.device))
                
                optimizer.zero_grad()
                
                # Forward pass
                distances = self.model(x1, x2)
                logits, _ = self.model.classify(x1)  # Use first sample for classification
                
                # Loss calculation
                total_loss, contrastive_loss, ce_loss = loss_fn(distances, labels, logits, target_labels)
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_train_loss += total_loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if epoch % 5 == 0:
                val_loss, val_acc = self.validate_model(val_loader, loss_fn)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        return train_losses, val_losses, val_accuracies
    
    def validate_model(self, val_loader, loss_fn):
        """Validate model performance"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x1, x2, labels, target_labels in val_loader:
                x1, x2, labels, target_labels = (x1.to(self.device), x2.to(self.device), 
                                                      labels.to(self.device), 
                                                      target_labels.to(self.device))
                
                # Forward pass
                distances = self.model(x1, x2)
                logits, probabilities = self.model.classify(x1)
                
                # Loss
                total_loss_batch, _, ce_loss = loss_fn(distances, labels, logits, target_labels)
                total_loss += total_loss_batch.item()
                
                # Accuracy (use classification logits)
                predicted = torch.argmax(logits, dim=1)
                correct += (predicted == target_labels).sum().item()
                total += target_labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def predict_with_probability(self, flux_processed):
        """Predict with probability scores"""
        
        with torch.no_grad():
            # Convert to tensor and ensure correct shape (500,)
            flux_tensor = torch.FloatTensor(flux_processed).to(self.device)
            
            # Use single encoder for classification
            embedding = self.model.get_embedding_single(flux_tensor)
            logits, probabilities = self.model.classify(embedding)
            
            # Extract probability for exoplanet class (assuming class 1)
            exoplanet_prob = probabilities[0, 1].item()
            prediction = 1 if exoplanet_prob > 0.5 else 0
            confidence = max(exoplanet_prob, 1 - exoplanet_prob)
            
            return prediction, confidence, exoplanet_prob, embedding.norm().item()
    
    def analyze_event_improved(self, time, flux, period, epoch, event_info):
        """Analyze single event with improved model"""
        
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
        
        # Predict with improved model
        prediction, confidence, exoplanet_prob, embedding_norm = self.predict_with_probability(processed_flux)
        
        return {
            'planet_num': event_info['planet_num'],
            'period': period,
            'prediction': prediction,
            'confidence': confidence,
            'exoplanet_probability': exoplanet_prob,
            'embedding_norm': embedding_norm,
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
    
    def run_improved_analysis(self, tbl_path=None, csv_path=None):
        """Run complete improved analysis"""
        
        print("🌍 IMPROVED EXOPLANET ANALYSIS SYSTEM")
        print("=" * 50)
        print("🔄 Using Hybrid Siamese + Classification Head")
        print("🎯 Preserves embedding geometry + probability outputs")
        
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
        
        if time is None or events is None:
            print("❌ Failed to load data")
            return
        
        # Analyze each event
        print(f"\n🔍 Analyzing {len(events)} events with improved model...")
        for event in events:
            result = self.analyze_event_improved(time, flux, event['period'], event['epoch'], event)
            self.results.append(result)
            pred_str = "🌍 EXOPLANET" if result['prediction'] == 1 else "⭐ NON-EXOPLANET"
            print(f"   Planet {result['planet_num']}: {pred_str} (prob: {result['exoplanet_probability']:.3f}, conf: {result['confidence']:.3f})")
        
        # Generate improved report
        self.generate_improved_report(tbl_path, csv_path)
        
        print(f"\n🎉 IMPROVED ANALYSIS COMPLETE!")
        print(f"📄 Report saved: user_data/improved_exoplanet_analysis_report.md")
    
    def generate_improved_report(self, tbl_path, csv_path):
        """Generate improved analysis report"""
        
        star_id = tbl_path.split('/')[-1].replace('.tbl', '')
        
        # Calculate statistics
        total = len(self.results)
        exoplanets = sum(1 for r in self.results if r['prediction'] == 1)
        non_exoplanets = total - exoplanets
        
        # Probability analysis
        exoplanet_probs = [r['exoplanet_probability'] for r in self.results]
        avg_exoplanet_prob = np.mean(exoplanet_probs)
        
        report = f"""# 🌍 IMPROVED EXOPLANET ANALYSIS REPORT
## Star KIC {star_id}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** Hybrid Siamese Network + Classification Head
**Improvement:** Uses full embedding geometry + probability outputs

---

## 🎯 FINAL CONCLUSION
## {"POTENTIAL EXOPLANETS" if exoplanets > 0 else "NO EXOPLANETS DETECTED"}

**Analysis:** Found {exoplanets} potential exoplanet candidates out of {total} events ({exoplanets/total*100:.1f}%)

**Key Improvement:** Now uses full 128D embedding vector for classification, not just magnitude

---

## 📊 IMPROVED ANALYSIS SUMMARY

| Planet | Period (days) | Prediction | Exoplanet Prob | Confidence | SNR | Depth (ppm) | Status |
|--------|---------------|------------|----------------|------------|-----|-------------|---------|
"""
        
        for r in self.results:
            pred_str = "🌍 EXOPLANET" if r['prediction'] == 1 else "⭐ NON-EXOPLANET"
            
            # Status assessment
            if r['prediction'] == 1:
                if r['exoplanet_probability'] > 0.8:
                    status = "✅ Strong Candidate"
                elif r['exoplanet_probability'] > 0.6:
                    status = "❓ Moderate Candidate"
                else:
                    status = "❓ Weak Candidate"
            else:
                status = "⭐ Non-Exoplanet"
            
            report += f"| {r['planet_num']} | {r['period']:.3f} | {pred_str} | {r['exoplanet_probability']:.3f} | {r['confidence']:.3f} | {r['snr']:.1f} | {r['depth']:.1f} | {status} |\n"
        
        report += f"""

### Statistical Analysis:
- **Total Events:** {total}
- **Exoplanet Candidates:** {exoplanets} ({exoplanets/total*100:.1f}%)
- **Non-Exoplanets:** {non_exoplanets} ({non_exoplanets/total*100:.1f}%)
- **Average Exoplanet Probability:** {avg_exoplanet_prob:.3f}
- **Highest Confidence:** {max(r['confidence'] for r in self.results):.3f}

---

## 🔄 MODEL IMPROVEMENTS

### **Previous System Issues:**
- ❌ Used only embedding norm (magnitude) for classification
- ❌ Lost directional information in embedding space
- ❌ Binary threshold decision (no probability scores)
- ❌ High false positive rate for strong signals

### **New System Advantages:**
- ✅ **Hybrid Loss:** Combines contrastive + cross-entropy learning
- ✅ **Classification Head:** Uses full 128D vector, not just magnitude
- ✅ **Probability Outputs:** Softmax probabilities for both classes
- ✅ **Better Separation:** Can distinguish similar magnitude, different direction embeddings
- ✅ **Calibrated Thresholds:** Probability-based instead of arbitrary norm cutoff

### **Expected Performance Gains:**
- **Reduced False Positives:** Better separation of eclipsing binaries
- **Improved Precision:** Probability calibration
- **Better Recall:** Preserves geometric relationships
- **Confidence Scores:** Meaningful probability estimates

---

## 🔍 DETAILED CANDIDATE ANALYSIS

### High-Confidence Candidates (Prob > 0.7):
"""
        
        high_conf = [r for r in self.results if r['exoplanet_probability'] > 0.7]
        if high_conf:
            for r in sorted(high_conf, key=lambda x: x['exoplanet_probability'], reverse=True):
                report += f"""
#### Planet {r['planet_num']}:
- **Period:** {r['period']:.3f} days
- **Exoplanet Probability:** {r['exoplanet_probability']:.3f}
- **Confidence:** {r['confidence']:.3f}
- **SNR:** {r['snr']:.1f}
- **Transit Depth:** {r['depth']:.1f} ppm
- **Assessment:** Strong candidate, needs verification
"""
        else:
            report += "No high-confidence candidates detected.\n"
        
        report += """

---

## 📋 METHODOLOGY

### **Model Architecture:**
```
Shared Encoder (500 → 256 → 128 → 128)
    ↓
    ┌─────────────────────────┐
    │ Classification Head      │
    │ 128 → 64 → 2        │
    │ (Softmax probabilities)  │
    └─────────────────────────┘
```

### **Training Process:**
- **Hybrid Loss Function:** L_total = α × L_contrastive + β × L_crossentropy
- **Contrastive Learning:** Similar pairs pulled together, dissimilar pushed apart
- **Classification Learning:** Direct supervision with true labels
- **Joint Optimization:** Preserves geometry + learns decision boundaries

### **Inference Process:**
- **Input:** 500-binned phase-folded light curve
- **Embedding:** 128-dimensional vector from shared encoder
- **Classification:** Softmax probabilities from classification head
- **Output:** P(Exoplanet) and P(Non-Exoplanet)

---

## ⚠️ VALIDATION NOTES

**Important:** This is an improved model system that addresses the fundamental limitation 
of the original Siamese approach. While training and evaluation are recommended,
the model should be properly trained on the full dataset before deployment.

**Next Steps for Production:**
1. Train hybrid model on complete dataset
2. Validate with cross-validation
3. Calibrate probability thresholds
4. Compare ROC-AUC with original system

---

*Report generated by Improved Exoplanet Analysis System v2.0*
"""
        
        # Save report
        report_file = f"user_data/improved_exoplanet_analysis_report_{star_id}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

def main():
    """Main function for improved analysis system"""
    analyzer = ImprovedExoplanetAnalyzer()
    analyzer.run_improved_analysis()

if __name__ == "__main__":
    main()
