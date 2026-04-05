import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import sys
import json
from datetime import datetime

class ExoplanetAnalyzer:
    """Complete exoplanet analysis system - one file, simple output"""
    
    def __init__(self):
        self.scaler = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
    def load_model(self):
        """Load trained model"""
        print("🔄 Loading model...")
        
        import joblib
        self.scaler = joblib.load("models/cleaned_scaler.pkl")
        
        class SiameseNetwork(nn.Module):
            def __init__(self, input_size, embedding_dim=128):
                super(SiameseNetwork, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, embedding_dim), nn.BatchNorm1d(embedding_dim)
                )
            def get_embedding(self, x):
                return self.encoder(x)
        
        self.model = SiameseNetwork(500, 128).to(self.device)
        self.model.load_state_dict(torch.load("models/cleaned_siamese.pth", map_location=self.device))
        self.model.eval()
        print("✅ Model loaded")
        
    def read_data(self, tbl_path, csv_path):
        """Read light curve and events data"""
        print(f"📖 Reading data from {tbl_path} and {csv_path}")
        
        # Read light curve
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
        
        # Read events
        events = pd.read_csv(csv_path, comment='#')
        events_list = []
        
        for _, row in events.iterrows():
            events_list.append({
                'planet_num': int(row['tce_plnt_num']),
                'period': float(row['tce_period']),
                'epoch': float(row['tce_time0bk']),
                'depth': float(row.get('tce_depth', 0)),
                'snr': float(row.get('tce_model_snr', 0)),
                'duration': float(row.get('tce_duration', 0)),
                'radius': float(row.get('tce_prad', 0))
            })
        
        print(f"✅ Loaded {len(time)} data points and {len(events_list)} events")
        return time, flux, quarter, events_list
    
    def analyze_event(self, time, flux, period, epoch, event_info):
        """Analyze single event"""
        
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
                    bin_centers[empty_bins], bin_centers[valid_indices], binned_flux[valid_indices]
                )
        
        # Normalize and process
        normalized_flux = binned_flux / np.median(binned_flux)
        processed_flux = self.scaler.transform(normalized_flux.reshape(1, -1))
        
        # Get prediction
        with torch.no_grad():
            embedding = self.model.get_embedding(torch.FloatTensor(processed_flux).to(self.device))
            embedding_norm = torch.norm(embedding).item()
        
        # Classify
        threshold = 0.2622
        if embedding_norm > threshold:
            prediction = "Exoplanet"
            confidence = min((embedding_norm - threshold) / (threshold * 0.5), 1.0)
        else:
            prediction = "Non-Exoplanet"
            confidence = min((threshold - embedding_norm) / threshold, 1.0)
        
        return {
            'planet_num': event_info['planet_num'],
            'period': period,
            'prediction': prediction,
            'confidence': confidence,
            'embedding_norm': embedding_norm,
            'depth': event_info['depth'],
            'snr': event_info['snr'],
            'duration': event_info['duration'],
            'radius': event_info['radius']
        }
    
    def make_final_judgment(self, results):
        """Make final scientific judgment"""
        
        exoplanet_candidates = [r for r in results if r['prediction'] == 'Exoplanet']
        
        if not exoplanet_candidates:
            return "NO EXOPLANETS", "No events show planetary characteristics"
        
        # Analyze candidates
        strong_candidates = []
        weak_candidates = []
        
        for candidate in exoplanet_candidates:
            # Check if it's likely a false positive
            if candidate['embedding_norm'] > 1.0:
                # Very high embedding norm - likely binary star
                weak_candidates.append(candidate)
            elif candidate['depth'] > 10000:  # >1% depth
                # Too deep for planet - likely binary
                weak_candidates.append(candidate)
            elif candidate['snr'] < 7:
                # Low SNR - could be noise
                weak_candidates.append(candidate)
            else:
                strong_candidates.append(candidate)
        
        if strong_candidates:
            return f"POTENTIAL EXOPLANETS ({len(strong_candidates)})", f"Found {len(strong_candidates)} strong candidates that need verification"
        elif weak_candidates:
            return "LIKELY FALSE POSITIVES", f"Found {len(weak_candidates)} unusual signals, but likely eclipsing binaries or stellar activity"
        else:
            return "UNCERTAIN", "Signals detected but classification is ambiguous"
    
    def generate_report(self, star_id, results):
        """Generate final markdown report"""
        
        final_judgment, judgment_reason = self.make_final_judgment(results)
        
        report = f"""# 🌍 EXOPLANET ANALYSIS REPORT
## Star KIC {star_id}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 🎯 FINAL CONCLUSION
## {final_judgment}

**Reasoning:** {judgment_reason}

---

## 📊 ANALYSIS SUMMARY

| Planet | Period (days) | Prediction | Confidence | SNR | Depth (ppm) | Status |
|--------|---------------|------------|------------|-----|-------------|---------|
"""
        
        for r in results:
            # Add status assessment
            if r['prediction'] == 'Exoplanet':
                if r['embedding_norm'] > 1.0 or r['depth'] > 10000:
                    status = "⚠️ Likely False Positive"
                elif r['snr'] < 7:
                    status = "❓ Low Confidence"
                else:
                    status = "✅ Good Candidate"
            else:
                status = "⭐ Non-Exoplanet"
            
            report += f"| {r['planet_num']} | {r['period']:.3f} | {r['prediction']} | {r['confidence']:.3f} | {r['snr']:.1f} | {r['depth']:.1f} | {status} |\n"
        
        report += f"""

---

## 🔍 DETAILED ANALYSIS

### Total Events: {len(results)}
- **Exoplanet Candidates:** {sum(1 for r in results if r['prediction'] == 'Exoplanet')}
- **Non-Exoplanets:** {sum(1 for r in results if r['prediction'] == 'Non-Exoplanet')}

### Key Metrics:
- **Average Embedding Norm:** {np.mean([r['embedding_norm'] for r in results]):.4f}
- **Highest Confidence:** {max(r['confidence'] for r in results):.3f}
- **Average SNR:** {np.mean([r['snr'] for r in results]):.1f}

---

## ⚠️ IMPORTANT NOTES

1. **Machine Learning Predictions Only:** These are computational predictions that require astronomical verification
2. **False Positives Common:** Eclipsing binaries and stellar activity can mimic planetary signals
3. **Professional Verification Needed:** Radial velocity, transit timing, and spectral analysis required for confirmation

### Next Steps for Real Discovery:
- Radial velocity measurements to detect stellar wobble
- Transit timing variations analysis
- Spectroscopic characterization
- Long-term monitoring for stability

---

## 📋 METHODOLOGY

**Model:** Siamese Neural Network  
**Training Accuracy:** 81.73%  
**Input:** 500-binned phase-folded light curves  
**Threshold:** 90th percentile embedding norm (0.2622)  
**Data:** {len(set([r['planet_num'] for r in results]))} events analyzed

*Report generated by Exoplanet Analysis System*
"""
        
        return report
    
    def run_complete_analysis(self, tbl_path=None, csv_path=None):
        """Run complete analysis and generate report"""
        
        print("🌍 EXOPLANET ANALYSIS SYSTEM")
        print("=" * 40)
        
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
        time, flux, quarter, events = self.read_data(tbl_path, csv_path)
        
        # Analyze each event
        print(f"\n🔍 Analyzing {len(events)} events...")
        for event in events:
            result = self.analyze_event(time, flux, event['period'], event['epoch'], event)
            self.results.append(result)
            print(f"   Planet {result['planet_num']}: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        # Generate report
        star_id = tbl_path.split('/')[-1].replace('.tbl', '')
        report = self.generate_report(star_id, self.results)
        
        # Save report
        report_file = f"user_data/exoplanet_analysis_report_{star_id}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Report saved: {report_file}")
        print(f"🎯 Final Conclusion: {self.make_final_judgment(self.results)[0]}")
        print(f"\n✅ Analysis complete!")

def main():
    analyzer = ExoplanetAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
