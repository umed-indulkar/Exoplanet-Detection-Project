import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import os

class BaselineModels:
    """Baseline models for exoplanet detection using dataset_500"""
    
    def __init__(self, data_path="data/extracted_features/features_curve_500_pruned.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.models = {}
        
    def load_data(self):
        """Load and preprocess the dataset"""
        print("📊 Loading dataset_500...")
        
        # Load data
        self.data = pd.read_csv(self.data_path, low_memory=False)
        print(f"   Raw shape: {self.data.shape}")
        
        # Extract features and labels
        if 'Label' in self.data.columns:
            self.labels = self.data['Label'].values
            feature_columns = [col for col in self.data.columns if col not in ['kepid', 'Label']]
            self.features = self.data[feature_columns].values
        else:
            self.labels = self.data.iloc[:, 0].values
            self.features = self.data.iloc[:, 1:].values
        
        # Convert to proper dtypes
        self.labels = pd.Series(self.labels).fillna(0).astype(int).values
        self.features = pd.DataFrame(self.features).fillna(0).astype(np.float64).values
        
        print(f"   Features: {self.features.shape[1]}")
        print(f"   Samples: {len(self.labels)}")
        print(f"   Positives: {np.sum(self.labels == 1)}")
        print(f"   Negatives: {np.sum(self.labels == 0)}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=0.2, stratify=self.labels, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   Train samples: {len(self.X_train)}")
        print(f"   Test samples: {len(self.X_test)}")
        
    def train_random_forest(self):
        """Train Random Forest classifier"""
        print("\n🌲 Training Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(self.X_train_scaled, self.y_train)
        self.models['random_forest'] = rf
        
        # Evaluate
        y_pred = rf.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(classification_report(self.y_test, y_pred))
        
        # Save model
        joblib.dump(rf, 'models/random_forest.pkl')
        print(f"   Model saved: models/random_forest.pkl")
        
        return accuracy
    
    def train_logistic_regression(self):
        """Train Logistic Regression classifier"""
        print("\n📈 Training Logistic Regression...")
        
        lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['logistic_regression'] = lr
        
        # Evaluate
        y_pred = lr.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(classification_report(self.y_test, y_pred))
        
        # Save model
        joblib.dump(lr, 'models/logistic_regression.pkl')
        print(f"   Model saved: models/logistic_regression.pkl")
        
        return accuracy
    
    def train_xgboost(self):
        """Train XGBoost classifier"""
        print("\n🚀 Training XGBoost...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(self.X_train_scaled, self.y_train)
        self.models['xgboost'] = xgb_model
        
        # Evaluate
        y_pred = xgb_model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(classification_report(self.y_test, y_pred))
        
        # Save model
        joblib.dump(xgb_model, 'models/xgboost.pkl')
        print(f"   Model saved: models/xgboost.pkl")
        
        return accuracy
    
    def train_all_models(self):
        """Train all baseline models"""
        print("🚀 TRAINING ALL BASELINE MODELS")
        print("=" * 50)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Train models
        results = {}
        results['random_forest'] = self.train_random_forest()
        results['logistic_regression'] = self.train_logistic_regression()
        results['xgboost'] = self.train_xgboost()
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print(f"\n💾 Scaler saved: models/scaler.pkl")
        
        # Summary
        print("\n📊 MODEL COMPARISON")
        print("=" * 50)
        for model, accuracy in results.items():
            print(f"{model:20s}: {accuracy:.4f}")
        
        # Save results
        with open('output/baseline_results.txt', 'w') as f:
            f.write("BASELINE MODEL RESULTS\n")
            f.write("=" * 50 + "\n")
            for model, accuracy in results.items():
                f.write(f"{model:20s}: {accuracy:.4f}\n")
        
        print(f"\n📄 Results saved: output/baseline_results.txt")
        
        return results

if __name__ == "__main__":
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Train baseline models
    baseline = BaselineModels()
    results = baseline.train_all_models()
