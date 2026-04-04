import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- LOAD DATA ---
train = pd.read_csv(r"D:\ppp\data\features_500\train_1_curve_500.csv")
test = pd.read_csv(r"D:\ppp\data\features_500\test_1_curve_500.csv")

X_train, y_train = train.drop(columns=['Label']), train['Label']
X_test, y_test = test.drop(columns=['Label']), test['Label']

# --- TRAIN ---
print("🏎️ Training XGBoost Classifier...")
# use_label_encoder=False removes a common warning in newer versions
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    verbosity=1,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = model.predict(X_test)
print("\n[XGBOOST REPORT]")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='viridis')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("XGBoost Results")
plt.show()