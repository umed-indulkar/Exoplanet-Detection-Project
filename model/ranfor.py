import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- LOAD DATA ---
train = pd.read_csv(r"D:\ppp\data\features\train_balanced.csv")
test = pd.read_csv(r"D:\ppp\data\features\test_balanced.csv")

X_train, y_train = train.drop(columns=['Label']), train['Label']
X_test, y_test = test.drop(columns=['Label']), test['Label']

# --- TRAIN ---
print("🌲 Training Random Forest (100 Trees)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = model.predict(X_test)
print("\n[RANDOM FOREST REPORT]")
print(classification_report(y_test, y_pred))

# Feature Importance (Top 10)
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importances.head(10).plot(kind='barh', title="Top 10 RF Features")
plt.show()