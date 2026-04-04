import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- LOAD DATA ---
train = pd.read_csv(r"D:\ppp\data\features_500\train_1_curve_500.csv")
test = pd.read_csv(r"D:\ppp\data\features_500\test_1_curve_500.csv")

X_train, y_train = train.drop(columns=['Label']), train['Label']
X_test, y_test = test.drop(columns=['Label']), test['Label']

# --- TRAIN ---
print("🚀 Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = model.predict(X_test)
print("\n[LOGISTIC REGRESSION REPORT]")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.show()