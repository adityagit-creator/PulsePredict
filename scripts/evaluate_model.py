import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import joblib

# === Load dataset ===
df = pd.read_csv("data/features.csv")

# === Add derived feature ===
df["adverse_event_ratio"] = df["num_adverse_events"] / (df["num_entities"] + 1e-5)

# === Rule-based baseline evaluation ===
df["rule_based_prediction"] = df["adverse_event_ratio"] > 0.3

print("📏 Rule-Based Baseline (adverse_event_ratio > 0.3):")
print(classification_report(df["has_adverse_event"], df["rule_based_prediction"]))
print("🧩 Confusion Matrix (Rule-Based):")
print(confusion_matrix(df["has_adverse_event"], df["rule_based_prediction"]))
print("-" * 50)

# === Prepare features and labels ===
X = df.drop(columns=["has_adverse_event", "filename", "rule_based_prediction"], errors="ignore")
y = df["has_adverse_event"]

# === Check class distribution ===
value_counts = y.value_counts()
if value_counts.min() < 2:
    raise ValueError("❌ Not enough samples in both classes to evaluate. Add more data.")

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Train model ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)

print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Model Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Model Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save model ===
joblib.dump(model, "model/adverse_event_model.pkl")
print("💾 Model saved to model/adverse_event_model.pkl")
