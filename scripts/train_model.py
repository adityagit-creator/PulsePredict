import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def main():
    print("📊 Loading features...")
    df = pd.read_csv("features.csv")

    # Target column
    y = df["has_adverse_event"]
    
    # Feature columns (exclude filename and label)
    X = df.drop(columns=["filename", "has_adverse_event"])

    print("🔀 Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🎯 Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("🧪 Evaluating model...")
    y_pred = clf.predict(X_test)

    print("\n🧾 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n🔍 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(clf, "adverse_event_model.pkl")
    print("\n💾 Model saved to: adverse_event_model.pkl")

if __name__ == "__main__":
    main()
