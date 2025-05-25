import os
import json
import pandas as pd
from tqdm import tqdm

LABELED_DIR = "labeled_entities"
OUTPUT_FILE = "features.csv"

def extract_features(labeled_data):
    features = []
    for entry in labeled_data:
        if not isinstance(entry, dict):
            continue
        all_entities = entry.get("entities", [])
        features.append({
            "filename": entry.get("file", ""),
            "num_medications": sum(1 for e in all_entities if e.get("category") == "MEDICATION"),
            "num_symptoms": sum(1 for e in all_entities if e.get("category") == "SYMPTOM"),
            "num_procedures": sum(1 for e in all_entities if e.get("category") == "TEST_TREATMENT_PROCEDURE"),
            "num_adverse_events": sum(1 for e in all_entities if e.get("is_adverse_event") is True),
            "num_entities": len(all_entities),
            "has_adverse_event": any(e.get("is_adverse_event") is True for e in all_entities)
        })
    return features

def load_labeled_data():
    labeled_data = []
    for file in tqdm(os.listdir(LABELED_DIR), desc="🔎 Extracting features from labeled entity files"):
        if file.endswith(".json"):
            path = os.path.join(LABELED_DIR, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "entities" in data:
                        labeled_data.append(data)
                    else:
                        print(f"⚠️ Skipped malformed or unexpected file: {file}")
            except json.JSONDecodeError:
                print(f"⚠️ JSON decode error in file: {file}")
    return labeled_data

def main():
    data = load_labeled_data()
    if not data:
        print("❌ No valid labeled JSON files found.")
        return
    features = extract_features(data)
    df = pd.DataFrame(features)

    df["adverse_event_ratio"] = df["num_adverse_events"] / (df["num_entities"] + 1e-5)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Feature extraction complete. Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
