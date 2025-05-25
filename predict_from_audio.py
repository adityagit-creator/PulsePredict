import os
import sys
import json
import joblib
import string
import whisper
import boto3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# === Paths ===
TRANSCRIPT_DIR = "transcripts"
ENTITY_DIR = "entities"
LABELED_DIR = "labeled_entities"
MODEL_PATH = "model/adverse_event_model.pkl"
FAERS_PATH = "data/faers_adverse_events.txt"

# === Load FAERS adverse events ===
def load_faers_events(path):
    with open(path, "r") as f:
        return set(line.strip().lower() for line in f if line.strip())

# === Clean text ===
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# === Transcribe audio (using Whisper) ===
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcript = result["text"]

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"📝 Transcript saved to {out_path}")
    return transcript, base_name

# === Preprocess transcript ===
def preprocess_text(text):
    return clean_text(text)

# === Extract entities using AWS Comprehend Medical ===
def extract_entities(text, base_name):
    client = boto3.client("comprehendmedical")
    response = client.detect_entities_v2(Text=text)
    entities = response["Entities"]

    os.makedirs(ENTITY_DIR, exist_ok=True)
    out_path = os.path.join(ENTITY_DIR, f"{base_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2)

    print(f"🔍 Entities saved to {out_path}")
    return entities

# === Label entities using FAERS ===
def label_adverse_events(entities, faers_set, base_name):
    for entity in entities:
        entity_text = entity.get("Text", "").lower()
        entity["is_adverse_event"] = any(
            faers_event in entity_text for faers_event in faers_set
        )

    os.makedirs(LABELED_DIR, exist_ok=True)
    out_path = os.path.join(LABELED_DIR, f"{base_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "file": base_name,
            "entities": entities
        }, f, indent=2)

    print(f"🏷️ Labeled entities saved to {out_path}")
    return entities

# === Extract features ===
def extract_features(entities):
    return [{
        "num_medications": sum(1 for e in entities if e.get("Category") == "MEDICATION"),
        "num_symptoms": sum(1 for e in entities if e.get("Category") == "SYMPTOM"),
        "num_procedures": sum(1 for e in entities if e.get("Category") == "TEST_TREATMENT_PROCEDURE"),
        "num_adverse_events": sum(1 for e in entities if e.get("is_adverse_event") is True),
        "num_entities": len(entities),
    }]

# === Predict adverse events ===
def predict_adverse_events(X, model_path):
    model = joblib.load(model_path)
    return model.predict(X)

# === Main Pipeline ===
def run_pipeline(audio_path):
    print("🚀 Starting full prediction pipeline...")
    faers_set = load_faers_events(FAERS_PATH)

    transcript, base_name = transcribe_audio(audio_path)
    cleaned_text = preprocess_text(transcript)
    entities = extract_entities(cleaned_text, base_name)
    labeled_entities = label_adverse_events(entities, faers_set, base_name)

    features = extract_features(labeled_entities)
    df = pd.DataFrame(features)

    df["adverse_event_ratio"] = df["num_adverse_events"] / (df["num_entities"] + 1e-5)

    predictions = predict_adverse_events(df, MODEL_PATH)

    print("\n📋 Prediction Result:")
    if predictions[0]:
        print("❗ Adverse Event Likely")
    else:
        print("✔️ No Adverse Event Detected")

# === Entry Point ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_from_audio.py path_to_audio.wav")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not os.path.isfile(audio_file):
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)

    run_pipeline(audio_file)
