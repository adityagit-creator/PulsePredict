import boto3
import json
import os
from tqdm import tqdm

# Create Comprehend Medical client
comprehend = boto3.client(service_name='comprehendmedical', region_name='us-east-1')

# Input and output folders
INPUT_DIR = "cleaned_transcripts"
OUTPUT_DIR = "entities"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_entities_from_text(text):
    result = comprehend.detect_entities_v2(Text=text)
    return result['Entities']

print("🧠 Extracting medical entities from transcripts...")

# Loop through each cleaned transcript
for filename in tqdm(os.listdir(INPUT_DIR)):
    if filename.endswith(".txt"):
        filepath = os.path.join(INPUT_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        try:
            entities = extract_entities_from_text(text)
            output_file = os.path.join(OUTPUT_DIR, filename.replace(".txt", ".json"))
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(entities, out_f, indent=2)
            print(f"✅ Saved entities: {output_file}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

print("✅ All transcripts processed.")
