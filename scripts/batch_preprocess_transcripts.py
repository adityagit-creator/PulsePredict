import os
import re

INPUT_DIR = "transcripts"
OUTPUT_DIR = "cleaned_transcripts"

FILLER_WORDS = {
    "um", "uh", "like", "you know", "i mean", "so", "well", "hmm",
    "ah", "er", "eh", "okay", "right", "yeah", "huh", "basically"
}

def clean_text(text):
    text = text.lower()
    for word in FILLER_WORDS:
        pattern = r'\b' + re.escape(word) + r'\b'
        text = re.sub(pattern, '', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_transcripts():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]

    for file in files:
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file)

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        cleaned = clean_text(text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"✅ Cleaned and saved: {output_path}")

if __name__ == "__main__":
    print("🧹 Preprocessing transcripts...")
    preprocess_transcripts()
    print("✅ All transcripts cleaned.")
