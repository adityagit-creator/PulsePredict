import os
import json
import whisper

# Load Whisper model
model = whisper.load_model("base")  # or "medium" / "large" depending on quality needs

# Input and output paths
input_dir = "audio_calls"
output_dir = "transcripts"
os.makedirs(output_dir, exist_ok=True)

# Loop through all audio files
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".mp3", ".wav", ".m4a")):
        input_path = os.path.join(input_dir, filename)
        print(f"🔊 Transcribing: {filename}")

        # Transcribe
        result = model.transcribe(input_path)
        transcript = result["text"]

        # Save transcript
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"✅ Saved: {output_path}")
