import whisper

# Load the Whisper model 
model = whisper.load_model("base")

result = model.transcribe("patient_call.mp3")

print("\n--- TRANSCRIPTION ---\n")
print(result["text"])
