import os
import json
from tqdm import tqdm
from rapidfuzz import process

LABELED_DIR = "labeled_entities"
ENTITIES_DIR = "entities"
ADVERSE_EVENTS_FILE = "faers_adverse_events.txt"

def load_adverse_events(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f)

def label_entity(entity, adverse_event_set):
    entity_text_raw = entity.get("Text") or entity.get("text", "")
    entity_text = entity_text_raw.strip().lower()

    print(f"🔍 Checking entity: '{entity_text}'")

    # Exact match
    if entity_text in adverse_event_set:
        print(f"✅ Exact match: '{entity_text}'")
        return True

    # Fuzzy match
    match, score, _ = process.extractOne(entity_text, adverse_event_set)
    print(f"🟡 Fuzzy match: '{entity_text}' ≈ '{match}' (score={score:.2f})")
    return score >= 90

def main():
    os.makedirs(LABELED_DIR, exist_ok=True)
    adverse_event_set = load_adverse_events(ADVERSE_EVENTS_FILE)

    print("🏷️  Labeling entities with adverse events...")

    for filename in tqdm(os.listdir(ENTITIES_DIR)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(ENTITIES_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            try:
                entities = json.load(f)
                if not isinstance(entities, list):
                    print(f"⚠️ Skipping malformed file: {filename}")
                    continue
            except json.JSONDecodeError:
                print(f"⚠️ Skipping corrupted file: {filename}")
                continue

        # Label each entity
        for entity in entities:
            entity["is_adverse_event"] = label_entity(entity, adverse_event_set)

        # Save labeled format
        labeled_data = {
            "file": filename,
            "entities": entities
        }

        out_path = os.path.join(LABELED_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(labeled_data, out_f, indent=2)

    print(f"✅ Labeled entities saved to: {LABELED_DIR}")

if __name__ == "__main__":
    main()
