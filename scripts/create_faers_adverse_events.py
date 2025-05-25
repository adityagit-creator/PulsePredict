# create_faers_adverse_events.py

input_file = "REAC25Q1.txt"
output_file = "faers_adverse_events.txt"

unique_events = set()

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("$")
        if len(parts) >= 3:
            pt = parts[2].strip()
            if pt:
                unique_events.add(pt.lower())  # lowercase for normalization

# Save to output file
with open(output_file, "w", encoding="utf-8") as f:
    for event in sorted(unique_events):
        f.write(event + "\n")

print(f"✅ Extracted {len(unique_events)} unique adverse events to {output_file}")
