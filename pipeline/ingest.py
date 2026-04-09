import pandas as pd
import json
import re
import os
from tqdm import tqdm

print("Loading raw PMC notes...")
df = pd.read_csv("data/pmc_raw/pmc_notes.csv")
print(f"Loaded {len(df)} records")

def clean_note(text):
    if not isinstance(text, str):
        return ""
    # Remove excessive whitespace and newline artifacts
    text = text.replace("\\n", "\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()
    return text

def parse_summary(summary_str):
    try:
        return json.loads(summary_str)
    except:
        return {}

def extract_key_fields(summary_dict):
    # Safely get the patient info block
    patient_info = summary_dict.get("patient information", {})
    
    # If it's a list, take the first element if available, else use an empty dict
    if isinstance(patient_info, list):
        patient_info = patient_info[0] if len(patient_info) > 0 else {}
    
    # If it's None or something else unexpected, default to empty dict
    if not isinstance(patient_info, dict):
        patient_info = {}

    return {
        "visit_motivation": summary_dict.get("visit motivation", ""),
        "symptoms": summary_dict.get("symptoms", []),
        "treatments": summary_dict.get("treatments", []),
        "diagnoses": summary_dict.get("diagnosis tests", []),
        "patient_age": patient_info.get("age", ""),
        "patient_sex": patient_info.get("sex", "")
    }

print("Cleaning and processing notes...")
processed = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    cleaned_note = clean_note(row['full_note'])
    summary_dict = parse_summary(row['summary'])
    key_fields = extract_key_fields(summary_dict)

    # Skip notes that are too short to be useful
    if len(cleaned_note) < 200:
        continue

    processed.append({
        "idx": row['idx'],
        "full_note": cleaned_note,
        "note_length": len(cleaned_note),
        "visit_motivation": key_fields['visit_motivation'],
        "patient_age": key_fields['patient_age'],
        "patient_sex": key_fields['patient_sex'],
        "symptoms_raw": json.dumps(key_fields['symptoms']),
        "treatments_raw": json.dumps(key_fields['treatments']),
    })

processed_df = pd.DataFrame(processed)

output_path = "data/pmc_processed/pmc_cleaned.csv"
processed_df.to_csv(output_path, index=False)

print(f"\nProcessing complete")
print(f"Original records:  {len(df)}")
print(f"Processed records: {len(processed_df)}")
print(f"Dropped (too short): {len(df) - len(processed_df)}")
print(f"Saved to {output_path}")
print()
print("=== SAMPLE PROCESSED RECORD ===")
sample = processed_df.iloc[0]
print(f"IDX:             {sample['idx']}")
print(f"Note length:     {sample['note_length']} chars")
print(f"Patient age:     {sample['patient_age']}")
print(f"Patient sex:     {sample['patient_sex']}")
print(f"Visit motivation:{sample['visit_motivation'][:120]}...")
