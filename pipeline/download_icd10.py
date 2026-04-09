import pandas as pd
import os

print("Parsing ICD-10-CM order file...")

order_file = "data/icd10_raw/icd10cm-order-2025.txt"

records = []

with open(order_file, 'r', encoding='utf-8') as f:
    for line in f:
        # Fixed width format:
        # cols 0-4   = order number (5 chars)
        # cols 6-12  = code (7 chars)
        # col 14     = valid billing code flag (1 = valid, 0 = header only)
        # cols 16-75 = short description (60 chars)
        # cols 77+   = long description
        if len(line) < 77:
            continue

        order_num = line[0:5].strip()
        code = line[6:13].strip()
        valid = line[14].strip()
        short_desc = line[16:76].strip()
        long_desc = line[77:].strip()

        # Only keep valid billing codes (flag = 1)
        if valid == '1' and code and long_desc:
            records.append({
                "code": code,
                "short_description": short_desc,
                "long_description": long_desc,
                # Combined text for embedding — what the RAG layer searches
                "search_text": f"{code}: {long_desc}"
            })

df = pd.DataFrame(records)

out_path = "data/icd10_raw/icd10_codes_parsed.csv"
df.to_csv(out_path, index=False)

print(f"Total valid ICD-10-CM codes: {len(df)}")
print(f"Saved to {out_path}")
print()
print("=== SAMPLE CODES ===")
print(df.head(10).to_string(index=False))
print()
print("=== SAMPLE SEARCH IN CODE LIST ===")
# Quick test — find diabetes codes
diabetes = df[df['long_description'].str.contains('diabetes', case=False)]
print(f"Codes containing 'diabetes': {len(diabetes)}")
print(diabetes.head(5)[['code', 'long_description']].to_string(index=False))
