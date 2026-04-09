from datasets import load_dataset
import pandas as pd
import os

print("Starting PMC-Patients download...")

dataset = load_dataset("AGBonnet/augmented-clinical-notes", split="train")

print(f"Downloaded {len(dataset)} records")
print(f"Columns available: {dataset.column_names}")

df = dataset.to_pandas()

output_path = "data/pmc_raw/pmc_notes.csv"
df.to_csv(output_path, index=False)

print(f"Saved to {output_path}")
print("\nFirst note preview:")
print("-" * 60)
print(df.iloc[0])
