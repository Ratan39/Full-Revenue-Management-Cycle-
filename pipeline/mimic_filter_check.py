import pandas as pd

print("Loading diagnosis codes...")
dx = pd.read_csv("data/mimic/structured/diagnoses_icd.csv")

print(f"Total diagnosis rows: {len(dx)}")
print(f"\nICD version breakdown:")
print(dx['icd_version'].value_counts())

icd10 = dx[dx['icd_version'] == 10]
icd10_admissions = icd10['hadm_id'].nunique()
print(f"\nUnique admissions with ICD-10 codes: {icd10_admissions}")

print("\nLoading discharge notes...")
notes = pd.read_csv("data/mimic/notes/discharge.csv")

notes_ids = set(notes['hadm_id'].dropna().astype(int))
icd10_ids = set(icd10['hadm_id'].dropna().astype(int))
linked    = notes_ids & icd10_ids

print(f"Admissions with ICD-10 codes AND discharge note: {len(linked)}")
print(f"\nSample ICD-10 codes:")
print(icd10.head(10)[['hadm_id', 'icd_code', 'icd_version']].to_string(index=False))