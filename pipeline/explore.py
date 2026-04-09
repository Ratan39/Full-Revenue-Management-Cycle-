import pandas as pd
import json

df = pd.read_csv("data/pmc_raw/pmc_notes.csv")

print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()


print(df.isnull().sum())
print()


df['note_length'] = df['full_note'].str.len()
print(df['note_length'].describe())
print()


print("-" * 60)
print(df['full_note'].iloc[0])
print()


print("-" * 60)
try:
    summary = json.loads(df['summary'].iloc[0])
    for key, value in summary.items():
        print(f"{key}: {value}")
except:
    print(df['summary'].iloc[0])
