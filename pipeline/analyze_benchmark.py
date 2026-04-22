import pandas as pd
import json
from collections import Counter

df = pd.read_csv("data/mimic/benchmark_results.csv")

print("=" * 60)
print("BENCHMARK ANALYSIS")
print("=" * 60)

print(f"\nTotal notes: {len(df)}")
print(f"Parse success: {df['parse_success'].sum()}/{len(df)}")

print(f"\nCode count comparison:")
print(f"Avg ground truth codes: {df['num_gt_codes'].mean():.1f}")
print(f"Avg predicted codes:    {df['num_predicted'].mean():.1f}")
print(f"Avg true positives:     {df['true_positives'].mean():.2f}")

# Notes where model got at least one code right
at_least_one = df[df['true_positives'] > 0]
print(f"\nNotes with at least 1 correct code: {len(at_least_one)}/{len(df)}")

# Best performing notes
print(f"\nTop 5 notes by F1:")
top = df.nlargest(5, 'f1')[['hadm_id','num_gt_codes',
    'num_predicted','true_positives','precision','recall','f1']]
print(top.to_string(index=False))

# Worst performing
print(f"\nNotes with 0 true positives: {len(df[df['true_positives']==0])}")

# What codes did the model get right
all_correct = []
for _, row in df.iterrows():
    pred = set(c.replace('.','').upper()
               for c in json.loads(row['predicted_codes']))
    gt   = set(c.replace('.','').upper()
               for c in json.loads(row['ground_truth_codes']))
    correct = pred & gt
    all_correct.extend(list(correct))

print(f"\nMost commonly correctly predicted codes:")
counter = Counter(all_correct)
for code, count in counter.most_common(10):
    print(f"  {code}: {count} times")

# Candidate hit rate — was the correct code even in the candidates?
hit_in_candidates = 0
total_gt_codes    = 0
for _, row in df.iterrows():
    candidates = set(c.replace('.','').upper()
                     for c in json.loads(row['candidates']))
    gt         = set(c.replace('.','').upper()
                     for c in json.loads(row['ground_truth_codes']))
    total_gt_codes    += len(gt)
    hit_in_candidates += len(candidates & gt)

hit_rate = hit_in_candidates / total_gt_codes if total_gt_codes > 0 else 0
print(f"\nRAG candidate hit rate:")
print(f"Ground truth codes found in top-10 candidates: "
      f"{hit_in_candidates}/{total_gt_codes} ({hit_rate:.1%})")
print(f"\nThis tells you how often the correct code was even")
print(f"available for the model to select.")