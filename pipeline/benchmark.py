import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import json
import re
import os
from tqdm import tqdm
from datetime import datetime

# ── Config ────────────────────────────────────────────────
MODEL            = "alibayram/medgemma:4b"
NOTES_PATH       = "data/mimic/notes/discharge.csv"
DIAGNOSES_PATH   = "data/mimic/structured/diagnoses_icd.csv"
VECTOR_STORE     = "embeddings/icd10_vectorstore"
RESULTS_PATH     = "data/mimic/benchmark_results.csv"
NUM_CANDIDATES   = 15
NOTE_CHARS       = 8000
BENCHMARK_NOTES  = 100    # start with 100, increase later
RANDOM_SEED      = 42
# ──────────────────────────────────────────────────────────

print("Loading MIMIC data...")
notes = pd.read_csv(NOTES_PATH)
dx    = pd.read_csv(DIAGNOSES_PATH)

# Keep only ICD-10
dx10  = dx[dx['icd_version'] == 10].copy()

# Strip dots from MIMIC codes for comparison
# MIMIC stores codes without dots — standardise both sides
dx10['icd_code_clean'] = dx10['icd_code'].str.replace('.', '', regex=False).str.strip()

# Group ground truth codes per admission
ground_truth = dx10.groupby('hadm_id')['icd_code_clean'].apply(set).to_dict()

# Find admissions with both a note and ICD-10 codes
notes_with_ids  = notes.dropna(subset=['hadm_id']).copy()
notes_with_ids['hadm_id'] = notes_with_ids['hadm_id'].astype(int)
valid_hadm_ids  = set(ground_truth.keys())
benchmark_notes = notes_with_ids[
    notes_with_ids['hadm_id'].isin(valid_hadm_ids)
].sample(BENCHMARK_NOTES, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"Benchmark notes selected: {len(benchmark_notes)}")

print("Loading embedding model...")
embedder   = SentenceTransformer("neuml/pubmedbert-base-embeddings")

print("Connecting to vector store...")
client     = chromadb.PersistentClient(path=VECTOR_STORE)
collection = client.get_collection("icd10_codes")
print(f"Vector store ready — {collection.count()} codes indexed")


def extract_clinical_entities(note_text):
    """Use MedGemma to extract specific clinical entities 
    from the note for targeted retrieval."""
    
    prompt = f"""Read this clinical note and extract a list of specific 
medical conditions, diagnoses, symptoms, and complications mentioned.

Return ONLY a JSON object like this:
{{
  "entities": [
    "acute systolic heart failure",
    "acute kidney injury",
    "type 2 diabetes mellitus",
    "hyponatremia",
    "anemia"
  ]
}}

Extract up to 15 most specific clinical entities.
Clinical note:
{note_text[:3000]}"""

    response = ollama.chat(
        model=MODEL,
        options={"temperature": 0, "num_predict": 512},
        messages=[
            {
                "role": "system",
                "content": "You extract medical entities. Output only JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    raw = response["message"]["content"].strip()
    parsed = parse_response(raw)
    
    if parsed and "entities" in parsed:
        return parsed["entities"]
    
    # Fallback — split note into chunks and return as entities
    return [note_text[i:i+200] for i in range(0, min(3000, len(note_text)), 200)]


def retrieve_candidates(note_text, n=5):
    """Entity-driven retrieval — search for each 
    clinical entity separately."""
    
    entities = extract_clinical_entities(note_text)
    all_codes = {}
    
    for entity in entities[:15]:
        if not entity or len(entity.strip()) < 5:
            continue
            
        embedding = embedder.encode([entity]).tolist()
        results   = collection.query(
            query_embeddings=embedding,
            n_results=n
        )
        
        for code, meta in zip(
            results["ids"][0],
            results["metadatas"][0]
        ):
            if code not in all_codes:
                all_codes[code] = {
                    "code":        meta["code"],
                    "description": meta["long_description"],
                    "matched_entity": entity
                }
    
    candidates = list(all_codes.values())
    return candidates


def build_prompt(note_text, candidates):
    candidate_lines = "\n".join([
        f"  {c['code']}: {c['description']}"
        for c in candidates
    ])
    return f"""You are a certified professional medical coder (CPC).

Read the clinical note and select the correct ICD-10-CM diagnosis codes from the candidate list.

Instructions:
- Only select codes directly supported by explicit evidence in the note
- Do NOT repeat the same code twice
- Do NOT select codes that are only loosely related
- Return ONLY valid JSON, no text before or after

Clinical note:
{note_text[:NOTE_CHARS]}

Candidate ICD-10-CM codes:
{candidate_lines}

Return this exact JSON:
{{
  "selected_codes": [
    {{
      "code": "CODE_FROM_LIST",
      "description": "DESCRIPTION",
      "reason": "one sentence citing evidence from the note"
    }}
  ]
}}"""


def run_inference(note_text, candidates):
    response = ollama.chat(
        model=MODEL,
        options={
            "temperature":   0,
            "num_predict":   2048,
            "repeat_penalty": 1.3,
            "repeat_last_n": 64,
        },
        messages=[
            {
                "role": "system",
                "content": "You are a medical coding assistant. Output only valid JSON."
            },
            {
                "role": "user",
                "content": build_prompt(note_text, candidates)
            }
        ]
    )
    raw    = response["message"]["content"].strip()
    parsed = parse_response(raw)
    return parsed, raw


def parse_response(raw):
    # Strategy 1: direct parse
    try:
        return json.loads(raw.strip())
    except:
        pass

    # Strategy 2: extract JSON block
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except:
        pass

    # Strategy 3: markdown code block
    try:
        if "```" in raw:
            if "```json" in raw:
                start = raw.find("```json") + 7
            else:
                start = raw.find("```") + 3
            end = raw.rfind("```")
            if end > start:
                return json.loads(raw[start:end].strip())
    except:
        pass

    # Strategy 4: salvage complete entries
    try:
        pattern = r'\{\s*"code"\s*:\s*"([^"]+)"\s*,\s*"description"\s*:\s*"([^"]+)"\s*,\s*"reason"\s*:\s*"([^"]+)"\s*\}'
        matches = re.findall(pattern, raw)
        if matches:
            seen   = set()
            unique = []
            for m in matches:
                if m[0] not in seen:
                    seen.add(m[0])
                    unique.append({
                        "code":        m[0],
                        "description": m[1],
                        "reason":      m[2]
                    })
            return {"selected_codes": unique}
    except:
        pass

    return None


def calculate_metrics(predicted_codes, ground_truth_codes):
    # Strip dots from predicted codes for comparison
    pred_clean = set(
        c.replace('.', '').strip().upper()
        for c in predicted_codes
    )
    gt_clean = set(
        c.replace('.', '').strip().upper()
        for c in ground_truth_codes
    )

    if not pred_clean and not gt_clean:
        return 1.0, 1.0, 1.0

    tp = len(pred_clean & gt_clean)
    fp = len(pred_clean - gt_clean)
    fn = len(gt_clean - pred_clean)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return precision, recall, f1


# ── Main benchmark loop ───────────────────────────────────
os.makedirs("data/mimic", exist_ok=True)
results = []

print(f"\nRunning benchmark on {BENCHMARK_NOTES} notes...")
print(f"Model: {MODEL}\n")

for i, row in tqdm(
    benchmark_notes.iterrows(),
    total=BENCHMARK_NOTES,
    desc="Benchmarking"
):
    hadm_id   = int(row['hadm_id'])
    note_text = str(row['text'])
    gt_codes  = ground_truth.get(hadm_id, set())

    candidates           = retrieve_candidates(note_text)
    parsed, raw_response = run_inference(note_text, candidates)

    selected     = parsed.get("selected_codes", []) if parsed else []
    pred_codes   = [c['code'] for c in selected]

    precision, recall, f1 = calculate_metrics(pred_codes, gt_codes)

    results.append({
        "hadm_id":            hadm_id,
        "note_preview":       note_text[:150],
        "ground_truth_codes": json.dumps(list(gt_codes)),
        "predicted_codes":    json.dumps(pred_codes),
        "candidates":         json.dumps([c['code'] for c in candidates]),
        "num_gt_codes":       len(gt_codes),
        "num_predicted":      len(pred_codes),
        "true_positives":     len(
            set(c.replace('.','').upper() for c in pred_codes) &
            set(c.replace('.','').upper() for c in gt_codes)
        ),
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "f1":                 round(f1, 4),
        "parse_success":      parsed is not None
    })

    if (i + 1) % 10 == 0:
        so_far = pd.DataFrame(results)
        print(f"\n--- After {i+1} notes ---")
        print(f"Avg precision: {so_far['precision'].mean():.3f}")
        print(f"Avg recall:    {so_far['recall'].mean():.3f}")
        print(f"Avg F1:        {so_far['f1'].mean():.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH, index=False)

print(f"\n{'='*60}")
print(f"BENCHMARK COMPLETE")
print(f"{'='*60}")
print(f"Notes benchmarked:  {len(results_df)}")
print(f"Parse success rate: {results_df['parse_success'].sum()}/{len(results_df)}")
print(f"Avg ground truth codes per note: {results_df['num_gt_codes'].mean():.1f}")
print(f"Avg predicted codes per note:    {results_df['num_predicted'].mean():.1f}")
print(f"Avg true positives per note:     {results_df['true_positives'].mean():.1f}")
print(f"\nPERFORMANCE METRICS (base model, no fine-tuning)")
print(f"{'─'*40}")
print(f"Precision: {results_df['precision'].mean():.3f}")
print(f"Recall:    {results_df['recall'].mean():.3f}")
print(f"F1 Score:  {results_df['f1'].mean():.3f}")
print(f"\nResults saved to: {RESULTS_PATH}")
print(f"\nThis is your Phase 3 baseline.")
print(f"After fine-tuning in Phase 5, re-run this script")
print(f"and compare the new F1 against this number.")