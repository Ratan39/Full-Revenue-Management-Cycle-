import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import json
import os
import re
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────
MODEL         = "alibayram/medgemma:4b"
NOTES_PATH    = "data/pmc_processed/pmc_cleaned.csv"
VECTOR_STORE  = "embeddings/icd10_vectorstore"
RESULTS_PATH  = "data/processed/coding_results.csv"
NUM_CANDIDATES = 10
TEST_NOTES     = 5       
NOTE_CHARS     = 1500   
# ──────────────────────────────────────────────────────────

print("Loading processed PMC notes...")
df = pd.read_csv(NOTES_PATH)
print(f"Total notes available: {len(df)}")

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to vector store...")
client     = chromadb.PersistentClient(path=VECTOR_STORE)
collection = client.get_collection("icd10_codes")
print(f"Vector store ready — {collection.count()} codes indexed")
print(f"Running inference with model: {MODEL}\n")


def retrieve_candidates(note_text, n=NUM_CANDIDATES):
    embedding = embedder.encode([note_text]).tolist()
    results   = collection.query(query_embeddings=embedding, n_results=n)
    return [
        {
            "code":        meta["code"],
            "description": meta["long_description"]
        }
        for meta in results["metadatas"][0]
    ]


def build_prompt(note_text, candidates):
    candidate_lines = "\n".join([
        f"  {c['code']}: {c['description']}"
        for c in candidates
    ])
    return f"""You are a certified professional medical coder (CPC).

A clinical note is provided below. Your task is to select the correct ICD-10-CM diagnosis codes from the candidate list.

Instructions:
- Read the clinical note carefully
- Select ONLY codes from the candidate list that genuinely apply to this patient
- Do NOT select codes that are not supported by the note
- Do NOT invent codes outside the candidate list
- Only select a code if the note contains explicit evidence supporting it
- Do NOT repeat the same code twice
- Do NOT select codes that are only loosely or indirectly related
- Return ONLY a valid JSON object — no text before or after

Clinical note:
{note_text[:NOTE_CHARS]}

Candidate ICD-10-CM codes (choose from these only):
{candidate_lines}

Return this exact JSON structure with your selections:
{{
  "selected_codes": [
    {{
      "code": "CODE_FROM_LIST",
      "description": "DESCRIPTION_FROM_LIST",
      "reason": "one sentence citing specific evidence from the note"
    }}
  ]
}}"""


def parse_response(raw):
    # Strategy 1: direct JSON parse
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

    return None



def run_inference(note_text, candidates):
    prompt   = build_prompt(note_text, candidates)
    response = ollama.chat(
        model=MODEL,
        options={
            "temperature": 0,
            "num_predict": 2048,
            "repeat_penalty": 1.3,
            "repeat_last_n": 64,
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical coding assistant. "
                    "You output only valid JSON. "
                    "Never add explanation outside the JSON object."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    raw    = response["message"]["content"].strip()
    parsed = parse_response(raw)
    return parsed, raw


# ── Main loop ─────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
results = []

for i, row in tqdm(
    df.head(TEST_NOTES).iterrows(),
    total=TEST_NOTES,
    desc="Processing notes"
):
    note     = row["full_note"]
    note_id  = row["idx"]

    candidates           = retrieve_candidates(note)
    parsed, raw_response = run_inference(note, candidates)

    selected = parsed.get("selected_codes", []) if parsed else []

    results.append({
        "note_id":              note_id,
        "note_preview":         note[:200],
        "candidates_retrieved": json.dumps(candidates),
        "selected_codes":       json.dumps(selected),
        "raw_response":         raw_response,
        "parse_success":        parsed is not None,
        "codes_selected_count": len(selected)
    })

    print(f"\n{'─'*60}")
    print(f"Note {i+1} | ID: {note_id}")
    print(f"Preview: {note[:100]}...")
    print(f"Candidates retrieved: {len(candidates)}")

    if selected:
        print(f"Codes selected ({len(selected)}):")
        for code in selected:
            print(f"  ✓ {code['code']} — {code['description']}")
            print(f"    → {code.get('reason', '')}")
    else:
        print("No codes selected or parse failed")
        print(f"Raw response FULL ({len(raw_response)} chars):")
        print(raw_response)

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH, index=False)

print(f"\n{'='*60}")
print(f"DONE")
print(f"Notes processed:    {len(results_df)}")
print(f"Successful parses:  {results_df['parse_success'].sum()} / {len(results_df)}")
print(f"Avg codes per note: {results_df['codes_selected_count'].mean():.1f}")
print(f"Results saved to:   {RESULTS_PATH}")
