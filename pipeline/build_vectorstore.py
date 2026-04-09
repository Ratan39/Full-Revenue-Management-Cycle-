import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

print("Loading ICD-10 codes...")
df = pd.read_csv("data/icd10_raw/icd10_codes_parsed.csv")
print(f"Total codes to embed: {len(df)}")

print("\nLoading embedding model (downloads once, cached after)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

print("\nSetting up ChromaDB...")
client = chromadb.PersistentClient(path="embeddings/icd10_vectorstore")
collection_name = "icd10_codes"

# Delete if exists so we start fresh
existing = [c.name for c in client.list_collections()]
if collection_name in existing:
    client.delete_collection(collection_name)

collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

print("Embedding and storing codes in batches...")
BATCH_SIZE = 500

for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch = df.iloc[i:i+BATCH_SIZE]

    texts = batch["search_text"].tolist()
    ids = batch["code"].tolist()
    metadatas = [
        {
            "code": row["code"],
            "short_description": row["short_description"],
            "long_description": row["long_description"]
        }
        for _, row in batch.iterrows()
    ]

    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

print(f"\nVector store built successfully.")
print(f"Total codes stored: {collection.count()}")

print("\n=== TEST SEARCH ===")
print("Query: 'patient with sustained muscle contractions abnormal posture neck'")
print()

query = "patient with sustained muscle contractions abnormal posture neck"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

for i, (doc, meta) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0]
)):
    print(f"  {i+1}. {meta['code']} — {meta['long_description']}")
