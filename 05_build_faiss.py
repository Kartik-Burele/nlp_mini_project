import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_PATH = "bge_finetuned2"

model = SentenceTransformer(MODEL_PATH)

# ✅ Load ALL splits to build a complete corpus (not just train)
all_data = []
for fname in ["train.json", "val.json", "test.json"]:
    with open(fname, "r") as f:
        all_data.extend(json.load(f))

contexts = [d["positive"] for d in all_data]
contexts = list(set(contexts))  # Unique contexts

print(f"📦 Total unique contexts in corpus: {len(contexts)}")

# Encode
embeddings = model.encode(contexts, show_progress_bar=True, batch_size=64)

embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)  # Normalize for cosine similarity

# ✅ Build FAISS index with Inner Product (cosine similarity with normalized vectors)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

# Save index
faiss.write_index(index, "faiss_index.bin")

# Save mapping
with open("contexts.json", "w") as f:
    json.dump(contexts, f)

print("✅ FAISS index built")