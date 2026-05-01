import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_PATH = "bge_squad_finetuned"

model = SentenceTransformer(MODEL_PATH)

# ✅ Load ALL splits to build a complete corpus (not just train)
all_data = []
for fname in ["squad_triplets_train.json", "squad_triplets_val.json", "squad_triplets_test.json"]:
    with open(fname, "r") as f:
        all_data.extend(json.load(f))

contexts = [d["positive"] for d in all_data]
contexts = list(set(contexts))  # Unique contexts

print(f"📦 Total unique contexts in corpus: {len(contexts)}")

# Encode
embeddings = model.encode(contexts, show_progress_bar=True, batch_size=64)

embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)  # Normalize for cosine similarity

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

# Save index
faiss.write_index(index, "faiss_index_squad.bin")

# Save mapping
with open("contexts_squad.json", "w") as f:
    json.dump(contexts, f)

print("✅ SQuAD FAISS index built")