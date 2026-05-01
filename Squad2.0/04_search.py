import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = "bge_squad_finetuned"

model = SentenceTransformer(MODEL_PATH)

# Load FAISS index
index = faiss.read_index("faiss_index_squad.bin")

# Load contexts
with open("contexts_squad.json", "r") as f:
    contexts = json.load(f)

def search(query, top_k=3):
    query = "query: " + query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx, dist in zip(indices[0], distances[0]):
        results.append({"context": contexts[idx], "score": float(dist)})

    return results


# TEST
if __name__ == "__main__":
    q = input("Enter query: ")

    results = search(q)

    print("\nTop Results:\n")
    for r in results:
        print(f"- [Score: {r['score']:.4f}] {r['context'][:300]}\n")