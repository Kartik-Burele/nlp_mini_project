import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "bge_squad_finetuned"
TEST_FILE = "squad_triplets_test.json"
TOP_K = [1, 3, 5]

# ---------------------------
# LOAD MODEL
# ---------------------------
model = SentenceTransformer(MODEL_PATH)

def compute_f1(pred, truth):
    pred_tokens = pred.split()
    truth_tokens = truth.split()

    common = set(pred_tokens) & set(truth_tokens)

    if len(common) == 0:
        return 0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    return 2 * (precision * recall) / (precision + recall)
# ---------------------------
# LOAD TEST DATA
# ---------------------------
with open(TEST_FILE, "r") as f:
    test_data = json.load(f)

# ✅ Build corpus from ALL splits (consistent with 03_build_faiss.py)
all_data = []
for fname in ["squad_triplets_train.json", "squad_triplets_val.json", "squad_triplets_test.json"]:
    with open(fname, "r") as f:
        all_data.extend(json.load(f))

corpus = list(set([d["positive"] for d in all_data]))

print(f"📦 Corpus size: {len(corpus)} unique passages")

# ---------------------------
# ENCODE CORPUS
# ---------------------------
print("🔄 Encoding corpus...")
corpus_embeddings = model.encode(corpus, show_progress_bar=True, batch_size=64)
corpus_embeddings = np.array(corpus_embeddings).astype("float32")

# ---------------------------
# BUILD FAISS INDEX
# ---------------------------
faiss.normalize_L2(corpus_embeddings)  # Normalize for cosine similarity
dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(corpus_embeddings)

# ---------------------------
# EVALUATION
# ---------------------------
hits = {k: 0 for k in TOP_K}
total = len(test_data)

print("\n🔍 Evaluating...\n")

f1_total = 0
mrr_total = 0

for item in tqdm(test_data):
    query = item["query"]
    true_context = item["positive"]

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, max(TOP_K))

    retrieved = [corpus[i] for i in indices[0]]

    pred_context = retrieved[0]

    f1 = compute_f1(pred_context, true_context)
    f1_total += f1

    # ✅ Recall@K
    for k in TOP_K:
        if true_context in retrieved[:k]:
            hits[k] += 1

    # ✅ MRR computed in the same loop (no need for a second pass)
    for rank, idx in enumerate(indices[0]):
        if corpus[idx] == true_context:
            mrr_total += 1 / (rank + 1)
            break

# ---------------------------
# RESULTS
# ---------------------------
print("\n📊 Evaluation Results:\n")

for k in TOP_K:
    recall = hits[k] / total
    print(f"Recall@{k}: {recall:.4f}")

accuracy = hits[1] / total
print(f"Accuracy@1: {accuracy:.4f}")

f1_score = f1_total / total
print(f"Avg F1 Score: {f1_score:.4f}")

mrr = mrr_total / total
print(f"MRR: {mrr:.4f}")

# ---------------------------
# SAMPLE QUERIES
# ---------------------------
print("\n🧪 Sample Queries:\n")

sample_queries = [
    "Earlier television systems were based on what?",
    "When did the Hospital Vilardebo open?",
    "Along with the Prospect of Whitby and the Cheshire Cheese, what pub did Dickens visit?",
]

for q in sample_queries:
    q_tagged = "query: " + q
    q_emb = model.encode([q_tagged])
    q_emb = np.array(q_emb).astype("float32")

    # ✅ Fixed: normalize query embedding (was missing before)
    faiss.normalize_L2(q_emb)

    distances, indices = index.search(q_emb, 3)

    print(f"\nQuery: {q}")
    print("Top Results:")

    for i, dist in zip(indices[0], distances[0]):
        print(f"  - [Score: {dist:.4f}] {corpus[i][:200]}")
