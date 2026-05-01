"""
===========================================================
END-TO-END PIPELINE EVALUATION
===========================================================

Evaluates the full Retriever → Reader pipeline:
  Query → BGE+FAISS (retrieve) → BERT QA (extract answer)

Metrics:
  - Retrieval Recall@K (does correct passage appear?)
  - End-to-end Exact Match (is the final answer correct?)
  - End-to-end F1 Score (token overlap of final answer)
  - Avg QA Confidence

===========================================================
"""

import json
import re
import string
import collections
import numpy as np
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ---------------------------
# CONFIG
# ---------------------------
RETRIEVER_MODEL_PATH = "bge_finetuned2"
QA_MODEL_PATH = "qa_model_finetuned"
TEST_FILE = "qa_test.json"
TOP_K = [1, 3, 5]
MAX_LENGTH = 384

# ---------------------------
# LOAD MODELS
# ---------------------------
print("📥 Loading retriever (BGE)...")
retriever = SentenceTransformer(RETRIEVER_MODEL_PATH)

print("📥 Loading QA reader (BERT)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_PATH)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_PATH)
qa_model.to(device)
qa_model.eval()

# ---------------------------
# BUILD RETRIEVAL INDEX
# ---------------------------
# Load ALL triplet splits to build corpus (consistent with 05_build_faiss.py)
print("📥 Building retrieval corpus...")
all_data = []
for fname in ["train.json", "val.json", "test.json"]:
    with open(fname, "r") as f:
        all_data.extend(json.load(f))

corpus = list(set([d["positive"] for d in all_data]))
print(f"📦 Corpus size: {len(corpus)} unique passages")

print("🔄 Encoding corpus...")
corpus_embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=64)
corpus_embeddings = np.array(corpus_embeddings).astype("float32")
faiss.normalize_L2(corpus_embeddings)

dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(corpus_embeddings)

# ---------------------------
# LOAD TEST DATA
# ---------------------------
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"📊 Test records: {len(test_data)}")


# ---------------------------
# QA HELPER
# ---------------------------
def extract_answer(question, context):
    """Extract answer span from context using the QA model."""
    inputs = qa_tokenizer(
        question,
        context,
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offset_mapping = inputs.pop("offset_mapping")[0]
    sequence_ids = inputs.encodings[0].sequence_ids
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    n_best = 20
    start_indices = torch.topk(start_logits, n_best).indices.tolist()
    end_indices = torch.topk(end_logits, n_best).indices.tolist()

    start_probs = torch.softmax(start_logits, dim=0)
    end_probs = torch.softmax(end_logits, dim=0)

    best_answer = ""
    best_score = 0.0

    for s_idx in start_indices:
        for e_idx in end_indices:
            if e_idx < s_idx or e_idx - s_idx > 50:
                continue
            if s_idx >= len(offset_mapping) or e_idx >= len(offset_mapping):
                continue
            if sequence_ids[s_idx] != 1 or sequence_ids[e_idx] != 1:
                continue
            if offset_mapping[s_idx][0] == 0 and offset_mapping[s_idx][1] == 0:
                continue

            start_char = offset_mapping[s_idx][0].item()
            end_char = offset_mapping[e_idx][1].item()
            answer = context[start_char:end_char].strip()

            if not answer:
                continue

            score = start_probs[s_idx].item() * end_probs[e_idx].item()

            if score > best_score:
                best_score = score
                best_answer = answer

    return {"answer": best_answer, "score": best_score}


# ---------------------------
# EVALUATION METRICS
# ---------------------------
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


# ---------------------------
# EVALUATE
# ---------------------------
print("\n🔍 Evaluating end-to-end pipeline...\n")

max_k = max(TOP_K)

# Metrics accumulators
retrieval_hits = {k: 0 for k in TOP_K}
e2e_em = {k: 0 for k in TOP_K}
e2e_f1 = {k: 0.0 for k in TOP_K}
e2e_confidence = {k: 0.0 for k in TOP_K}
mrr_total = 0

total = len(test_data)

for item in tqdm(test_data):
    question = item["question"]
    ground_truth_answer = item["answers"]["text"][0]
    ground_truth_context = item["context"]

    # Step 1: Retrieve passages
    tagged_query = "query: " + question
    q_emb = retriever.encode([tagged_query])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)

    distances, indices = index.search(q_emb, max_k)
    retrieved_passages = [corpus[i] for i in indices[0]]
    retrieved_scores = [float(d) for d in distances[0]]

    # Retrieval metrics
    for k in TOP_K:
        if ground_truth_context in retrieved_passages[:k]:
            retrieval_hits[k] += 1

    # MRR
    for rank, idx in enumerate(indices[0]):
        if corpus[idx] == ground_truth_context:
            mrr_total += 1 / (rank + 1)
            break

    # Step 2: Run QA on retrieved passages for each K
    for k in TOP_K:
        passages_at_k = retrieved_passages[:k]
        scores_at_k = retrieved_scores[:k]

        # Extract answer from each passage, pick best by combined score
        best_answer = ""
        best_combined_score = -1

        for passage, ret_score in zip(passages_at_k, scores_at_k):
            try:
                result = extract_answer(question, passage)
                combined = result["score"] * ret_score
                if combined > best_combined_score:
                    best_combined_score = combined
                    best_answer = result["answer"]
            except Exception:
                continue

        if best_answer:
            em = compute_exact_match(best_answer, ground_truth_answer)
            f1 = compute_f1(best_answer, ground_truth_answer)
            e2e_em[k] += em
            e2e_f1[k] += f1
            e2e_confidence[k] += best_combined_score

# ---------------------------
# RESULTS
# ---------------------------
print("\n" + "=" * 60)
print("📊 END-TO-END EVALUATION RESULTS")
print("=" * 60)

print("\n--- Retrieval Metrics ---")
for k in TOP_K:
    recall = retrieval_hits[k] / total
    print(f"  Recall@{k}: {recall:.4f}")

mrr = mrr_total / total
print(f"  MRR:       {mrr:.4f}")

print("\n--- End-to-End QA Metrics ---")
for k in TOP_K:
    em = e2e_em[k] / total
    f1 = e2e_f1[k] / total
    conf = e2e_confidence[k] / total
    print(f"\n  Top-{k} Retrieval + QA:")
    print(f"    Exact Match: {em:.4f}")
    print(f"    F1 Score:    {f1:.4f}")
    print(f"    Avg Combined Confidence: {conf:.4f}")

print("\n" + "=" * 60)

# ---------------------------
# SAMPLE QUERIES
# ---------------------------
print("\n🧪 Sample End-to-End Predictions:\n")

sample_queries = [
    "What is wafer?",
    "What is diameter specification?",
    "What is ACK code?",
]

for q in sample_queries:
    tagged = "query: " + q
    q_emb = retriever.encode([tagged])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)

    distances, indices = index.search(q_emb, 3)

    print(f"  Query: {q}")

    best_answer = ""
    best_score = -1

    for idx, dist in zip(indices[0], distances[0]):
        passage = corpus[idx]
        try:
            result = extract_answer(q, passage)
            combined = result["score"] * float(dist)

            if combined > best_score:
                best_score = combined
                best_answer = result["answer"]

            print(f"    [Ret: {dist:.4f}] [QA: {result['score']:.4f}] → {result['answer']}")
        except Exception as e:
            print(f"    [Error] {e}")

    print(f"  ✅ Best Answer: {best_answer} (score: {best_score:.4f})")
    print()
