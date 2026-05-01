import faiss
import json
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_PATH = "bge_finetuned2"
QA_MODEL_PATH = "qa_model_finetuned"
MAX_LENGTH = 384

model = SentenceTransformer(MODEL_PATH)

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load contexts
with open("contexts.json", "r") as f:
    contexts = json.load(f)

# ✅ Load QA model if available (using direct model, not pipeline)
qa_tokenizer = None
qa_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(QA_MODEL_PATH):
    try:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_PATH)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_PATH)
        qa_model.to(device)
        qa_model.eval()
        print("✅ QA reader loaded")
    except Exception as e:
        print(f"⚠️ QA reader not available: {e}")


def extract_answer_from_context(question, context):
    """Extract answer span from context using the QA model."""
    if qa_tokenizer is None or qa_model is None:
        return None

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


def search(query, top_k=3, use_qa=True):
    """
    Search for relevant passages and optionally extract answers.

    Args:
        query: The user's question
        top_k: Number of passages to retrieve
        use_qa: If True and QA model is loaded, extract answer spans

    Returns:
        List of results with context, score, and optionally answer
    """
    # ✅ BGE requires "query: " prefix for best retrieval performance
    tagged_query = "query: " + query
    query_embedding = model.encode([tagged_query])
    query_embedding = np.array(query_embedding).astype("float32")
    # ✅ Normalize query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx, dist in zip(indices[0], distances[0]):
        result = {
            "context": contexts[idx],
            "retrieval_score": float(dist),
        }

        # ✅ Run QA reader if available
        if use_qa and qa_model is not None:
            try:
                qa_result = extract_answer_from_context(query, contexts[idx])
                if qa_result:
                    result["answer"] = qa_result["answer"]
                    result["qa_score"] = qa_result["score"]
                    result["combined_score"] = qa_result["score"] * float(dist)
            except Exception:
                result["answer"] = None
                result["qa_score"] = 0.0
                result["combined_score"] = 0.0

        results.append(result)

    # Sort by combined score if QA is used
    if use_qa and qa_model is not None:
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

    return results


# TEST
if __name__ == "__main__":
    q = input("Enter query: ")

    results = search(q)

    print("\nTop Results:\n")
    for r in results:
        if "answer" in r and r["answer"]:
            print(f"  📌 Answer: {r['answer']}")
            print(f"     QA Confidence:   {r['qa_score']:.4f}")
            print(f"     Retrieval Score: {r['retrieval_score']:.4f}")
            print(f"     Combined Score:  {r['combined_score']:.4f}")
            print(f"     Context: {r['context'][:200]}...")
        else:
            print(f"- [Score: {r['retrieval_score']:.4f}] {r['context'][:300]}")
        print()