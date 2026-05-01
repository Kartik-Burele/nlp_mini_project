"""
===========================================================
RETRIEVER-READER PIPELINE
===========================================================

End-to-end pipeline combining:
  1. BGE Retriever (FAISS) → finds relevant passages
  2. BERT QA Reader → extracts precise answer spans

Usage:
  python 12_pipeline.py                    # Interactive mode
  python 12_pipeline.py "What is wafer?"   # Single query mode

Can also be imported:
  from 12_pipeline import answer_question

===========================================================
"""

import sys
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ---------------------------
# CONFIG
# ---------------------------
RETRIEVER_MODEL_PATH = "bge_finetuned2"
QA_MODEL_PATH = "qa_model_finetuned"
FAISS_INDEX_PATH = "faiss_index.bin"
CONTEXTS_PATH = "contexts.json"
DEFAULT_TOP_K = 3
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

print("📥 Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

with open(CONTEXTS_PATH, "r") as f:
    contexts = json.load(f)

print(f"✅ Pipeline ready! ({len(contexts)} passages indexed)\n")


# ---------------------------
# RETRIEVER
# ---------------------------
def retrieve(query, top_k=DEFAULT_TOP_K):
    """Retrieve top-K passages using BGE + FAISS."""
    tagged_query = "query: " + query
    query_embedding = retriever.encode([tagged_query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "context": contexts[idx],
            "retrieval_score": float(dist),
            "index": int(idx),
        })

    return results


# ---------------------------
# QA READER
# ---------------------------
def extract_answer(question, context):
    """Extract answer span from a single context using QA model.
    
    Handles the [CLS] empty-answer problem by searching top-N candidates
    for the best valid span that falls within the context tokens.
    """
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
    best_start = 0
    best_end = 0

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
                best_start = start_char
                best_end = end_char

    return {
        "answer": best_answer,
        "qa_score": best_score,
        "start": best_start,
        "end": best_end,
    }


# ---------------------------
# COMBINED PIPELINE
# ---------------------------
def answer_question(query, top_k=DEFAULT_TOP_K):
    """
    End-to-end: retrieve passages → extract answers → rank by confidence.

    Returns a list of answers sorted by QA confidence score.
    """
    # Step 1: Retrieve top-K passages
    passages = retrieve(query, top_k)

    # Step 2: Run QA model on each retrieved passage
    answers = []
    for passage in passages:
        qa_result = extract_answer(query, passage["context"])

        answers.append({
            "answer": qa_result["answer"],
            "qa_score": qa_result["qa_score"],
            "retrieval_score": passage["retrieval_score"],
            "combined_score": qa_result["qa_score"] * passage["retrieval_score"],
            "context": passage["context"],
            "answer_start": qa_result["start"],
            "answer_end": qa_result["end"],
        })

    # Step 3: Rank by combined score (QA confidence × retrieval score)
    answers.sort(key=lambda x: x["combined_score"], reverse=True)

    return answers


# ---------------------------
# DISPLAY
# ---------------------------
def display_results(query, answers):
    """Pretty-print the results."""
    print(f"\n{'='*60}")
    print(f"  Query: {query}")
    print(f"{'='*60}")

    for i, ans in enumerate(answers):
        print(f"\n  📌 Answer #{i+1}: {ans['answer']}")
        print(f"     QA Confidence:    {ans['qa_score']:.4f}")
        print(f"     Retrieval Score:  {ans['retrieval_score']:.4f}")
        print(f"     Combined Score:   {ans['combined_score']:.4f}")

        # Highlight answer in context
        ctx = ans["context"]
        start = ans["answer_start"]
        end = ans["answer_end"]

        # Show context snippet around the answer
        snippet_start = max(0, start - 80)
        snippet_end = min(len(ctx), end + 80)
        snippet = ctx[snippet_start:snippet_end]

        # Mark the answer in the snippet
        ans_in_snippet_start = start - snippet_start
        ans_in_snippet_end = end - snippet_start

        highlighted = (
            snippet[:ans_in_snippet_start]
            + ">>>" + snippet[ans_in_snippet_start:ans_in_snippet_end] + "<<<"
            + snippet[ans_in_snippet_end:]
        )

        print(f"     Context: ...{highlighted}...")

    print(f"\n{'='*60}\n")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    # Single query mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        answers = answer_question(query)
        display_results(query, answers)
        sys.exit(0)

    # Interactive mode
    print("🔍 SemiSage QA Pipeline - Interactive Mode")
    print("   Type your question and press Enter.")
    print("   Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("❓ Question: ").strip()

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("👋 Bye!")
                break

            answers = answer_question(query, top_k=3)
            display_results(query, answers)

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
