"""
===========================================================
DATASET CLEANING PIPELINE (FINAL VERSION)
===========================================================

This script:
1. Merges multiple JSON files from output2/ (SQuAD v2.0 format)
2. Flattens nested SQuAD structure to flat records
3. Cleans OCR noise
4. Fixes answer_start
5. Removes bad questions
6. Removes exact duplicates
7. Removes paraphrase duplicates (semantic)
8. Outputs final clean dataset

===========================================================
"""

import os
import json
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import snapshot_download

# ---------------------------
# CONFIG
# ---------------------------
INPUT_FOLDER = "output2"   # folder with SQuAD v2.0 JSON files from LLM
OUTPUT_FILE = "cleaned_dataset.json"

SIM_THRESHOLD = 0.90   # semantic similarity threshold
MIN_QUESTION_LEN = 3   # minimum words in question

# Load embedding model for semantic filtering (from local hf_cache)
minilm_path = snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=r"/home/administrator/Desktop/Kartik/Kartik/SemiSagev2.0 (2)/SemiSagev2.0/hf_cache",
    local_files_only=True
)
model = SentenceTransformer(minilm_path)

# ---------------------------
# BASIC CLEANING
# ---------------------------
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\bSEMI\b.*?\d{4}', ' ', text)  # remove headers
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_bad_answer(answer):
    bad_keywords = [
        "purpose", "scope", "notice",
        "standard", "committee", "approved"
    ]

    a = answer.lower()

    return any(k in a for k in bad_keywords)

def is_generic_question(q):
    return q.lower() in [
        "what is wafer?",
        "what is system?",
        "what is process?"
    ]
# ---------------------------
# VALID QUESTION FILTER
# ---------------------------
def is_valid_question(q):
    if len(q.split()) < MIN_QUESTION_LEN:
        return False

    bad_patterns = ["what is n", "what is cases", "what is dot"]

    q_lower = q.lower()
    if any(p in q_lower for p in bad_patterns):
        return False

    return True

def is_valid_answer(context, answer):
    return answer.lower() in context.lower()
# ---------------------------
# FIX ANSWER START
# ---------------------------
def fix_answer_start(context, answer):
    context_l = context.lower()
    answer_l = answer.lower()

    start = context_l.find(answer_l)

    if start == -1:
        return None

    return start

# ---------------------------
# SEMANTIC DUPLICATE CHECK
# ---------------------------
def is_similar(q1, q2):
    emb1 = model.encode(q1, convert_to_tensor=True)
    emb2 = model.encode(q2, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2)

    return score > SIM_THRESHOLD

# ---------------------------
# STEP 1: MERGE ALL FILES (SQuAD v2.0 format)
# ---------------------------
print("🔄 Merging JSON files from output2/ (SQuAD v2.0 format)...")

all_data = []

for file in sorted(os.listdir(INPUT_FOLDER)):
    if file.endswith(".json") and file.startswith("final_dataset"):
        path = os.path.join(INPUT_FOLDER, file)
        with open(path, "r", encoding="utf-8") as f:
            try:
                squad_data = json.load(f)

                # Parse SQuAD v2.0 nested format → flat records
                for article in squad_data.get("data", []):
                    source = article.get("title", "unknown")
                    for para in article.get("paragraphs", []):
                        context = para.get("context", "")
                        for qa in para.get("qas", []):
                            # Skip unanswerable questions
                            if qa.get("is_impossible", False):
                                continue
                            if not qa.get("answers"):
                                continue

                            question = qa.get("question", "")
                            answer_obj = qa["answers"][0]
                            answer = answer_obj.get("text", "")
                            answer_start = answer_obj.get("answer_start", 0)
                            qtype = qa.get("type", "")

                            all_data.append({
                                "question": question,
                                "context": context,
                                "answer": answer,
                                "answer_start": answer_start,
                                "type": qtype,
                                "source": source
                            })
            except Exception as e:
                print(f"⚠️ Skipped bad file: {file} ({e})")

print(f"📊 Total merged samples: {len(all_data)}")

# ---------------------------
# STEP 2: CLEAN + FILTER
# ---------------------------
print("🧹 Cleaning dataset...")

cleaned = []
seen_exact = set()
context_questions = defaultdict(list)

dropped = 0

for item in all_data:

    q = clean_text(item.get("question", ""))
    c = clean_text(item.get("context", ""))
    a = clean_text(item.get("answer", ""))

    # Basic validation
    if not q or not c or not a:
        dropped += 1
        continue

    if is_bad_answer(a):
        dropped += 1
        continue

    if not is_valid_question(q):
        dropped += 1
        continue

    # ✅ Validate answer exists in context
    if not is_valid_answer(c, a):
        dropped += 1
        continue

    # Fix answer_start
    start = fix_answer_start(c, a)
    if start is None:
        dropped += 1
        continue

    # Exact duplicate check (question + context)
    key = (q, c)
    if key in seen_exact:
        dropped += 1
        continue
    seen_exact.add(key)

    # Semantic duplicate check (within same context)
    similar_found = False
    for existing_q in context_questions[c]:
        if is_similar(existing_q, q):
            similar_found = True
            break

    if similar_found:
        dropped += 1
        continue

    # Keep question
    context_questions[c].append(q)

    # Final cleaned item
    new_item = {
        "question": q,
        "context": c,
        "answer": a,
        "answer_start": start,
        "type": item.get("type", ""),
        "source": item.get("source", "")
    }

    cleaned.append(new_item)

# ---------------------------
# STEP 3: SAVE OUTPUT
# ---------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

# ---------------------------
# FINAL STATS
# ---------------------------
print("\n✅ Cleaning Complete!")
print(f"📊 Final dataset size: {len(cleaned)}")
print(f"❌ Dropped samples: {dropped}")
print(f"📉 Reduction: {round((dropped / len(all_data)) * 100, 2)}%")
