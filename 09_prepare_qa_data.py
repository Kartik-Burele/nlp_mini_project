"""
===========================================================
PREPARE QA TRAINING DATA
===========================================================

Converts cleaned_dataset.json to HuggingFace QA format
(SQuAD-style) for training an extractive QA model.

Input:  cleaned_dataset.json (flat records)
Output: qa_train.json, qa_val.json, qa_test.json

===========================================================
"""

import json
import random
import uuid
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
INPUT_FILE = "cleaned_dataset.json"
OUTPUT_TRAIN = "qa_train.json"
OUTPUT_VAL = "qa_val.json"
OUTPUT_TEST = "qa_test.json"

SEED = 42
random.seed(SEED)

# ---------------------------
# LOAD DATA
# ---------------------------
print("🔄 Loading cleaned dataset...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📊 Total records: {len(data)}")

# ---------------------------
# VALIDATE & CONVERT TO QA FORMAT
# ---------------------------
print("🔧 Converting to QA format...")

qa_records = []
skipped = 0

for item in tqdm(data, desc="Converting"):
    question = item["question"].strip()
    context = item["context"].strip()
    answer = item["answer"].strip()
    answer_start = item.get("answer_start", -1)

    # Validate: answer must exist in context
    # Re-find answer_start to ensure correctness
    actual_start = context.find(answer)
    if actual_start == -1:
        # Try case-insensitive
        actual_start = context.lower().find(answer.lower())
        if actual_start != -1:
            # Use the actual text from context
            answer = context[actual_start:actual_start + len(answer)]
        else:
            skipped += 1
            continue

    answer_start = actual_start

    # Verify the span is correct
    extracted = context[answer_start:answer_start + len(answer)]
    if extracted.lower() != answer.lower():
        skipped += 1
        continue

    qa_records.append({
        "id": str(uuid.uuid4()),
        "question": question,
        "context": context,
        "answers": {
            "text": [answer],
            "answer_start": [answer_start]
        },
        "type": item.get("type", ""),
        "source": item.get("source", "")
    })

print(f"✅ Valid QA records: {len(qa_records)}")
print(f"❌ Skipped (bad spans): {skipped}")

# ---------------------------
# SPLIT DATA
# ---------------------------
random.shuffle(qa_records)

n = len(qa_records)
train_split = int(0.8 * n)
val_split = int(0.9 * n)

train_data = qa_records[:train_split]
val_data = qa_records[train_split:val_split]
test_data = qa_records[val_split:]

print(f"\n📊 Split sizes:")
print(f"  Train: {len(train_data)}")
print(f"  Val:   {len(val_data)}")
print(f"  Test:  {len(test_data)}")

# ---------------------------
# SAVE
# ---------------------------
for split_name, split_data, output_file in [
    ("train", train_data, OUTPUT_TRAIN),
    ("val", val_data, OUTPUT_VAL),
    ("test", test_data, OUTPUT_TEST),
]:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {split_name} → {output_file}")

# ---------------------------
# SANITY CHECK
# ---------------------------
print("\n🧪 Sanity check (first 3 train records):")
for rec in train_data[:3]:
    q = rec["question"]
    ctx = rec["context"][:100] + "..."
    a = rec["answers"]["text"][0]
    s = rec["answers"]["answer_start"][0]
    extracted = rec["context"][s:s + len(a)]
    match = "✅" if extracted == a else "❌"
    print(f"  Q: {q}")
    print(f"  A: {a}")
    print(f"  Span check: {match} (extracted: '{extracted[:50]}')")
    print()
