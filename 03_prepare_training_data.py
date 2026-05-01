import json
import random
from tqdm import tqdm

INPUT_FILE = "cleaned_dataset.json"
OUTPUT_TRAIN = "train.json"
OUTPUT_VAL = "val.json"
OUTPUT_TEST = "test.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Shuffle the data
random.shuffle(data)

n = len(data)
train_split = int(0.8 * n)
val_split = int(0.9 * n)

train_data_raw = data[:train_split]
val_data_raw = data[train_split:val_split]
test_data_raw = data[val_split:]

def normalize(text):
    """Normalize whitespace without lowercasing — preserving case is important for BGE embeddings."""
    return " ".join(text.strip().split())

def get_negative(item, all_contexts, max_tries=10):
    """Sample a negative context that is different AND does not contain the answer."""
    context = item["context"]
    answer = item["answer"]

    for _ in range(max_tries):
        neg = random.choice(all_contexts)
        if neg != context and answer.lower() not in neg.lower():
            return neg

    # Fallback: at least return a different context
    for _ in range(max_tries):
        neg = random.choice(all_contexts)
        if neg != context:
            return neg

    return None

def process_split(split_data, split_name):
    triplets = []

    # Collect all contexts from this split (for negatives)
    all_contexts = [item["context"] for item in split_data]

    for item in tqdm(split_data, desc=f"Processing {split_name}"):

        start = item["answer_start"]
        end = start + len(item["answer"])
        # ✅ Positive = windowed context around the answer (±100 chars)
        left = max(0, start - 100)
        right = min(len(item["context"]), end + 100)
        positive = normalize(item["context"][left:right])

        # ✅ Query format (BGE requirement)
        query = "query: " + normalize(item["question"])

        # ✅ Negative sampling with answer-leakage filter
        negative = get_negative(item, all_contexts)

        if negative is None:
            continue

        negative = normalize(negative)

        triplets.append({
            "query": query,
            "positive": positive,
            "negative": negative
        })

    return triplets

# Process each split
train_set = process_split(train_data_raw, "train")
val_set = process_split(val_data_raw, "val")
test_set = process_split(test_data_raw, "test")

print(f"\nTotal articles: {n}")
print(f"Train articles: {len(train_data_raw)}")
print(f"Val articles: {len(val_data_raw)}")
print(f"Test articles: {len(test_data_raw)}")

# Save the splits
with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
    json.dump(train_set, f, indent=2)

with open(OUTPUT_VAL, "w", encoding="utf-8") as f:
    json.dump(val_set, f, indent=2)

with open(OUTPUT_TEST, "w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=2)

print("\n✅ Split done:")
print(f"Generated {len(train_set)} train triplets")
print(f"Generated {len(val_set)} val triplets")
print(f"Generated {len(test_set)} test triplets")