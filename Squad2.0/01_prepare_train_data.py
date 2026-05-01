import json
import random
from tqdm import tqdm

INPUT_FILE = "train-v2.0.json"   # SQuAD file
OUTPUT_TRAIN = "squad_triplets_train.json"
OUTPUT_VAL = "squad_triplets_val.json"
OUTPUT_TEST = "squad_triplets_test.json"

def normalize(text):
    return " ".join(text.strip().split())

def get_negative(context, answer, all_contexts, max_tries=10):
    """Sample a negative context that is different AND does not contain the answer."""
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

def create_triplets(data):
    triplets = []

    all_contexts = []

    # Collect all contexts from this split (for negatives)
    for article in data:
        for para in article["paragraphs"]:
            context = normalize(para["context"])
            all_contexts.append(context)

    for article in tqdm(data):
        for para in article["paragraphs"]:
            context = normalize(para["context"])

            for qa in para["qas"]:
                # ✅ Skip unanswerable questions (SQuAD 2.0)
                if qa.get("is_impossible", False) or not qa["answers"]:
                    continue

                question = normalize(qa["question"])
                answer = normalize(qa["answers"][0]["text"])
                answer_start = qa["answers"][0]["answer_start"]

                # ✅ Query format (BGE requirement)
                query = "query: " + question

                # ✅ Positive = windowed context around the answer (not answer + context)
                answer_end = answer_start + len(qa["answers"][0]["text"])
                left = max(0, answer_start - 100)
                right = min(len(para["context"]), answer_end + 100)
                positive = normalize(para["context"][left:right])

                # ✅ Negative sampling with answer-leakage filter
                negative = get_negative(context, answer, all_contexts)
                if negative is None:
                    continue

                triplets.append({
                    "query": query,
                    "positive": positive,
                    "negative": negative
                })

    return triplets


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        squad = json.load(f)

    data = squad["data"]

    # Shuffle the articles
    random.shuffle(data)

    n = len(data)
    train_split = int(0.8 * n)
    val_split = int(0.9 * n)

    train_data = data[:train_split]
    val_data = data[train_split:val_split]
    test_data = data[val_split:]

    print(f"Total articles: {n}")
    print(f"Train articles: {len(train_data)}")
    print(f"Val articles: {len(val_data)}")
    print(f"Test articles: {len(test_data)}")

    # Process each split
    train_triplets = create_triplets(train_data)
    val_triplets = create_triplets(val_data)
    test_triplets = create_triplets(test_data)

    print(f"Generated {len(train_triplets)} train triplets")
    print(f"Generated {len(val_triplets)} val triplets")
    print(f"Generated {len(test_triplets)} test triplets")

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        json.dump(train_triplets, f, indent=2)

    with open(OUTPUT_VAL, "w", encoding="utf-8") as f:
        json.dump(val_triplets, f, indent=2)

    with open(OUTPUT_TEST, "w", encoding="utf-8") as f:
        json.dump(test_triplets, f, indent=2)

    print("Saved train triplets to", OUTPUT_TRAIN)
    print("Saved val triplets to", OUTPUT_VAL)
    print("Saved test triplets to", OUTPUT_TEST)


if __name__ == "__main__":
    main()