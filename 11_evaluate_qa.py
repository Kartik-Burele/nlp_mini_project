"""
===========================================================
EVALUATE QA MODEL (STANDALONE)
===========================================================

Evaluates the fine-tuned QA model in isolation:
given the correct context, how well does it extract the answer?

Metrics: Exact Match (EM), F1 Score, Answer Confidence

Input:  qa_test.json + qa_model_finetuned/
Output: Evaluation metrics + sample predictions

===========================================================
"""

import json
import re
import string
import collections
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "qa_model_finetuned"
TEST_FILE = "qa_test.json"
MAX_LENGTH = 384
DOC_STRIDE = 128

# ---------------------------
# LOAD MODEL
# ---------------------------
print("📥 Loading QA model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

print("✅ Model loaded")


def extract_answer(question, context):
    """Extract answer span from context using the QA model.
    
    Handles the [CLS] empty-answer problem by searching top-N candidates
    for the best valid span that falls within the context tokens.
    """
    inputs = tokenizer(
        question,
        context,
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offset_mapping = inputs.pop("offset_mapping")[0]
    sequence_ids = inputs.encodings[0].sequence_ids  # 0=question, 1=context, None=special

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Get top-20 start and end candidates
    n_best = 20
    start_indices = torch.topk(start_logits, n_best).indices.tolist()
    end_indices = torch.topk(end_logits, n_best).indices.tolist()

    start_probs = torch.softmax(start_logits, dim=0)
    end_probs = torch.softmax(end_logits, dim=0)

    best_answer = ""
    best_score = 0.0

    for s_idx in start_indices:
        for e_idx in end_indices:
            # Skip invalid spans
            if e_idx < s_idx:
                continue
            if e_idx - s_idx > 50:  # Max answer length in tokens
                continue

            # Skip special tokens and question tokens — only allow context tokens
            if s_idx >= len(offset_mapping) or e_idx >= len(offset_mapping):
                continue
            if sequence_ids[s_idx] != 1 or sequence_ids[e_idx] != 1:
                continue

            # Skip [CLS]-like positions where offset is (0, 0)
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
# LOAD TEST DATA
# ---------------------------
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"📊 Test records: {len(test_data)}")


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
    """Check if normalized prediction matches normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_f1(prediction, ground_truth):
    """Compute token-level F1 between prediction and ground truth."""
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

    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------
# EVALUATE
# ---------------------------
print("\n🔍 Evaluating...\n")

em_total = 0
f1_total = 0
confidence_total = 0

predictions = []

for item in tqdm(test_data):
    question = item["question"]
    context = item["context"]
    ground_truth = item["answers"]["text"][0]

    # Run QA model
    result = extract_answer(question, context)

    pred_answer = result["answer"]
    pred_score = result["score"]

    # Compute metrics
    em = compute_exact_match(pred_answer, ground_truth)
    f1 = compute_f1(pred_answer, ground_truth)

    em_total += em
    f1_total += f1
    confidence_total += pred_score

    predictions.append({
        "question": question,
        "ground_truth": ground_truth,
        "prediction": pred_answer,
        "score": pred_score,
        "em": em,
        "f1": f1,
    })

# ---------------------------
# RESULTS
# ---------------------------
total = len(test_data)

print("\n📊 QA Evaluation Results (Standalone - Correct Context):\n")
print(f"  Exact Match (EM): {em_total / total:.4f}")
print(f"  F1 Score:         {f1_total / total:.4f}")
print(f"  Avg Confidence:   {confidence_total / total:.4f}")
print(f"  Total samples:    {total}")

# ---------------------------
# SAMPLE PREDICTIONS
# ---------------------------
print("\n🧪 Sample Predictions:\n")

# Show some good and bad predictions
good_preds = [p for p in predictions if p["em"]][:3]
bad_preds = [p for p in predictions if not p["em"] and p["f1"] < 0.5][:3]

print("✅ Good predictions (EM=1):")
for p in good_preds:
    print(f"  Q: {p['question']}")
    print(f"  A: {p['ground_truth']}")
    print(f"  P: {p['prediction']} (conf: {p['score']:.4f})")
    print()

print("❌ Difficult cases (EM=0, F1<0.5):")
for p in bad_preds:
    print(f"  Q: {p['question']}")
    print(f"  A: {p['ground_truth']}")
    print(f"  P: {p['prediction']} (conf: {p['score']:.4f}, F1: {p['f1']:.4f})")
    print()

# ---------------------------
# SAVE PREDICTIONS
# ---------------------------
with open("qa_predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)

print("💾 Full predictions saved to qa_predictions.json")
