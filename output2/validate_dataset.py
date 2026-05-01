"""
Deep quality validation of the generated SQuAD dataset.
Checks: schema, answer_start accuracy, span quality, duplicates, OCR artifacts.
"""
import json
import re
from collections import Counter

FILE = r"k:\MTECH\NLP\Mini Project\SemiSage3.0\SemiSagev3.0\output2\final_dataset.json"

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ========================
# 1. SCHEMA VALIDATION
# ========================
print("=" * 60)
print("1. SCHEMA VALIDATION")
print("=" * 60)

schema_errors = []
if "version" not in data:
    schema_errors.append("Missing 'version' field")
if "data" not in data:
    schema_errors.append("Missing 'data' field")

for doc in data["data"]:
    if "title" not in doc:
        schema_errors.append(f"Missing 'title' in document")
    if "paragraphs" not in doc:
        schema_errors.append(f"Missing 'paragraphs' in document {doc.get('title')}")
    for p in doc.get("paragraphs", []):
        if "context" not in p:
            schema_errors.append("Missing 'context' in paragraph")
        if "qas" not in p:
            schema_errors.append("Missing 'qas' in paragraph")
        for qa in p.get("qas", []):
            if "id" not in qa:
                schema_errors.append(f"Missing 'id' in QA: {qa.get('question', '')[:50]}")
            if "question" not in qa:
                schema_errors.append("Missing 'question' in QA")
            if "answers" not in qa:
                schema_errors.append(f"Missing 'answers' in QA: {qa.get('question', '')[:50]}")
            elif not isinstance(qa["answers"], list):
                schema_errors.append(f"'answers' is not a list: {qa.get('question', '')[:50]}")
            else:
                for ans in qa["answers"]:
                    if "text" not in ans:
                        schema_errors.append("Missing 'text' in answer")
                    if "answer_start" not in ans:
                        schema_errors.append("Missing 'answer_start' in answer")

if schema_errors:
    print(f"  FAIL: {len(schema_errors)} schema errors")
    for e in schema_errors[:10]:
        print(f"    - {e}")
else:
    print("  PASS: SQuAD v2.0 schema is correct")

# ========================
# 2. ANSWER_START ACCURACY
# ========================
print("\n" + "=" * 60)
print("2. ANSWER_START VERIFICATION")
print("=" * 60)

total_answers = 0
correct_starts = 0
wrong_starts = []
not_found = []

for doc in data["data"]:
    for p in doc["paragraphs"]:
        context = p["context"]
        for qa in p["qas"]:
            for ans in qa["answers"]:
                total_answers += 1
                text = ans["text"]
                claimed_start = ans["answer_start"]

                # Check if the answer text exists at the claimed position
                actual_at_pos = context[claimed_start:claimed_start + len(text)]
                if actual_at_pos == text:
                    correct_starts += 1
                else:
                    # Check if it exists anywhere in context
                    real_start = context.find(text)
                    if real_start != -1:
                        wrong_starts.append({
                            "question": qa["question"][:60],
                            "answer": text[:60],
                            "claimed": claimed_start,
                            "actual": real_start,
                            "doc": doc["title"]
                        })
                    else:
                        not_found.append({
                            "question": qa["question"][:60],
                            "answer": text[:60],
                            "doc": doc["title"]
                        })

pct = round(correct_starts / max(total_answers, 1) * 100, 1)
print(f"  Total answers: {total_answers}")
print(f"  Correct answer_start: {correct_starts} ({pct}%)")
print(f"  Wrong answer_start (span exists but offset wrong): {len(wrong_starts)}")
print(f"  Answer NOT FOUND in context: {len(not_found)}")

if wrong_starts:
    print(f"\n  Sample wrong offsets (first 5):")
    for w in wrong_starts[:5]:
        print(f"    [{w['doc']}] Q: {w['question']}")
        print(f"      A: {w['answer']}...")
        print(f"      Claimed: {w['claimed']}, Actual: {w['actual']}")

if not_found:
    print(f"\n  CRITICAL — Answers not found in context (first 5):")
    for nf in not_found[:5]:
        print(f"    [{nf['doc']}] Q: {nf['question']}")
        print(f"      A: {nf['answer']}...")

# ========================
# 3. ANSWER QUALITY
# ========================
print("\n" + "=" * 60)
print("3. ANSWER QUALITY ANALYSIS")
print("=" * 60)

answer_lengths = []
short_answers = 0    # < 10 chars
long_answers = 0     # > 300 chars
ocr_garbage = 0

for doc in data["data"]:
    for p in doc["paragraphs"]:
        for qa in p["qas"]:
            for ans in qa["answers"]:
                text = ans["text"]
                answer_lengths.append(len(text))

                if len(text) < 10:
                    short_answers += 1
                if len(text) > 300:
                    long_answers += 1

                # Check for OCR artifacts
                if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
                    ocr_garbage += 1
                # Check for garbled text patterns
                if re.search(r'[^\x20-\x7E]{3,}', text):
                    ocr_garbage += 1

avg_len = round(sum(answer_lengths) / max(len(answer_lengths), 1), 1)
print(f"  Avg answer length: {avg_len} chars")
print(f"  Min answer length: {min(answer_lengths)} chars")
print(f"  Max answer length: {max(answer_lengths)} chars")
print(f"  Very short answers (<10 chars): {short_answers}")
print(f"  Very long answers (>300 chars): {long_answers}")
print(f"  Answers with OCR garbage: {ocr_garbage}")

# ========================
# 4. QUESTION QUALITY
# ========================
print("\n" + "=" * 60)
print("4. QUESTION QUALITY ANALYSIS")
print("=" * 60)

q_types = Counter()
questions_without_qmark = 0
yes_no_questions = 0
all_questions = []

for doc in data["data"]:
    for p in doc["paragraphs"]:
        for qa in p["qas"]:
            q = qa["question"]
            qtype = qa.get("type", "unknown")
            q_types[qtype] += 1
            all_questions.append(q)

            if not q.strip().endswith("?"):
                questions_without_qmark += 1
            if q.strip().lower().startswith(("is ", "are ", "does ", "do ", "can ", "will ")):
                yes_no_questions += 1

print(f"  Question type distribution:")
for qt, count in q_types.most_common():
    print(f"    {qt}: {count} ({round(count/len(all_questions)*100,1)}%)")
print(f"  Questions without '?': {questions_without_qmark}")
print(f"  Potential yes/no questions: {yes_no_questions}")

# ========================
# 5. DUPLICATE DETECTION
# ========================
print("\n" + "=" * 60)
print("5. DUPLICATE DETECTION")
print("=" * 60)

q_counter = Counter(q.lower().strip() for q in all_questions)
duplicates = {q: c for q, c in q_counter.items() if c > 1}

print(f"  Total unique questions: {len(q_counter)}")
print(f"  Duplicate questions (across chunks): {len(duplicates)}")
if duplicates:
    print(f"  Top duplicates:")
    for q, c in sorted(duplicates.items(), key=lambda x: -x[1])[:10]:
        print(f"    [{c}x] {q[:80]}")

# ========================
# 6. PER-DOCUMENT BREAKDOWN
# ========================
print("\n" + "=" * 60)
print("6. PER-DOCUMENT BREAKDOWN")
print("=" * 60)

for doc in data["data"]:
    n_para = len(doc["paragraphs"])
    n_qa = sum(len(p["qas"]) for p in doc["paragraphs"])
    avg = round(n_qa / max(n_para, 1), 1)
    print(f"  {doc['title']}: {n_para} contexts, {n_qa} QAs (avg {avg}/ctx)")

# ========================
# 7. SAMPLE QA PAIRS
# ========================
print("\n" + "=" * 60)
print("7. SAMPLE QA PAIRS (first 3 from each document)")
print("=" * 60)

for doc in data["data"]:
    print(f"\n  --- {doc['title']} ---")
    count = 0
    for p in doc["paragraphs"]:
        for qa in p["qas"]:
            if count >= 3:
                break
            ans = qa["answers"][0]
            print(f"  Q: {qa['question']}")
            print(f"  A: {ans['text'][:120]}{'...' if len(ans['text'])>120 else ''}")
            print(f"  Type: {qa['type']} | Start: {ans['answer_start']}")
            print()
            count += 1
        if count >= 3:
            break

# ========================
# SUMMARY
# ========================
print("\n" + "=" * 60)
print("OVERALL QUALITY SCORE")
print("=" * 60)

score = 100
issues = []

if schema_errors:
    score -= 30
    issues.append(f"Schema errors: {len(schema_errors)}")

wrong_pct = len(wrong_starts) / max(total_answers, 1) * 100
if wrong_pct > 5:
    score -= 20
    issues.append(f"answer_start accuracy: {pct}%")
elif wrong_pct > 1:
    score -= 10
    issues.append(f"answer_start accuracy: {pct}%")

if not_found:
    score -= 25
    issues.append(f"Answers missing from context: {len(not_found)}")

if ocr_garbage > 0:
    score -= 15
    issues.append(f"OCR garbage in answers: {ocr_garbage}")

if len(duplicates) > total_answers * 0.1:
    score -= 10
    issues.append(f"High duplicate rate: {len(duplicates)}")

if yes_no_questions > total_answers * 0.1:
    score -= 5
    issues.append(f"Yes/no questions: {yes_no_questions}")

score = max(score, 0)

print(f"  Score: {score}/100")
if issues:
    print(f"  Issues found:")
    for i in issues:
        print(f"    - {i}")
else:
    print("  No significant issues found!")
