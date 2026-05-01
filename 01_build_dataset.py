import os 
import random
import re 
import json
import time
import requests 
import hashlib 
import spacy 
import fitz

def extract_pdf_text(path):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    return text

# --------------------------- 
# CONFIGURATION 
# --------------------------- 
OCR_TEXT_DIR = "data/ocr_text" 
if not os.path.isdir(OCR_TEXT_DIR) and os.path.isdir("data/ocr_texts"):
    OCR_TEXT_DIR = "data/ocr_texts"
OUTPUT_FILE = "output/dataset.json" 

MODEL_PRIMARY = "phi3:mini" # fast, stable 
MODEL_SECONDARY = "mistral:7b" # optional, slower 

MAX_WORDS_PER_CHUNK = 250
PARAPHRASES_PER_QUESTION = 2

USE_SECONDARY_MODEL = False # set True if you want mistral passes 

paraphrase_cache = {}

# --------------------------- 
# LOAD NLP MODEL (OFFLINE) 
# --------------------------- 
nlp = spacy.load("en_core_web_sm") 

# --------------------------- 
# OLLAMA CALL (LOCAL) 
# --------------------------- 
def ollama_call(prompt, model=MODEL_PRIMARY):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()

    return data.get("response", "").strip()

# ---------------------------
# TABLE PARSING
# ---------------------------
def is_table_block(text):
    lines = text.split("\n")

    return (
        len(lines) > 5 and
        sum(1 for l in lines if len(re.split(r'\s{2,}', l)) >= 2) > 3
    )

def parse_table_generic(text):
    rows = []

    for line in text.split("\n"):
        line = line.strip()

        if len(line) < 5:
            continue

        cols = re.split(r'\s{2,}', line)

        if len(cols) >= 2:
            rows.append(cols)

    if not rows:
        return [], []

    header = rows[0]
    data = rows[1:]

    return header, data

def infer_table_semantics(header):
    header_text = " ".join(header).lower()

    if any(k in header_text for k in ["parameter", "requirement"]):
        return "entity-attribute"

    if any(k in header_text for k in ["name", "code", "function"]):
        return "entity-description"

    if any(k in header_text for k in ["pin", "signal"]):
        return "mapping"

    return "generic"

def generate_qna_generic(header, data):
    qnas = []

    for row in data:

        for i in range(len(header)):
            if i >= len(row):
                continue

            column_name = header[i].strip()
            value = row[i].strip()

            if len(value) < 2:
                continue

            # Identify entity (first column)
            entity = row[0].strip()

            # Skip entity column itself
            if i == 0:
                continue

            # 🔥 GENERIC QUESTION
            question = f"What is the {column_name} of {entity} in this document?"

            qnas.append({
                "question": question,
                "answer": value,
                "type": "fact"
            })

    return qnas

def beautify_question(col, entity):
    col = col.lower()

    if "code" in col:
        return f"What is the code for {entity} in this document?"

    if "function" in col:
        return f"What is the function of {entity} in this document?"

    if "signal" in col:
        return f"What signal corresponds to {entity} in this document?"

    return f"What is the {col} of {entity} in this document?"

# ---------------------------
# NEW CLEANING FUNCTIONS
# ---------------------------
def clean_questions(qna_list):
    filtered = []

    for item in qna_list:
        q = item["question"].lower()

        if len(q.split()) < 3:
            continue

        if any(x in q for x in ["©", "semr", "figure", "note", ".3", "mm?"]):
            continue

        if re.search(r'[^a-zA-Z0-9\s?]', q):
            continue

        filtered.append(item)

    return filtered

def clean_answers(qna_list):
    filtered = []

    for item in qna_list:
        a = item["answer"]

        if len(a.split()) < 4:
            continue

        if re.search(r'[^\x00-\x7F]+', a):
            continue

        filtered.append(item)

    return filtered

# --------------------------- 
# TEXT INGESTION 
# --------------------------- 
def load_texts(folder):
    texts = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)

        if f.lower().endswith(".txt"):
            with open(path, encoding="utf-8") as fh:
                texts.append((fh.read(), f))

        elif f.lower().endswith(".pdf"):
            texts.append((extract_pdf_text(path), f))

    return texts

# --------------------------- 
# CLEAN OCR NOISE 
# --------------------------- 
def clean_text(text): 
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text) 
    text = re.sub(r'[�■│]{2,}', '', text) 
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'-\n', '', text) 
    text = re.sub(r'\s+', ' ', text) 
    return text.strip() 

# --------------------------- 
# CHUNKING 
# --------------------------- 
def chunk_text(text, max_words): 
    words = text.split() 
    chunks, cur = [], [] 
    for w in words: 
        cur.append(w) 
        if len(cur) >= max_words: 
            chunks.append(" ".join(cur)) 
            cur = [] 
    if cur: 
        chunks.append(" ".join(cur)) 
    return chunks 

# ---------------------------
# SENTENCE SPLITTING (FIX)
# ---------------------------
def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

# --------------------------- 
# KNOWLEDGE EXTRACTION 
# --------------------------- 
def extract_definitions(sentences):
    return [
        s for s in sentences
        if re.search(r'\b(is|means|refers to|defined as)\b', s.lower())
    ]

def extract_facts(sentences):
    return [
        s for s in sentences
        if (
            re.search(r'\d', s) or
            any(k in s.lower() for k in ["mm", "µm", "diameter", "thickness","code", "signal", "value", "pin"])
        )
    ]

def extract_context(sentences):
    return [
        s for s in sentences
        if any(k in s.lower() for k in ["purpose", "shall", "required"])
    ]

# --------------------------- 
# BASIC NLP & QUALITY FILTERS 
# --------------------------- 
def grammar_filter(text): 
    doc = nlp(text) 
    has_verb = any(t.pos_ == "VERB" for t in doc) 
    has_subject = any(t.dep_ in ("nsubj", "nsubjpass") for t in doc) 
    return has_verb and has_subject 

def basic_quality_filter(answer, chunk): 
    if answer not in chunk: 
        return False 
    if len(answer) < 10: 
        return False 
    if not grammar_filter(answer): 
        return False 
    return True 

# --------------------------- 
# QNA GENERATION 
# --------------------------- 
def extract_term(sentence):
    sentence = sentence.lower()

    KEY_TERMS = [
        "diameter", "thickness", "edge", "fiducial",
        "wafer", "bonding", "glass", "surface",
        "roughness", "cracks", "chips", "modulus"
    ]

    for term in KEY_TERMS:
        if term in sentence:
            return term

    # fallback: longest noun
    doc = nlp(sentence)
    nouns = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN")]

    if nouns:
        return max(nouns, key=len)

    return "parameter"
 
def make_qna(sentence, qtype):
    term = extract_term(sentence)
    s = sentence.lower()

    if qtype == "definition":
        templates = [
            f"What is {term}?",
            f"Define {term}.",
            f"Explain {term}."
        ]

    elif qtype == "fact":
        templates = [
            f"What is the specification of {term}?",
            f"What is defined for {term}?",
            f"What are the requirements for {term}?",
            f"What value or condition is specified for {term}?",
            f"What is the {term} specified in this document?"
        ]

    else:
        templates = [
            f"Why is {term} required?",
            f"What is the purpose of {term}?"
        ]

    random.shuffle(templates)

    return {
        "question": random.choice(templates),
        "answer": sentence.strip(),
        "type": qtype
    }

def generate_numeric_questions(sentence):
    s = sentence.lower()

    if not any(unit in s for unit in ["mm", "µm", "nm"]):
        return None

    if "diameter" in s:
        q = "What is the diameter value specified?"

    elif "thickness" in s:
        q = "What is the thickness value specified?"

    elif "edge" in s:
        q = "What is the edge dimension specified?"

    else:
        return None

    return {
        "question": q,
        "answer": sentence,
        "type": "fact"
    }

def format_duration(seconds):
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

# --------------------------- 
# PARAPHRASE EXPANSION 
# --------------------------- 
def paraphrase_question(question):

    # ✅ CACHE HIT (fast)
    if question in paraphrase_cache:
        return paraphrase_cache[question]

    prompt = f"""
Rewrite the question in {PARAPHRASES_PER_QUESTION} different ways.

STRICT RULES:
- Do NOT change meaning
- Do NOT introduce new topics
- Stay within semiconductor / wafer domain
- Keep it short and technical

Question:
{question}

Return only questions.
"""

    try:
        out = ollama_call(prompt)

        lines = [
            l.strip("- ").strip()
            for l in out.split("\n")
            if "?" in l
        ]

        result = lines[:PARAPHRASES_PER_QUESTION]

        # ✅ SAVE TO CACHE
        paraphrase_cache[question] = result

        return result
    except Exception as e:
        print(f"Paraphrase error: {e}")
        return []
    
# --------------------------- 
# DATASET BUILD 
# --------------------------- 
def build_dataset(): 
    texts = load_texts(OCR_TEXT_DIR) 
    if not texts:
        print(f"No files found in {OCR_TEXT_DIR}")
        return
    all_qnas = [] 
    seen = set() 
    items = []

    for raw, current_source in texts: 
        clean = clean_text(raw) 
        chunks = chunk_text(clean, MAX_WORDS_PER_CHUNK)
        items.extend((chunk, current_source) for chunk in chunks)
    
    total_chunks = len(items)
    start_time = time.time()
    print(f"Starting dataset build: {len(texts)} files, {total_chunks} chunks")

    for index, (chunk, current_source) in enumerate(items, start=1):
        if is_table_block(chunk):

            header, data = parse_table_generic(chunk)

            if not header:
                continue

            qnas = generate_qna_generic(header, data)


            for qna in qnas:
                answer = qna["answer"]
                start = chunk.find(answer)

                if start == -1:
                    continue

                if len(qna["question"].split()) < 3:
                    continue

                item = {
                    "question": qna["question"],
                    "answer": qna["answer"],
                    "context": chunk,
                    "answer_start": start,
                    "type": qna["type"],
                    "source": current_source
                }
                h = hashlib.md5(json.dumps(qna, sort_keys=True).encode()).hexdigest()

                if h not in seen:
                    seen.add(h)
                    all_qnas.append(item)
            elapsed = time.time() - start_time
            if index % 20 == 0 or index == total_chunks:
                remaining = (elapsed / index) * (total_chunks - index) if index else 0
            print(f"Processed {index}/{total_chunks} chunks · elapsed {format_duration(elapsed)} · remaining {format_duration(remaining)}", end="\r")

            continue
        sentences = split_sentences(chunk)

        defs = extract_definitions(sentences)
        facts = extract_facts(sentences)
        ctxs = extract_context(sentences)

        for s in defs:
            qna = make_qna(s, "definition")

            if basic_quality_filter(qna["answer"], chunk):

                # Step 1: Clean answer early
                temp = [{"question": qna["question"], "context": chunk, "answer": qna["answer"], "type": qna["type"], "source": current_source}]
                temp = clean_answers(temp)

                if not temp:
                    continue

                answer = qna["answer"]
                start = chunk.find(answer)

                if start == -1:
                    continue

                # Step 2: Paraphrase
                variants = paraphrase_question(qna["question"])
                variants.append(qna["question"])

                # Step 3: Filter paraphrases
                variants = variants[:3]

                # Step 4: Store
                for q in variants:
                    item = {
                        "question": q,
                        "context": chunk,
                        "answer": qna["answer"],
                        "answer_start": start,
                        "type": qna["type"],
                        "source": current_source
                    }

                    h = hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()

                    if h not in seen:
                        seen.add(h)
                        all_qnas.append(item)

        for s in facts:

            qna = make_qna(s, "fact")

            numeric_qna = generate_numeric_questions(s)

            qna_list = [qna]

            if numeric_qna:
                qna_list.append(numeric_qna)

            for qna_item in qna_list:

                if basic_quality_filter(qna_item["answer"], chunk):


                    temp = [{
                        "question": qna_item["question"],
                        "context": chunk,
                        "answer": qna_item["answer"],
                        "type": qna_item["type"],
                        "source": current_source
                    }]

                    temp = clean_answers(temp)

                    if not temp:
                        continue

                    answer = qna_item["answer"]
                    start = chunk.find(answer)

                    if start == -1:
                        continue

                    if qna_item["type"] == "definition":
                        variants = paraphrase_question(qna_item["question"])
                    else:
                        variants = []
                    variants.append(qna_item["question"])

                    variants = variants[:3]

                    for q in variants:
                        item = {
                            "question": q,
                            "context": chunk,
                            "answer": qna_item["answer"],
                            "answer_start": start,
                            "type": qna_item["type"],
                            "source": current_source
                        }

                        h = hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()

                        if h not in seen:
                            seen.add(h)
                            all_qnas.append(item)

        for s in ctxs: 
            qna = make_qna(s, "context") 
            if basic_quality_filter(qna["answer"], chunk): 
                answer = qna["answer"]
                start = chunk.find(answer)

                if start == -1:
                    continue

                qna["context"] = chunk
                qna["answer_start"] = start
                qna["source"] = current_source
                h = hashlib.md5(json.dumps(qna, sort_keys=True).encode()).hexdigest() 
                if h not in seen: 
                    seen.add(h) 
                    all_qnas.append(qna)

        if index % 20 == 0 or index == total_chunks:
            elapsed = time.time() - start_time
            remaining = (elapsed / index) * (total_chunks - index) if index else 0
            print(f"Processed {index}/{total_chunks} chunks · elapsed {format_duration(elapsed)} · remaining {format_duration(remaining)}", end="\r") 

    final = all_qnas

    os.makedirs("output", exist_ok=True) 
    # Final cleaning
    final = clean_answers(final)
    final = clean_questions(final)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f: 
        json.dump(final, f, indent=2, ensure_ascii=False) 

    elapsed = time.time() - start_time
    print()
    print(f"✅ Dataset built: {len(final)} QnA pairs in {format_duration(elapsed)}") 

# --------------------------- 
# RUN 
# --------------------------- 
if __name__ == "__main__": 
    build_dataset()
