import os
import json
import time
import re
import uuid
import hashlib
import logging
import requests
import fitz  # PyMuPDF
from tqdm import tqdm

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("dataset_build.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= CONFIG =================
PDF_DIR = "data/ocr_texts/DS17"
OUTPUT_DIR = "output2"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_dataset17.json")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

# Model: adjust based on your GPU VRAM
# GTX 1650 4GB → use 4B param models or smaller
MODEL = "gemma4:e4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Optimized for GTX 1650 4GB VRAM + i5 9th gen + 24GB RAM
CHUNK_SIZE = 250       # Reduced from 450 for better span accuracy on smaller models
OVERLAP = 50           # Reduced from 100 to avoid excessive duplication

MIN_QA_PER_CHUNK = 2   # Lowered: smaller chunks may yield fewer QAs
MAX_QA_PER_CHUNK = 8

MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 1
# ==========================================


# -------- OCR TEXT CLEANING (ported from 01_build_dataset.py) --------
def clean_text(text):
    """Clean OCR noise from extracted PDF text."""
    # Remove page numbers on their own lines
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Remove control characters and common OCR artifacts
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x05■│]+', '', text)
    # Remove non-ASCII characters (OCR garbage like ©, ®, garbled symbols)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Fix hyphenated line breaks (e.g., "diam-\neter" → "diameter")
    text = re.sub(r'-\n', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -------- PDF TEXT EXTRACTION --------
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF with proper cleanup."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
    return text


# -------- LLM CALL --------
def call_llm(prompt):
    """Call Ollama LLM with proper error handling and retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,     # Lower = more precise span extraction
                        "num_predict": 2048,    # Enough tokens for JSON output
                    }
                },
                timeout=180
            )
            response.raise_for_status()

            text = response.json().get("response", "")
            # Strip markdown code fences
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            return text.strip()

        except requests.exceptions.Timeout:
            logger.warning(f"LLM timeout (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(3)
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {OLLAMA_URL}. Is it running?")
            time.sleep(5)
        except Exception as e:
            logger.warning(f"LLM call error (attempt {attempt + 1}): {type(e).__name__}: {e}")
            time.sleep(2)

    return None


# -------- CHUNKING --------
def chunk_text(text):
    """Split text into overlapping chunks with minimum length filter."""
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - OVERLAP

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + CHUNK_SIZE])

        # Skip chunks too short to generate meaningful QAs
        if len(chunk.split()) > 80:
            chunks.append(chunk)

    return chunks


# -------- PROMPT (with concrete example for better LLM compliance) --------
def build_prompt(context):
    return f"""You are an expert SQuAD dataset generator.

Read the context and generate question-answer pairs.

STRICT RULES:
1. Generate {MIN_QA_PER_CHUNK} to {MAX_QA_PER_CHUNK} QnA pairs
2. Answers MUST be EXACT text spans copied from the context (no paraphrasing)
3. No yes/no questions
4. No duplicate questions
5. Mix question types: "definition", "fact", "reasoning"
6. For table-like data, generate one QnA per row/entry

IMPORTANT: The "answer" must be a word-for-word substring from the context.

Return ONLY valid JSON in this exact format:
{{
  "qas": [
    {{
      "type": "definition",
      "question": "What is a glass carrier wafer?",
      "answer": "a glass wafer used for bonding to a device wafer temporarily during one or more process steps"
    }},
    {{
      "type": "fact",
      "question": "What is the nominal diameter specified for glass carrier wafers?",
      "answer": "200 mm and 300 mm"
    }}
  ]
}}

Context:
{context}"""


# -------- ANSWER QUALITY FILTERS --------
def is_clean_answer(answer):
    """Filter out low-quality answers."""
    # Too short to be meaningful
    if len(answer) < 5:
        return False

    # Too long (likely grabbed entire paragraph)
    if len(answer) > 500:
        return False

    # Contains leftover control characters
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', answer):
        return False

    # Mostly non-alphanumeric (table formatting artifacts)
    alnum_count = sum(c.isalnum() or c.isspace() for c in answer)
    if alnum_count / max(len(answer), 1) < 0.6:
        return False

    return True


def is_clean_question(question):
    """Filter out low-quality questions."""
    # Too short
    if len(question.split()) < 3:
        return False

    # Must end with question mark
    if not question.strip().endswith("?"):
        return False

    # Contains control characters
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', question):
        return False

    return True


# -------- GENERATE --------
def generate_qna(context):
    """Generate QnA pairs from context using LLM."""
    output = call_llm(build_prompt(context))

    if not output:
        logger.warning("LLM returned empty response")
        return None

    try:
        # Sometimes LLM adds text before/after the JSON block
        json_match = re.search(r'\{[\s\S]*\}', output)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(output)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        logger.debug(f"Raw LLM output: {output[:300]}")
        return None


# -------- VALIDATE --------
def validate_qas(qas, context):
    """Validate QA pairs: check span existence, apply quality filters, deduplicate."""
    valid = []
    seen_questions = set()
    seen_answers = set()

    for qa in qas:
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        qtype = qa.get("type", "fact").strip()

        # Basic presence check
        if not q or not a:
            continue

        # Quality filters
        if not is_clean_question(q):
            continue
        if not is_clean_answer(a):
            continue

        # Deduplicate by question text
        q_lower = q.lower()
        if q_lower in seen_questions:
            continue
        seen_questions.add(q_lower)

        # Deduplicate by answer text (avoid same span, different phrasing)
        a_hash = hashlib.md5(a.lower().encode()).hexdigest()
        if a_hash in seen_answers:
            continue
        seen_answers.add(a_hash)

        # Verify answer exists as exact span in context
        start = context.find(a)
        if start == -1:
            # Fallback: case-insensitive search
            context_lower = context.lower()
            a_lower = a.lower()
            start = context_lower.find(a_lower)
            if start != -1:
                # Use the actual text from context to preserve original casing
                a = context[start:start + len(a)]
            else:
                continue

        valid.append({
            "id": str(uuid.uuid4()),
            "question": q,
            "answers": [
                {
                    "text": a,
                    "answer_start": start
                }
            ],
            "type": qtype
        })

    return valid


# -------- PROCESS SINGLE PDF --------
def process_pdf(pdf_path):
    """Process a single PDF: extract → clean → chunk → generate QAs."""
    filename = os.path.basename(pdf_path)
    logger.info(f"Processing: {filename}")

    # Extract and clean text
    raw_text = extract_text_from_pdf(pdf_path)
    if len(raw_text.strip()) < 200:
        logger.warning(f"Skipping {filename}: too little text ({len(raw_text)} chars)")
        return []

    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)
    logger.info(f"  {filename}: {len(chunks)} chunks from {len(cleaned)} chars")

    results = []

    for i, chunk in enumerate(chunks):
        qna = generate_qna(chunk)

        if qna and "qas" in qna:
            valid_qas = validate_qas(qna["qas"], chunk)

            if len(valid_qas) >= MIN_QA_PER_CHUNK:
                results.append({
                    "context": chunk,
                    "qas": valid_qas,
                    "source": filename
                })
                logger.info(f"  Chunk {i+1}/{len(chunks)}: {len(valid_qas)} valid QAs")
            else:
                logger.debug(f"  Chunk {i+1}/{len(chunks)}: only {len(valid_qas)} valid QAs (below threshold, skipped)")
        else:
            logger.debug(f"  Chunk {i+1}/{len(chunks)}: LLM failed or returned no QAs")

        time.sleep(SLEEP_BETWEEN_CALLS)

    return results


# -------- CHECKPOINT / RESUME --------
def save_checkpoint(processed_files, all_data):
    """Save progress after each file for crash recovery."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint = {
        "processed_files": processed_files,
        "data_count": len(all_data)
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)

    # Save intermediate data alongside
    intermediate = os.path.join(OUTPUT_DIR, "intermediate_data.json")
    with open(intermediate, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)


def load_checkpoint():
    """Load previous checkpoint and intermediate data if they exist."""
    if not os.path.exists(CHECKPOINT_FILE):
        return [], []

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    processed_files = checkpoint.get("processed_files", [])
    all_data = []

    intermediate = os.path.join(OUTPUT_DIR, "intermediate_data.json")
    if os.path.exists(intermediate):
        with open(intermediate, "r", encoding="utf-8") as f:
            all_data = json.load(f)

    return processed_files, all_data


# -------- MAIN PIPELINE --------
def build_dataset():
    """Main dataset build pipeline with checkpoint/resume support."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try to resume from checkpoint
    processed_files, all_data = load_checkpoint()
    if processed_files:
        logger.info(f"Resumed from checkpoint: {len(processed_files)} files done, {len(all_data)} contexts")

    # Discover PDF files
    files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    remaining = [f for f in files if f not in processed_files]

    if not remaining:
        logger.info("All files already processed.")
    else:
        logger.info(f"Files to process: {len(remaining)}/{len(files)}")

    # Process each PDF
    for file in tqdm(remaining, desc="Processing PDFs"):
        path = os.path.join(PDF_DIR, file)

        try:
            data = process_pdf(path)
            all_data.extend(data)
            processed_files.append(file)

            new_qas = sum(len(p["qas"]) for p in data)
            total_qas = sum(len(p["qas"]) for p in all_data)
            logger.info(f"  {file}: +{new_qas} QAs | Running total: {total_qas}")

            # Checkpoint after each file (crash-safe)
            save_checkpoint(processed_files, all_data)

        except Exception as e:
            logger.error(f"Error processing {file}: {e}", exc_info=True)

    # -------- Build proper SQuAD v2.0 JSON --------
    # Group paragraphs by source PDF (each PDF becomes a SQuAD "document")
    paragraphs_by_source = {}
    for item in all_data:
        source = item.get("source", "unknown")
        if source not in paragraphs_by_source:
            paragraphs_by_source[source] = []
        paragraphs_by_source[source].append({
            "context": item["context"],
            "qas": item["qas"]
        })

    squad = {
        "version": "v2.0",
        "data": [
            {
                "title": source.replace(".pdf", ""),
                "paragraphs": paragraphs
            }
            for source, paragraphs in paragraphs_by_source.items()
        ]
    }

    # Write final output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(squad, f, indent=2, ensure_ascii=False)

    # -------- Stats --------
    total_docs = len(squad["data"])
    total_contexts = sum(len(d["paragraphs"]) for d in squad["data"])
    total_questions = sum(
        len(p["qas"])
        for d in squad["data"]
        for p in d["paragraphs"]
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"Dataset saved to {OUTPUT_FILE}")
    logger.info(f"Documents: {total_docs}")
    logger.info(f"Total contexts: {total_contexts}")
    logger.info(f"Total QA pairs: {total_questions}")
    logger.info(f"{'='*50}")

    # Clean up checkpoint after successful completion
    for cleanup_file in [CHECKPOINT_FILE, os.path.join(OUTPUT_DIR, "intermediate_data.json")]:
        if os.path.exists(cleanup_file):
            os.remove(cleanup_file)

    print(f"\n{'='*50}")
    print(f"  Dataset saved to {OUTPUT_FILE}")
    print(f"  Documents: {total_docs}")
    print(f"  Total contexts: {total_contexts}")
    print(f"  Total QA pairs: {total_questions}")
    print(f"{'='*50}")


if __name__ == "__main__":
    build_dataset()