# SemiSage v2.0 — Retriever-Reader QA System for SEMI Standards

A domain-specific Question Answering system for SEMI semiconductor industry standards. Uses a two-stage **Retriever-Reader** architecture: a fine-tuned BGE encoder retrieves relevant passages from 49K+ indexed contexts, then a fine-tuned BERT model extracts precise answer spans.

## Architecture

```
User Query → BGE Retriever (FAISS) → Top-K Passages → BERT QA Reader → Precise Answer
```

## Results

| Component | Metric | Score |
|---|---|---|
| Retriever | Recall@5 | **0.8176** |
| Retriever | MRR | 0.6680 |
| QA Reader (standalone) | F1 Score | **0.7509** |
| QA Reader (standalone) | Exact Match | 0.5479 |
| End-to-End (Top-1) | F1 Score | **0.6915** |
| End-to-End (Top-1) | Exact Match | 0.4743 |

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support (tested on RTX A2000 12GB)
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) (only needed for Step 1 — dataset generation)
- SEMI standard PDFs placed in `data/ocr_texts/` subdirectories (not included — confidential)

## Quick Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd SemiSagev2.0

# 2. Install dependencies (uv will create .venv automatically)
uv sync

# 3. Download required HuggingFace models (needs internet — one time only)
uv run download_qa_model.py   # Downloads bert-base-uncased to hf_cache/
```

> **Note**: `bge-base-en-v1.5` and `all-MiniLM-L6-v2` will be downloaded automatically
> by the training scripts on first run if not in `hf_cache/`. After that, everything
> runs **fully offline**.

## End-to-End Pipeline — How to Run

### Step 1: Generate QA Dataset from PDFs

> **Skip this step** if you already have `output2/final_dataset*.json` files.

Place your SEMI standard PDFs in `data/ocr_texts/DS*/` subdirectories, then:

```bash
# Start Ollama with the Gemma model
ollama run gemma4:e4b

# In another terminal — generate QA pairs from PDFs
uv run 08_build_dataset_llm.py
```

This processes each PDF → extracts text → chunks → generates QA pairs using the local LLM.
Output: `output2/final_dataset*.json` files in SQuAD v2.0 format.

**Config (edit in script):**
- `PDF_DIR` — path to the PDF subdirectory to process
- `OUTPUT_FILE` — output JSON filename
- `MODEL` — Ollama model name (default: `gemma4:e4b`)

### Step 2: Clean & Deduplicate Dataset

```bash
uv run 02_clean_dataset_pipeline.py
```

- **Input**: `output2/final_dataset*.json` (SQuAD v2.0 format)
- **Output**: `cleaned_dataset.json` (~49K QA pairs)
- **What it does**: Merges all files → cleans OCR noise → validates answer spans → removes exact + semantic duplicates

### Step 3: Prepare Retrieval Training Data

```bash
uv run 03_prepare_training_data.py
```

- **Input**: `cleaned_dataset.json`
- **Output**: `train.json`, `val.json`, `test.json` (triplets with query/positive/negative)
- **What it does**: Creates training triplets with `"query: "` prefix, windowed positives, and answer-leakage-filtered negatives

### Step 4: Train Retrieval Encoder (BGE)

```bash
uv run 04_train_bge.py
```

- **Input**: `train.json`, `val.json`
- **Output**: `bge_finetuned2/` (fine-tuned BGE model)
- **Training**: MNRL loss, batch=32, lr=2e-5, 3 epochs
- **Checkpoints**: Saved to `checkpoints/` every 2000 steps

### Step 5: Build FAISS Index

```bash
uv run 05_build_faiss.py
```

- **Input**: `train.json`, `val.json`, `test.json` + `bge_finetuned2/`
- **Output**: `faiss_index.bin`, `contexts.json`
- **What it does**: Encodes all unique positive passages → builds FAISS IndexFlatIP (cosine similarity)

### Step 6: Evaluate Retrieval

```bash
uv run 07_evaluate.py
```

- **Output**: Recall@1/3/5, MRR, F1, sample queries

### Step 7: Prepare QA Training Data

```bash
uv run 09_prepare_qa_data.py
```

- **Input**: `cleaned_dataset.json`
- **Output**: `qa_train.json`, `qa_val.json`, `qa_test.json`

### Step 8: Train QA Reader (BERT)

```bash
uv run 10_train_qa.py
```

- **Input**: `qa_train.json`, `qa_val.json`
- **Output**: `qa_model_finetuned/` (fine-tuned BERT QA model)
- **Training**: 3 epochs, batch=16, lr=3e-5, ~82 minutes on RTX A2000

### Step 9: Evaluate QA Model (Standalone)

```bash
uv run 11_evaluate_qa.py
```

- **Output**: Exact Match, F1, sample predictions

### Step 10: Evaluate Full Pipeline (End-to-End)

```bash
uv run 13_evaluate_pipeline.py
```

- **Output**: Retrieval Recall@K + End-to-End EM/F1

### Step 11: Interactive Demo

```bash
# Interactive mode
uv run 12_pipeline.py

# Single query mode
uv run 12_pipeline.py "What is ACK code?"

# Search mode (with optional QA)
uv run 06_search.py
```

## Project Structure

```
SemiSagev2.0/
├── 01_build_dataset.py           # Original dataset builder (legacy)
├── 02_clean_dataset_pipeline.py  # Clean & deduplicate dataset
├── 03_prepare_training_data.py   # Create retrieval training triplets
├── 04_train_bge.py               # Train BGE retrieval encoder
├── 05_build_faiss.py             # Build FAISS vector index
├── 06_search.py                  # Search API with optional QA
├── 07_evaluate.py                # Evaluate retrieval metrics
├── 08_build_dataset_llm.py       # Generate QA pairs using local LLM
├── 09_prepare_qa_data.py         # Prepare QA training data
├── 10_train_qa.py                # Train BERT QA reader
├── 11_evaluate_qa.py             # Evaluate QA standalone
├── 12_pipeline.py                # Combined retriever-reader pipeline
├── 13_evaluate_pipeline.py       # End-to-end evaluation
├── download_qa_model.py          # One-time model download script
├── pyproject.toml                # Dependencies
├── data/ocr_texts/               # PDF source files (NOT included)
├── output2/                      # Generated QA dataset (SQuAD v2.0 format)
├── hf_cache/                     # Cached HuggingFace models
├── bge_finetuned2/               # Fine-tuned retrieval model
├── qa_model_finetuned/           # Fine-tuned QA model
├── checkpoints/                  # Training checkpoints
└── Squad2.0/                     # SQuAD 2.0 benchmark experiments
```

## Minimum Files Needed to Run Inference (Demo Only)

If you just want to run the interactive pipeline without retraining:

```
├── 06_search.py
├── 12_pipeline.py
├── bge_finetuned2/         # Fine-tuned retrieval model (~419MB)
├── qa_model_finetuned/     # Fine-tuned QA model (~417MB)
├── faiss_index.bin         # FAISS index (~144MB)
├── contexts.json           # Passage corpus (~13MB)
└── pyproject.toml          # Dependencies
```

## Minimum Files Needed to Retrain from Scratch

```
├── 02_clean_dataset_pipeline.py
├── 03_prepare_training_data.py
├── 04_train_bge.py
├── 05_build_faiss.py
├── 07_evaluate.py
├── 09_prepare_qa_data.py
├── 10_train_qa.py
├── 11_evaluate_qa.py
├── 12_pipeline.py
├── 13_evaluate_pipeline.py
├── download_qa_model.py
├── output2/                # Generated dataset (or regenerate with 08_build_dataset_llm.py)
├── hf_cache/               # Base models (downloaded once)
└── pyproject.toml
```

## Technology Stack

| Component | Technology |
|---|---|
| PDF extraction | PyMuPDF |
| Dataset generation | Gemma 4B (local, via Ollama) |
| Retrieval encoder | BAAI/bge-base-en-v1.5 |
| Vector search | FAISS (IndexFlatIP, cosine similarity) |
| QA reader | bert-base-uncased |
| Framework | HuggingFace Transformers + Sentence-Transformers |
| Package manager | uv |
| Hardware | NVIDIA RTX A2000 12GB |