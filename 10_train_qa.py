"""
===========================================================
TRAIN EXTRACTIVE QA MODEL
===========================================================

Fine-tunes bert-base-uncased for extractive question answering
on the domain-specific dataset.

Input:  qa_train.json, qa_val.json (from 09_prepare_qa_data.py)
Output: qa_model_finetuned/ (saved model)

Optimized for NVIDIA RTX A2000 12GB.

===========================================================
"""

import json
import time
import torch
import numpy as np
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from datasets import Dataset

# ---------------------------
# CONFIG
# ---------------------------
TRAIN_FILE = "qa_train.json"
VAL_FILE = "qa_val.json"
OUTPUT_DIR = "qa_model_finetuned"
CHECKPOINT_DIR = "qa_checkpoints"

MAX_LENGTH = 384      # Max sequence length (question + context)
DOC_STRIDE = 128      # Sliding window stride for long contexts
BATCH_SIZE = 16       # Safe for 12GB VRAM with AMP
LEARNING_RATE = 3e-5  # Standard for BERT QA
EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# ---------------------------
# LOAD MODEL
# ---------------------------
start = time.time()

print("📥 Loading bert-base-uncased from local cache...")

model_path = snapshot_download(
    repo_id="bert-base-uncased",
    cache_dir=r"/home/administrator/Desktop/Kartik/Kartik/SemiSagev2.0 (2)/SemiSagev2.0/hf_cache",
    local_files_only=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
model.to(device)

print(f"✅ Model loaded: {model.num_parameters():,} parameters")

# ---------------------------
# LOAD DATA
# ---------------------------
print("\n🔄 Loading training data...")

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(VAL_FILE, "r", encoding="utf-8") as f:
    val_data = json.load(f)

print(f"  Train: {len(train_data)} records")
print(f"  Val:   {len(val_data)} records")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)


# ---------------------------
# TOKENIZATION
# ---------------------------
def preprocess_function(examples):
    """Tokenize questions + contexts and find answer token positions."""
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    # Tokenize with sliding window
    tokenized = tokenizer(
        questions,
        contexts,
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map from tokenized example back to original example
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]

        # If no answer, set CLS token as answer
        if len(answers["answer_start"]) == 0 or len(answers["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Find the start and end of the context in the tokenized sequence
        sequence_ids = tokenized.sequence_ids(i)

        # Find context start/end token indices
        context_start = 0
        context_end = len(sequence_ids) - 1

        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        # If the answer is not fully inside the context span, label as (0, 0)
        if (
            context_start > context_end
            or offsets[context_start][0] > start_char
            or offsets[context_end][1] < end_char
        ):
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Find the token that contains the start of the answer
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)

            # Find the token that contains the end of the answer
            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized


print("\n🔧 Tokenizing training data...")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train",
)

print("🔧 Tokenizing validation data...")
tokenized_val = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing val",
)

print(f"  Tokenized train: {len(tokenized_train)} examples")
print(f"  Tokenized val:   {len(tokenized_val)} examples")

# ---------------------------
# TRAINING
# ---------------------------
print("\n🚀 Starting training...")

# Calculate warmup steps (10% of total training steps)
total_steps = (len(tokenized_train) // BATCH_SIZE) * EPOCHS
warmup_steps = int(0.1 * total_steps)

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=warmup_steps,
    fp16=torch.cuda.is_available(),  # Mixed precision for GPU
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    logging_steps=100,
    report_to="none",  # Disable wandb/tensorboard
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

data_collator = DefaultDataCollator()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,
)

try:
    trainer.train()
    print("\n✅ Training completed!")

    # Save the best model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Model saved to {OUTPUT_DIR}/")

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    raise

end = time.time()
print(f"\n⏱ Total time: {round((end - start) / 60, 2)} minutes")
