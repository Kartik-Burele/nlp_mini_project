from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
import torch
from torch.utils.data import DataLoader
import json
from huggingface_hub import snapshot_download
import time

start = time.time()

model_path = snapshot_download(
    repo_id="BAAI/bge-base-en-v1.5",
    cache_dir=r"/home/administrator/Desktop/Kartik/Kartik/SemiSagev2.0 (2)/SemiSagev2.0/hf_cache", 
    local_files_only=True
)

# model = SentenceTransformer(model_path, local_files_only=True)
# model = SentenceTransformer(model_path, local_files_only=True, device="cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer(
    model_path,
    local_files_only=True,
    device=device
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"

torch.backends.cudnn.benchmark = True  # speed up kernels

with open("train.json") as f:
    train_data = json.load(f)

with open("val.json") as f:
    val_data = json.load(f)

# ✅ MNRL only needs (query, positive) pairs — in-batch negatives are automatic
train_examples = [
    InputExample(texts=[d["query"], d["positive"]])
    for d in train_data
]

# Evaluator still uses triplets for validation
val_examples = [
    InputExample(texts=[d["query"], d["positive"], d["negative"]])
    for d in val_data
]

evaluator = TripletEvaluator.from_input_examples(val_examples)

# ✅ batch_size=32 for 12GB RTX A2000 — gives 31 in-batch negatives per query
train_dataloader = DataLoader(
    train_examples, 
    shuffle=True, 
    batch_size=32,
    num_workers=4,  # parallelize data loading
    pin_memory=True, # helps with GPU transfers
    prefetch_factor=2,  # load 2 batches ahead
)

# ✅ MultipleNegativesRankingLoss — standard loss for retrieval encoders
# Uses all other positives in the batch as in-batch negatives (31 negatives per query)
train_loss = losses.MultipleNegativesRankingLoss(model)


try:
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=3,
        warmup_steps=500,
        use_amp=True,
        optimizer_params={'lr': 2e-5},
        evaluation_steps=1000,
        save_best_model=True,
        output_path="bge_finetuned2",
        checkpoint_save_steps=2000,
        checkpoint_path="checkpoints",
    )

    print("✅ Model trained and saved successfully!")

except Exception as e:
    print("❌ Training failed:", str(e))

end = time.time()

print(f"⏱ Training time: {round((end-start)/60, 2)} minutes")