"""
One-time script to download bert-base-uncased for QA fine-tuning.
Run this with internet access, then everything runs offline.
"""
from huggingface_hub import snapshot_download

print("📥 Downloading bert-base-uncased to hf_cache/...")

model_path = snapshot_download(
    repo_id="bert-base-uncased",
    cache_dir="hf_cache"
)

print(f"✅ Model downloaded to: {model_path}")
print("You can now run all scripts offline.")
