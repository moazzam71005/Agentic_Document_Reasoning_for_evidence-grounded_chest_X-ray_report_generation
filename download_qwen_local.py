import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
print("Starting fast download of Qwen/Qwen2.5-3B-Instruct...")
snapshot_download(repo_id="Qwen/Qwen2.5-3B-Instruct", local_dir="qwen_weights")
print("Download finished!")
