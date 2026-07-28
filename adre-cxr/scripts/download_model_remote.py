import os
import sys
from huggingface_hub import snapshot_download

def main():
    repo_id = "aaditya/Llama3-OpenBioLLM-8B"
    # Target directory on remote server
    local_dir = "/home/tukl/Taimoor/ADRE-CXR/weights/Llama3-OpenBioLLM-8B"
    
    print(f"Starting download of {repo_id}...")
    print(f"Target directory (no symlinks): {local_dir}")
    
    try:
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.gguf", "*.ggml"],
            max_workers=4
        )
        print("[SUCCESS] Model downloaded and saved directly to weights directory!")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
