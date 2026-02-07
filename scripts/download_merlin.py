from huggingface_hub import snapshot_download
import os

def download_merlin():
    repo_id = "stanfordmimi/Merlin"
    local_dir = "external_sources/weights/Merlin"
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading {repo_id} to {local_dir}...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print("Download complete.")

if __name__ == "__main__":
    download_merlin()
