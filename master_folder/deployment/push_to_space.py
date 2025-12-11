import os
from huggingface_hub import create_repo, upload_file
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is required.")

SPACE_ID = "Disha252001/wellness-tourism-space"  
LOCAL_DEPLOY_DIR = Path(__file__).resolve().parent  

# 1) Create Space as DOCKER TYPE (correct)
try:
    create_repo(
        repo_id=SPACE_ID,
        token=HF_TOKEN,
        repo_type="space",
        private=False,
        space_sdk="docker"      # <-- FIXED
    )
    print(f"Space created: {SPACE_ID}")
except Exception as e:
    print(f"create_repo info (safe to ignore if exists): {e}")

# 2) Upload files
files_to_upload = ["Dockerfile", "requirements.txt", "app.py", ".gitignore"]

for fname in files_to_upload:
    local_path = LOCAL_DEPLOY_DIR / fname
    if not local_path.exists():
        print(f"Skipping {fname} — not found.")
        continue

    print(f"Uploading {fname}...")
    upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=fname,
        repo_id=SPACE_ID,
        repo_type="space",
        token=HF_TOKEN
    )

print(f"Done! Visit your Space:")
print(f"https://huggingface.co/spaces/{SPACE_ID}")
