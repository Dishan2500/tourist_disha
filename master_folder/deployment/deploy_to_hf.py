from huggingface_hub import upload_file, create_repo
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found. Set it as an environment variable.")

# Docker Space repo
space_repo_id = "Disha252001/Tourism"

# Create the Space if it doesn't exist
create_repo(
    repo_id=space_repo_id,
    repo_type="space",
    token=HF_TOKEN,
    exist_ok=True,
    space_sdk="docker",
    private=False
)

# Files to upload
files = ["Dockerfile", "app.py", "requirements.txt"]

for file in files:
    upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=space_repo_id,
        repo_type="space",
        token=HF_TOKEN,
        commit_message=f"Add {file}"
    )

print(f"✔ All deployment files uploaded successfully to {space_repo_id}")
