import os
import sys
from huggingface_hub import HfApi, upload_file, create_repo
from pathlib import Path

"""Simple script to push deployment files into a Hugging Face Space repo.

Usage: set HF_TOKEN in env (or export), then run:
    python deployment/push_to_space.py --space-id <username/space-name>

This will create the space repo if missing and upload files from the
deployment/ directory and root `app.py`.
"""


def gather_files():
    files = []
    base = Path("deployment")
    if base.exists():
        for p in base.rglob("*"):
            if p.is_file():
                files.append(p)
    # include root app.py if present
    root_app = Path("app.py")
    if root_app.exists():
        files.append(root_app)
    return files


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--space-id", required=True, help="space id like username/space-name")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN not found in environment. Set HF_TOKEN and retry.")
        sys.exit(1)

    api = HfApi()
    print(f"Ensuring Space repo {args.space_id} exists...")
    try:
        create_repo(repo_id=args.space_id, repo_type="space", token=hf_token, exist_ok=True)
    except Exception as e:
        print(f"Warning creating repo: {e}")

    files = gather_files()
    if not files:
        print("No deployment files found to upload.")
        sys.exit(0)

    for f in files:
        path_in_repo = str(f.relative_to("."))
        print(f"Uploading {f} -> {path_in_repo}")
        try:
            upload_file(
                path_or_fileobj=str(f),
                path_in_repo=path_in_repo,
                repo_id=args.space_id,
                repo_type="space",
                token=hf_token,
                commit_message=f"Upload {path_in_repo} from CI"
            )
        except Exception as e:
            print(f"Failed to upload {f}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
