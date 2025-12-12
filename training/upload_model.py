import os
import sys
import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    model_path = Path("models") / "best_model.joblib"
    if not model_path.exists():
        logger.warning(f"Model file not found at {model_path}; skipping upload.")
        sys.exit(0)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not found in environment; cannot upload to Hugging Face.")
        sys.exit(0)

    repo_id = "Disha252001/wellness-tourism-model"
    api = HfApi()

    logger.info(f"Ensuring repository {repo_id} exists (or will be created).")
    try:
        create_repo(repo_id=repo_id, token=hf_token, exist_ok=True)
    except Exception as e:
        logger.warning(f"create_repo returned warning/exception: {e}")

    logger.info(f"Uploading {model_path} to {repo_id}")
    try:
        upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="best_model.joblib",
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message="Upload best_model.joblib from CI",
        )
        logger.info("Upload complete.")
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
