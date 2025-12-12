import os
import sys
import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# CHANGE THIS → your actual best model
MODEL_FILENAME = "GradientBoosting_best_model.joblib"


def main():
    # Look for model in root OR in /models folder
    root_path = Path(".") / MODEL_FILENAME
    models_path = Path("models") / MODEL_FILENAME

    if root_path.exists():
        model_path = root_path
    elif models_path.exists():
        model_path = models_path
    else:
        logger.error(f"❌ Model file '{MODEL_FILENAME}' not found in root or /models folder.")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("❌ HF_TOKEN not found in environment; cannot upload.")
        sys.exit(1)

    repo_id = "Disha252001/wellness-tourism-model"
    api = HfApi()

    logger.info(f"Ensuring model repo exists: {repo_id}")
    try:
        create_repo(repo_id=repo_id, token=hf_token, exist_ok=True)
    except Exception as e:
        logger.warning(f"Warning during create_repo: {e}")

    logger.info(f"Uploading model: {model_path} → {repo_id}")
    try:
        upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=MODEL_FILENAME,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message=f"Upload {MODEL_FILENAME} from CI",
        )
        logger.info("✅ Upload complete.")
    except Exception as e:
        logger.exception(f"❌ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
