"""Helper to fetch model artifacts from the Hugging Face Hub for Spaces.

If `HF_MODEL_REPO` (or `MODEL_REPO`) is set, required model files will be
pulled into local folders so `predict_alerts_module` can load them via
relative paths. If the env var isn't set, the function is a no-op.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

try:  # optional dependency; only needed when downloading
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - handled at runtime
    hf_hub_download = None  # type: ignore

APP_DIR = Path(__file__).resolve().parent


def _detect_project_root() -> Path:
    override = os.getenv("PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    start = APP_DIR
    for cand in [start, *start.parents]:
        if (cand / "src").exists() or (cand / "catboost_models").exists() or (cand / "models").exists():
            return cand
    return start


PROJECT_ROOT = _detect_project_root()

MODEL_FILES: List[str] = [
    "catboost_models/catboost_models_dict.joblib",
    "catboost_models/label_encoders.joblib",
    "catboost_models/tfidf.joblib",
    "catboost_models/svd.joblib",
    "catboost_action_improved/catboost_action_improved.joblib",
    "catboost_action_improved/tfidf.joblib",
    "catboost_action_improved/svd.joblib",
    "catboost_action_improved/action_label_encoder.joblib",
]


def ensure_artifacts(repo_id: str | None = None, revision: str | None = None, token: str | None = None) -> Dict:
    """Download required artifacts from the Hub if configured.

    Args:
        repo_id: Optional repository id. Falls back to env var HF_MODEL_REPO or MODEL_REPO.
        revision: Optional branch/tag/commit.
        token: Optional access token (needed for private repos).

    Returns:
        Dict with status and details about each file.
    """

    repo = repo_id or os.getenv("HF_MODEL_REPO") or os.getenv("MODEL_REPO")
    if not repo:
        return {"status": "skipped", "reason": "HF_MODEL_REPO not set"}

    if hf_hub_download is None:
        return {"status": "error", "reason": "huggingface_hub not installed"}

    details: List[Tuple[str, str]] = []
    for rel_path in MODEL_FILES:
        dest = PROJECT_ROOT / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            details.append((rel_path, "exists"))
            continue
        try:
            downloaded = hf_hub_download(repo_id=repo, filename=rel_path, revision=revision, token=token)
            shutil.copy(downloaded, dest)
            details.append((rel_path, "downloaded"))
        except Exception as e:  # pragma: no cover - network/runtime issues
            details.append((rel_path, f"error: {e}"))

    return {"status": "ok", "details": details}


__all__ = ["ensure_artifacts", "MODEL_FILES"]
