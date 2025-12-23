# save_load_utils.py
# Small utility wrapper for safely saving & loading model artifacts (joblib / CatBoost)

import os
import joblib
from catboost import CatBoostClassifier

def ensure_dir(path: str):
    """Create directory if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


# ============================================================
# Joblib Save / Load
# ============================================================

def save_joblib(obj, filepath: str):
    """
    Safely save an object using joblib.
    """
    ensure_dir(os.path.dirname(filepath))
    joblib.dump(obj, filepath)
    print(f"[OK] Saved: {filepath}")


def load_joblib(filepath: str):
    """
    Load a joblib artifact. Raises error if missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERR] File not found: {filepath}")
    print(f"[OK] Loaded: {filepath}")
    return joblib.load(filepath)


# ============================================================
# CatBoost Save / Load
# ============================================================

def save_catboost(model: CatBoostClassifier, filepath: str):
    """
    Save a CatBoost model in .cbm format.
    """
    ensure_dir(os.path.dirname(filepath))
    model.save_model(filepath)
    print(f"[OK] Saved CatBoost model: {filepath}")


def load_catboost(filepath: str) -> CatBoostClassifier:
    """
    Load a CatBoost model from .cbm.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERR] CatBoost model missing: {filepath}")

    model = CatBoostClassifier()
    model.load_model(filepath)
    print(f"[OK] Loaded CatBoost model: {filepath}")
    return model


# ============================================================
# Helper: Check all required model files
# ============================================================

def verify_files_exist(file_list):
    """
    Input: list of file paths
    Output: True if all exist, else raises error
    """
    missing = [f for f in file_list if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"[ERR] Missing required model files:\n" + "\n".join(missing)
        )
    print("[OK] All model files verified.")
    return True


# ============================================================
# Helper: Load dictionary of joblib models
# ============================================================

def load_models_dict(dir_path: str):
    """
    Loads all *.joblib files from a directory.
    Useful for catboost_models_dict.joblib.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"[ERR] Models directory not found: {dir_path}")

    models = {}
    for file in os.listdir(dir_path):
        if file.endswith(".joblib"):
            models[file.replace(".joblib", "")] = load_joblib(
                os.path.join(dir_path, file)
            )
    return models
