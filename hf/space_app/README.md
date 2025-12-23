---
title: SOC Alert Classifier
emoji: "🚨"
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: "6.2.0"
app_file: app.py
pinned: false
---
# Hugging Face Space: SOC Alert Classifier (CSV ➜ CSV)

This Gradio Space lets you upload a CSV of alerts and returns predictions for **Status**, **Category**, and **Action Taken** using the CatBoost models from this repo.

## How it works
- The app lives in `hf/space_app/app.py`.
- It imports `predict_alerts_module` from the project `src/` and runs predictions on uploaded rows.
- Artifacts are loaded from the repo paths (`catboost_models/` and `catboost_action_improved/`).
- Optional: set `HF_MODEL_REPO` (and `HF_TOKEN` if private) to download artifacts at startup via `hf/space_app/download_artifacts.py`.

## Running locally
From project root (ensure `.venv` is active or use the provided path):

```powershell
& .venv\Scripts\python.exe hf/space_app/app.py
```

Open the printed Gradio URL and upload a CSV (e.g., `data/test_alerts.csv`). The app will show a preview and let you download the predictions CSV.

## Deploying to a Hugging Face Space
1. Create a Space (Gradio SDK, Python).
2. Push these files (including model artifacts or rely on Hub download).
3. Set Space secrets if pulling from another repo:
   - `HF_MODEL_REPO`: repo id containing the model files (e.g., `username/soc-alert-models`).
   - `HF_TOKEN`: token if the repo is private.

## Files
- `app.py` — Gradio UI and inference handler.
- `download_artifacts.py` — optional helper to fetch model files from the Hub.
- `requirements.txt` — Space dependencies (gradio, catboost, pandas, scikit-learn, joblib, huggingface_hub).
