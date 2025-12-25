# SOC-Alert-ML-Classification

## CI/CD (GitHub Actions + Hugging Face)

This repo includes automated workflows for training/evaluating models and deploying the Gradio app to a Hugging Face Space.

Workflows:
- CI: .github/workflows/ci-train-eval.yml
	- Installs deps, trains CatBoost models, runs evaluation, and uploads artifacts (models and metrics).
- CD: .github/workflows/cd-deploy-hf.yml
	- Syncs the Gradio app under hf/space_app and model artifacts from models/ to a Hugging Face Space.

Configure repository secrets:
- HF_TOKEN: a Hugging Face access token with write permission.
- HF_SPACE_ID: target Space repo id, e.g. "org-or-user/your-space".

Run locally:
```powershell
# From repo root on Windows
& .venv\Scripts\python.exe -m pip install -r requirements.txt
& .venv\Scripts\python.exe -m pip install -r hf\space_app\requirements.txt
$env:GRADIO_SERVER_NAME="127.0.0.1"; $env:GRADIO_SERVER_PORT="7861"
& .venv\Scripts\python.exe hf\space_app\app.py
```

Notes:
- Training scripts expect data under data/. CI moves resulting artifacts to the models/ folder and mirrors root-level folders for evaluation compatibility.
- The Space app will use local artifacts if present, or can download from a Hub repo if the env var HF_MODEL_REPO is set in Space secrets.
