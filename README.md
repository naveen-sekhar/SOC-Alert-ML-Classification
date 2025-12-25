# SOC-Alert-ML-Classification

A machine learning-based Security Operations Center (SOC) alert classification system that automatically predicts **Status**, **Category**, and **Action Taken** for security alerts using CatBoost models with CI/CD automation via GitHub Actions and Hugging Face Spaces.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [How It Works](#how-it-works)
4. [Model Architecture](#model-architecture)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Setup & Usage](#setup--usage)

---

## Project Overview

### What is this project?

This project builds an automated pipeline to **classify SOC (Security Operations Center) alerts** into three key dimensions:
- **Status**: Malicious / Legitimate / Unknown
- **Category**: Malware, Reconnaissance, Policy Violation, Benign Activity, False Positive
- **Action Taken**: Blocked, Escalated, Investigated, Closed, No Action Required

The system uses machine learning (CatBoost) trained on synthetic yet realistic SIEM alert data, and is deployed as a web-based Gradio app on Hugging Face Spaces for easy access and batch predictions.

### Use Cases
- **SOC Teams**: Automate alert triage and reduce manual review overhead
- **Incident Responders**: Prioritize high-risk alerts (Malicious + Escalate)
- **Security Analysts**: Quickly assess alert legitimacy and recommended actions

---

## Directory Structure

```
SOC-Alert-ML-Classification/
├── .github/
│   └── workflows/
│       ├── ci-train-eval.yml          # CI: Train models, evaluate, upload artifacts
│       └── cd-deploy-hf.yml           # CD: Deploy app + models to HF Space
├── data/
│   ├── smart_siem_dataset.csv         # Original synthetic SIEM alert dataset (1500 samples)
│   ├── cleaned_smart_siem_dataset.csv # Cleaned dataset after preprocessing
│   ├── fe_smart_siem_dataset.csv      # Feature-engineered dataset for Action model training
│   ├── test_alerts.csv                # Sample alerts for Gradio Space demo
│   ├── test_alerts_with_preds.csv     # Predictions on test alerts
│   └── metrics/                       # Evaluation results (confusion matrices, reports)
├── models/
│   ├── catboost_models/               # Baseline models (Status, Category, Action)
│   │   ├── catboost_Status.cbm        # Trained CatBoost for Status prediction
│   │   ├── catboost_Category.cbm      # Trained CatBoost for Category prediction
│   │   ├── catboost_Action Taken.cbm  # Trained CatBoost for Action Taken prediction
│   │   ├── catboost_models_dict.joblib
│   │   ├── label_encoders.joblib      # LabelEncoders for all 3 targets
│   │   ├── tfidf.joblib               # TF-IDF vectorizer (executive summary → features)
│   │   └── svd.joblib                 # SVD reducer (TF-IDF → 50 dimensions)
│   └── catboost_action_improved/      # Improved Action model (with class weights & feature engineering)
│       ├── catboost_action_improved.cbm
│       ├── catboost_action_improved.joblib
│       ├── tfidf.joblib
│       ├── svd.joblib
│       └── action_label_encoder.joblib
├── src/
│   ├── 01_Dataset_Generation.py       # Generate synthetic SIEM alert data
│   ├── 02_Data_Preprocessing.py       # Clean & normalize data
│   ├── 03_Feature_Engineering_Action.py # Engineer additional features for Action model
│   ├── 04_CatBoost_Full_Train.py      # Train baseline Status/Category/Action models
│   ├── 05_CatBoost_Action_Improved.py # Train improved Action model with class weights
│   ├── 06_predict_alerts.py           # Standalone prediction script
│   ├── evaluate_metrics.py            # Compute confusion matrices & classification reports
│   ├── evaluate_batch.py              # Batch evaluation on CSV files
│   ├── plot_confusion_matrices.py     # Visualize evaluation results
│   ├── predict_alerts_module.py       # Reusable prediction module (used by Gradio app)
│   └── utils/
│       └── save_load_utils.py         # Utility functions for model I/O
├── hf/
│   ├── space_app/                     # Gradio web app for Hugging Face Spaces
│   │   ├── app.py                     # Main Gradio app (CSV-in → CSV-out predictions)
│   │   ├── predict_alerts_module.py   # Local copy of prediction module
│   │   ├── download_artifacts.py      # Optional artifact downloader from Hub
│   │   └── requirements.txt           # Space runtime dependencies
│   └── hf_push_model.py               # Script to push model artifacts to Hub (optional)
├── infra/
│   └── Dockerfile                     # Container image for deployment (optional)
├── catboost_info/                     # CatBoost training logs & metrics (auto-generated)
├── requirements.txt                   # Python dependencies (training, inference, evaluation)
├── README.md                          # This file
└── .venv/                             # Virtual environment (local development)
```

### Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| `.github/workflows/` | GitHub Actions CI/CD automation |
| `data/` | Raw, cleaned, and engineered datasets; evaluation metrics |
| `models/` | Trained CatBoost models & preprocessors (TF-IDF, SVD, encoders) |
| `src/` | Python scripts for data pipeline, training, evaluation, inference |
| `hf/space_app/` | Gradio web app bundled for Hugging Face Spaces deployment |
| `infra/` | Docker & infrastructure templates (optional) |

---

## How It Works

### 1. Data Pipeline

**Step 1: Dataset Generation** (`src/01_Dataset_Generation.py`)
- Creates synthetic yet realistic SIEM alert data (1500 samples)
- Generates alerts with fields: Date, Time, Alert Name, Severity, Source Host, Destination IP, Protocol/Port, Executive Summary
- Labels each alert with ground-truth: Status, Category, Action Taken
- **Output**: `data/smart_siem_dataset.csv`

**Step 2: Data Preprocessing** (`src/02_Data_Preprocessing.py`)
- **Datetime Features**: Extract Hour, Minute, DayOfWeek, IsWeekend from Date+Time
- **Protocol/Port Split**: Parse "TCP port 443" → Protocol="TCP", Port=443
- **IP Features**: Detect private IPs (10.x.x.x, 172.16–31.x.x, 192.168.x.x), check same-subnet
- **Text Cleanup**: Normalize Executive Summary (whitespace, case)
- **Output**: `data/cleaned_smart_siem_dataset.csv`

**Step 3: Feature Engineering** (`src/03_Feature_Engineering_Action.py`)
- Extract **keyword features** from Executive Summary:
  - `has_malware_kw`: keywords like "trojan", "malware", "beacon", "c2"
  - `has_recon_kw`: keywords like "scan", "brute force"
  - `has_policy_kw`: keywords like "unauthorized", "ssh", "login failed"
- Extract **numeric patterns**: flow count, bytes transferred, duration
- Compute **severity ordinal** (Low=0, Medium=1, High=2, Critical=3)
- Identify **high-risk ports** (22, 445, 3389, 1492, etc.)
- Flag **off-hours activity** (0–5 AM)
- **Output**: `data/fe_smart_siem_dataset.csv`

### 2. Model Training & Deployment

**Baseline Models** (`src/04_CatBoost_Full_Train.py`)
- Train **three independent CatBoost classifiers**:
  - **Status**: Binary (Malicious / Legitimate)
  - **Category**: Multiclass (5 classes)
  - **Action Taken**: Multiclass (5 classes)
- Features: Categorical (Alert Name, Severity, Protocol) + Numeric + Text (TF-IDF reduced)
- **Output**: `models/catboost_models/`

**Improved Action Model** (`src/05_CatBoost_Action_Improved.py`)
- Retrains Action Taken model with **class weights** to handle imbalanced data
- Uses engineered features from Step 3
- Better precision/recall on minority classes
- **Output**: `models/catboost_action_improved/`

### 3. Inference & Evaluation

**Prediction Module** (`src/predict_alerts_module.py`)
- Loads trained models & preprocessors
- Accepts alert dict/list, preprocesses (same pipeline as training)
- Returns Status, Category, Action Taken predictions
- Used by both standalone script (`06_predict_alerts.py`) and Gradio app

**Evaluation** (`src/evaluate_metrics.py`)
- Computes confusion matrices & classification reports per target
- **Output**: `data/metrics/{target}_confusion.csv`, `{target}_report.txt`

---

## Model Architecture

### Model Type: CatBoost Classifier

**Why CatBoost?**
- Handles categorical features natively (no one-hot encoding needed)
- Built-in text preprocessing via TF-IDF + SVD
- Fast training & inference
- Good for imbalanced classification with class weights
- Interpretable feature importance

### Feature Composition

Each model receives a combined feature matrix:

```
Features = [Categorical] + [Numeric] + [Text Embeddings]
```

| Feature Type | Examples | Count |
|--------------|----------|-------|
| **Categorical** | Alert Name, Severity, Protocol | 3 |
| **Numeric** | Hour, Minute, DayOfWeek, IsWeekend, Port, Src_Private, Dst_Private, Same_Subnet, keyword_count, severity_ordinal, high_risk_port, off_hours, flow_count, bytes_transferred, duration_sec | 15 |
| **Text** | TF-IDF(Executive Summary) → SVD(50) | 50 |
| **Total** | — | **68** |

### Text Processing Pipeline

```
Executive Summary (raw text)
    ↓
TfidfVectorizer (max_features=2000, ngram_range=(1,2))
    ↓
TruncatedSVD (n_components=50)
    ↓
50-dimensional embedding
```

**Why TF-IDF + SVD?**
- TF-IDF captures keyword importance (e.g., "trojan", "scan")
- SVD reduces dimensionality to prevent overfitting
- Fast inference (~milliseconds)

### Model Hyperparameters

| Parameter | Baseline | Improved Action |
|-----------|----------|-----------------|
| Iterations | 2000 | 1500 |
| Learning Rate | 0.03 | 0.03 |
| Depth | 8 | 8 |
| Early Stopping Rounds | 100 | 80 |
| Class Weights | None | Balanced (imbalance-aware) |
| Loss Function | Logloss (binary) / MultiClass | MultiClass + weighted |

---

## Model Usage in This Project

### Training Flow

```
raw_data.csv
    ↓ [Preprocessing]
cleaned_data.csv
    ↓ [Feature Engineering]
fe_data.csv
    ↓ [Train/Test Split (80/20)]
X_train, X_test, y_train, y_test
    ↓ [CatBoost.fit(X_train, y_train)]
trained_model.cbm + preprocessors (TF-IDF, SVD, encoders)
    ↓ [Save to models/]
models/catboost_models/ and models/catboost_action_improved/
```

### Inference Flow

```
new_alert (dict)
    ↓ [preprocess_alerts() in predict_module]
    → Datetime parsing, IP feature extraction, keyword detection
    ↓
feature_matrix (68-dim)
    ↓ [CatBoost.predict()]
    ↓
Predictions: Status, Category, Action Taken
```

### In the Gradio App

```
User uploads CSV with raw alerts
    ↓
hf/space_app/app.py
    ↓ [handle_file_upload()]
    → Calls predict_alerts_module.predict_alerts()
    ↓
Returns augmented CSV with predictions
    → User downloads or views in table
```

---

## CI/CD Pipeline

### Overview

The project uses **GitHub Actions** to automate model training, evaluation, and deployment to Hugging Face Spaces.

```
Code Push to main
    ↓
┌─ CI: Train & Evaluate ──┐
│ • Setup Python 3.11     │
│ • Install deps          │
│ • Train models          │
│ • Evaluate metrics      │
│ • Upload artifacts      │
└─────────────────────────┘
    ↓
┌─ CD: Deploy to HF Space ┐
│ • Prepare payload       │
│ • Create Space (if new) │
│ • Sync app + models     │
│ • App goes live         │
└─────────────────────────┘
```

### CI Pipeline Details (`.github/workflows/ci-train-eval.yml`)

**Triggers**: On push to `main` or on pull requests

**Job: `train-eval`** (runs on `ubuntu-latest`)

| Step | Action | Details |
|------|--------|---------|
| 1. Checkout | `actions/checkout@v4` | Clone repo |
| 2. Setup Python | `actions/setup-python@v5` | Python 3.11, pip cache enabled |
| 3. Install deps | `pip install -r requirements.txt` | Install CatBoost, scikit-learn, pandas, etc. |
| 4. Train baseline | `python src/04_CatBoost_Full_Train.py` | Train Status/Category/Action models from `data/cleaned_smart_siem_dataset.csv` |
| 5. Train improved | `python src/05_CatBoost_Action_Improved.py` | Retrain Action with class weights from `data/fe_smart_siem_dataset.csv` |
| 6. Stage artifacts | Move `data/catboost_*` → `models/catboost_*` | Organize models in canonical location |
| 7. Mirror for eval | Copy `models/catboost_*` → repo root | Ensure evaluation scripts find models |
| 8. Evaluate | `python src/evaluate_metrics.py` | Compute confusion matrices & reports |
| 9. Upload artifacts | `actions/upload-artifact@v4` | Store `models/` and `data/metrics/` for download |

**Outputs**:
- `models/catboost_models/` (trained baseline)
- `models/catboost_action_improved/` (trained improved model)
- `data/metrics/` (confusion matrices, classification reports)
- GitHub Actions Artifacts (downloadable, 90-day retention)

### CD Pipeline Details (`.github/workflows/cd-deploy-hf.yml`)

**Triggers**: On push to `main` or manual `workflow_dispatch`

**Job: `deploy-space`** (runs on `ubuntu-latest`)

| Step | Action | Details |
|------|--------|---------|
| 1. Checkout | `actions/checkout@v4` | Clone repo |
| 2. Setup Python | `actions/setup-python@v5` | Python 3.11 |
| 3. Install HF CLI | `pip install -U "huggingface_hub[cli]"` | Install Hugging Face Hub tools |
| 4. Validate secrets | Check `HF_SPACE_ID`, `HF_TOKEN` | Fail if secrets missing |
| 5. Ensure Space | `HfApi.create_repo()` | Create Space if doesn't exist (gradio, public) |
| 6. Stage payload | Gather `hf/space_app/` + models | Prepare upload directory |
| 7. Upload to Space | `hf upload {SPACE_ID} ./space_payload .` | Sync all files to Space root |

**Environment Variables**:
- `HF_SPACE_ID`: e.g., `"username/soc-alert-classifier"` (set in GitHub Secrets)
- `HF_TOKEN`: Hugging Face API token with write access (set in GitHub Secrets)

**Outputs**:
- Gradio Space updated with latest app code + trained models
- App accessible at `https://huggingface.co/spaces/{HF_SPACE_ID}`

### CI/CD Workflow Execution

```
1. Developer pushes to main
2. GitHub Actions triggers both workflows in parallel
3. CI: Trains models, evaluates, artifacts ready (~15 min)
4. CD: Waits for artifacts or starts simultaneously
   → Gathers artifacts (from CI or local models/)
   → Uploads to Space
5. Space is live with new models & app code
6. User can immediately use the Space for predictions
```

---

## Setup & Usage

### Local Development

**Prerequisites**:
- Python 3.11+
- Git
- Windows PowerShell or Linux bash

**1. Clone & Setup Venv**
```powershell
git clone https://github.com/yourusername/SOC-Alert-ML-Classification.git
cd SOC-Alert-ML-Classification
python -m venv .venv
& .venv\Scripts\Activate.ps1
```

**2. Install Dependencies**
```powershell
pip install -r requirements.txt
pip install -r hf\space_app\requirements.txt
```

**3. Run Locally**

**Train models**:
```powershell
cd data
python ..\src\04_CatBoost_Full_Train.py
python ..\src\05_CatBoost_Action_Improved.py
cd ..
```

**Evaluate**:
```powershell
python src\evaluate_metrics.py --input data\smart_siem_dataset.csv --output-dir data\metrics
```

**Run Gradio App**:
```powershell
$env:GRADIO_SERVER_NAME="127.0.0.1"
$env:GRADIO_SERVER_PORT="7861"
& .venv\Scripts\python.exe hf\space_app\app.py
```

Open browser: `http://127.0.0.1:7861`

### Deployment to Hugging Face

**1. Create a Hugging Face Space**
- Visit https://huggingface.co/spaces
- Create new Space (SDK: Gradio, Public/Private as desired)
- Note the Space ID (e.g., `username/soc-alert-classifier`)

**2. Configure GitHub Secrets**
- Go to GitHub repo → Settings → Secrets and variables → Actions
- Add `HF_TOKEN`: Your Hugging Face API token (Settings → Access Tokens)
- Add `HF_SPACE_ID`: Your Space ID

**3. Push to Main**
```powershell
git add .
git commit -m "Trigger CI/CD"
git push origin main
```

**4. Monitor Workflows**
- GitHub repo → Actions
- Watch `CI - Train & Evaluate` and `CD - Deploy to HF Space`
- Once done, Space is live!

---

## Configuration

### GitHub Secrets

| Secret | Description | Example |
|--------|-------------|---------|
| `HF_TOKEN` | Hugging Face API token (write access) | `hf_abc123xyz...` |
| `HF_SPACE_ID` | Target Space repo ID | `username/soc-alert-classifier` |

### Environment Variables (Gradio App)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Bind address (localhost or 0.0.0.0) |
| `GRADIO_SERVER_PORT` | `7860` | Port number |
| `HF_MODEL_REPO` | (not set) | Optional: download models from Hub instead of local |

---

## Troubleshooting

### CI Workflow Fails

**"Cannot find data file"**
- Check that `data/smart_siem_dataset.csv` and `data/fe_smart_siem_dataset.csv` exist
- Scripts in `src/` expect these to be pre-generated

**"Module not found: catboost"**
- Ensure `requirements.txt` is up-to-date: `pip install -r requirements.txt`

### CD Workflow Fails

**"HF_SPACE_ID not set"**
- Verify secrets are configured in GitHub Settings

**"Authentication failed"**
- Check `HF_TOKEN` validity and that it has write permissions

### Gradio App Errors

**"Model file not found"**
- Ensure `models/catboost_models/` and `models/catboost_action_improved/` are populated
- Run training scripts or download from a prior CI run

**Port already in use**
- Change `GRADIO_SERVER_PORT` to an available port (e.g., 7862, 7863)

---

## Performance & Metrics

### Model Performance (Example)

| Target | Baseline Accuracy | Improved Accuracy | Notes |
|--------|-------------------|-------------------|-------|
| Status | ~92% | N/A | Binary classification |
| Category | ~88% | N/A | 5-class, some imbalance |
| Action Taken | ~85% | ~87% | Improved via class weights |

*Metrics saved in `data/metrics/` after evaluation.*

### Inference Speed

- **Per-alert prediction**: ~10–50 ms (depends on hardware)
- **Batch (1000 alerts)**: ~10–15 sec on modern CPU

---

## Contributing

1. Make changes to code or data pipeline
2. Test locally: `python src/04_CatBoost_Full_Train.py`
3. Commit & push to a branch
4. Open a pull request
5. CI runs automatically; merge when tests pass

---

## License

This project is provided as-is for educational and research purposes.

---

## Contact & Support

For issues or questions:
- Open a GitHub issue
- Check `data/metrics/` for detailed evaluation reports
- Review workflow logs in GitHub Actions

**Happy classifying! 🎯**
