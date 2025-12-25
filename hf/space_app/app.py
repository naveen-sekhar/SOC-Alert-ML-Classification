"""Gradio Space app for CSV-in / CSV-out alert classification.

Features:
- Upload a CSV with alert rows, run predictions for Status, Category, Action Taken.
- Returns a downloadable CSV with appended prediction columns plus on-screen preview.
- Optionally pulls model artifacts from the Hugging Face Hub via env var `HF_MODEL_REPO`.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent


def _detect_project_root() -> Path:
	"""Locate project root robustly for local runs and HF Spaces."""
	override = os.getenv("PROJECT_ROOT")
	if override:
		return Path(override).resolve()
	start = APP_DIR
	for cand in [start, *start.parents]:
		if (cand / "src").exists() or (cand / "catboost_models").exists() or (cand / "models").exists():
			return cand
	return start


PROJECT_ROOT = _detect_project_root()

# Ensure project root and src/ are importable
for p in (PROJECT_ROOT, PROJECT_ROOT / "src", APP_DIR):
	if str(p) not in sys.path:
		sys.path.insert(0, str(p))

# Work from project root so predict_alerts_module finds artifacts via relative paths
os.chdir(PROJECT_ROOT)

from download_artifacts import ensure_artifacts

# Download artifacts if repo id provided; silently skip if not configured
ensure_artifacts()

import predict_alerts_module as pam  # noqa: E402  # after sys.path adjustments


# ---------------------------------------------------------------------------
# Helpers & styling
# ---------------------------------------------------------------------------
PRED_COLS = ["Status", "Category", "Action Taken"]

CUSTOM_CSS = """
/* Global Styles & Animations */
@keyframes gradient-shift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}

@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
  50% { box-shadow: 0 0 40px rgba(59, 130, 246, 0.6); }
}

@keyframes slide-in {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes flow-animation {
  0% { stroke-dashoffset: 1000; }
  100% { stroke-dashoffset: 0; }
}

@keyframes rotate-3d {
  0% { transform: perspective(1000px) rotateY(0deg); }
  100% { transform: perspective(1000px) rotateY(360deg); }
}

@keyframes bounce-in {
  0% { transform: scale(0.3); opacity: 0; }
  50% { transform: scale(1.05); }
  70% { transform: scale(0.9); }
  100% { transform: scale(1); opacity: 1; }
}

@keyframes data-flow {
  0% { transform: translateX(-100%); opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { transform: translateX(400%); opacity: 0; }
}

/* Body & Background */
body {
  background: linear-gradient(-45deg, #0a0e27, #1a1f3a, #0f1419, #1e293b);
  background-size: 400% 400%;
  animation: gradient-shift 15s ease infinite;
  color: #e2e8f0;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Hero Section */
.hero {
  background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(147,51,234,0.15) 50%, rgba(236,72,153,0.15) 100%);
  border: 2px solid rgba(59, 130, 246, 0.3);
  border-radius: 24px;
  padding: 32px 36px;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 
              0 0 0 1px rgba(255, 255, 255, 0.05) inset;
  backdrop-filter: blur(12px);
  position: relative;
  overflow: hidden;
  animation: slide-in 0.6s ease-out;
}

.hero::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(circle at 30% 50%, rgba(59,130,246,0.1) 0%, transparent 60%);
  animation: float 6s ease-in-out infinite;
  pointer-events: none;
}

.hero h2 {
  font-size: 2.5rem;
  font-weight: 800;
  background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 12px;
  letter-spacing: -0.5px;
  position: relative;
}

/* Feature Chips */
.chips {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 16px;
}

.chip {
  padding: 8px 16px;
  border-radius: 999px;
  background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(147,51,234,0.2));
  border: 1px solid rgba(255, 255, 255, 0.15);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.3px;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.chip:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
  border-color: rgba(96, 165, 250, 0.5);
}

/* Status Pills */
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 12px 20px;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(5,150,105,0.25));
  border: 2px solid rgba(16,185,129,0.5);
  color: #34d399;
  font-weight: 700;
  font-size: 15px;
  animation: pulse-glow 2s ease-in-out infinite;
  box-shadow: 0 8px 24px rgba(16,185,129,0.3);
}

.status-pill::before {
  content: '✓';
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: rgba(16,185,129,0.3);
  border: 2px solid rgba(16,185,129,0.6);
}

.status-pill.error {
  background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(220,38,38,0.25));
  border-color: rgba(239,68,68,0.5);
  color: #fca5a5;
  box-shadow: 0 8px 24px rgba(239,68,68,0.3);
}

.status-pill.error::before {
  content: '⚠';
  background: rgba(239,68,68,0.3);
  border-color: rgba(239,68,68,0.6);
}

/* Instructions */
.steps {
  padding-left: 24px;
  line-height: 2;
  color: #cbd5e1;
  font-size: 15px;
}

.steps li {
  margin: 12px 0;
  padding-left: 8px;
  position: relative;
}

.steps li::marker {
  color: #60a5fa;
  font-weight: bold;
}

.steps strong {
  color: #f472b6;
  font-weight: 700;
}

/* Buttons */
.gr-button {
  font-weight: 700;
  letter-spacing: 0.5px;
  border-radius: 12px;
  padding: 14px 28px;
  transition: all 0.3s ease;
  font-size: 15px;
  text-transform: uppercase;
  box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
  background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
  border: none;
}

.gr-button:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 32px rgba(59, 130, 246, 0.5);
}

.gr-button:active {
  transform: translateY(-1px);
}

/* File Upload */
.gr-file {
  border: 2px dashed rgba(96, 165, 250, 0.5) !important;
  border-radius: 16px !important;
  background: rgba(15, 23, 42, 0.6) !important;
  transition: all 0.3s ease;
}

.gr-file:hover {
  border-color: rgba(96, 165, 250, 0.8) !important;
  background: rgba(30, 41, 59, 0.8) !important;
}

/* Dataframe */
.gr-dataframe {
  border-radius: 16px !important;
  overflow: hidden;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4) !important;
  border: 1px solid rgba(59, 130, 246, 0.3) !important;
}

/* Accordion */
.gr-accordion {
  background: rgba(15, 23, 42, 0.6) !important;
  border: 1px solid rgba(59, 130, 246, 0.3) !important;
  border-radius: 12px !important;
  margin-top: 24px;
}

/* Footer */
.footer {
  color: #94a3b8;
  font-size: 14px;
  text-align: center;
  margin-top: 32px;
  padding: 20px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  font-weight: 500;
  letter-spacing: 0.5px;
}

.muted {
  color: #94a3b8;
  font-size: 14px;
  font-style: italic;
}

/* Cards & Containers */
.gr-box {
  background: rgba(15, 23, 42, 0.8) !important;
  border: 1px solid rgba(59, 130, 246, 0.2) !important;
  border-radius: 16px !important;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
}

/* Column spacing */
.gr-column {
  padding: 16px;
}

/* 3D Workflow Visualization */
.workflow-container {
  perspective: 1500px;
  padding: 40px 20px;
  margin: 30px 0;
  background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.95) 100%);
  border-radius: 20px;
  border: 2px solid rgba(59,130,246,0.3);
  box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  overflow: hidden;
  position: relative;
}

.workflow-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(circle at 50% 50%, rgba(59,130,246,0.1) 0%, transparent 70%);
  pointer-events: none;
}

.workflow-title {
  text-align: center;
  font-size: 28px;
  font-weight: 800;
  margin-bottom: 40px;
  background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  position: relative;
  z-index: 1;
}

.pipeline-flow {
  display: flex;
  align-items: center;
  justify-content: space-around;
  flex-wrap: wrap;
  gap: 20px;
  position: relative;
  z-index: 1;
}

.pipeline-stage {
  flex: 1;
  min-width: 200px;
  max-width: 250px;
  padding: 24px 20px;
  background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(147,51,234,0.2));
  border: 2px solid rgba(96,165,250,0.4);
  border-radius: 16px;
  text-align: center;
  transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  cursor: pointer;
  position: relative;
  transform-style: preserve-3d;
  animation: bounce-in 0.6s ease-out backwards;
}

.pipeline-stage:nth-child(1) { animation-delay: 0.1s; }
.pipeline-stage:nth-child(2) { animation-delay: 0.2s; }
.pipeline-stage:nth-child(3) { animation-delay: 0.3s; }
.pipeline-stage:nth-child(4) { animation-delay: 0.4s; }
.pipeline-stage:nth-child(5) { animation-delay: 0.5s; }

.pipeline-stage:hover {
  transform: translateY(-15px) scale(1.05);
  box-shadow: 0 20px 50px rgba(59,130,246,0.5);
  border-color: rgba(96,165,250,0.8);
  background: linear-gradient(135deg, rgba(59,130,246,0.4), rgba(147,51,234,0.4));
}

.pipeline-stage::before {
  content: '';
  position: absolute;
  top: -2px;
  left: -2px;
  right: -2px;
  bottom: -2px;
  background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
  border-radius: 16px;
  opacity: 0;
  transition: opacity 0.3s ease;
  z-index: -1;
}

.pipeline-stage:hover::before {
  opacity: 0.3;
}

.stage-icon {
  font-size: 48px;
  margin-bottom: 12px;
  display: block;
  filter: drop-shadow(0 4px 8px rgba(59,130,246,0.5));
  animation: float 3s ease-in-out infinite;
}

.stage-title {
  font-size: 16px;
  font-weight: 700;
  color: #e2e8f0;
  margin-bottom: 8px;
  letter-spacing: 0.5px;
}

.stage-desc {
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.5;
}

.flow-arrow {
  font-size: 36px;
  color: rgba(96,165,250,0.6);
  animation: pulse-glow 2s ease-in-out infinite;
  margin: 0 10px;
  filter: drop-shadow(0 0 10px rgba(96,165,250,0.8));
}

.data-particles {
  position: absolute;
  top: 50%;
  left: 0;
  width: 10px;
  height: 10px;
  background: linear-gradient(135deg, #60a5fa, #a78bfa);
  border-radius: 50%;
  box-shadow: 0 0 20px rgba(96,165,250,0.8);
  animation: data-flow 4s linear infinite;
  z-index: 0;
}

.data-particles:nth-child(2) { animation-delay: 0.5s; top: 45%; }
.data-particles:nth-child(3) { animation-delay: 1s; top: 55%; }
.data-particles:nth-child(4) { animation-delay: 1.5s; top: 50%; }
.data-particles:nth-child(5) { animation-delay: 2s; top: 48%; }

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin-top: 40px;
  position: relative;
  z-index: 1;
}

.metric-card {
  background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(147,51,234,0.15));
  border: 1px solid rgba(96,165,250,0.3);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
  transition: all 0.3s ease;
  animation: slide-in 0.6s ease-out backwards;
}

.metric-card:nth-child(1) { animation-delay: 0.6s; }
.metric-card:nth-child(2) { animation-delay: 0.7s; }
.metric-card:nth-child(3) { animation-delay: 0.8s; }

.metric-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 12px 30px rgba(59,130,246,0.3);
  border-color: rgba(96,165,250,0.6);
}

.metric-value {
  font-size: 32px;
  font-weight: 800;
  background: linear-gradient(135deg, #60a5fa, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 8px;
}

.metric-label {
  font-size: 13px;
  color: #94a3b8;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
}
"""


def _status_html(msg: str, ok: bool = True) -> str:
	css_class = "status-pill" + ("" if ok else " error")
	return f"<div class='{css_class}'>{msg}</div>"


def run_predictions(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
	"""Run model predictions and return augmented DataFrame + message."""
	records = df.to_dict(orient="records")
	preds = pam.predict_alerts(records)
	out_df = df.copy()
	for col in PRED_COLS:
		out_df[col] = preds[col]
	return out_df, _status_html("Predictions complete. Download or preview below.")


def handle_file_upload(file_obj) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
	"""Gradio handler: read CSV, predict, return message, preview, and file path."""
	if file_obj is None:
		return _status_html("Please upload a CSV file.", ok=False), None, None

	path = getattr(file_obj, "name", file_obj)

	try:
		df = pd.read_csv(path)
	except Exception as e:  # pragma: no cover - UI convenience
		return _status_html(f"Failed to read CSV: {e}", ok=False), None, None

	if df.empty:
		return _status_html("CSV is empty; nothing to predict.", ok=False), None, None

	try:
		out_df, msg = run_predictions(df)
	except Exception as e:  # pragma: no cover - surface inference errors
		return _status_html(f"Prediction failed: {e}", ok=False), None, None

	with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
		out_df.to_csv(tmp.name, index=False)
		download_path = tmp.name

	preview_rows = min(50, len(out_df))
	return msg, out_df.head(preview_rows), download_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
DESCRIPTION = """
<div class="hero">
  <h2>🛡️ SOC Alert Classifier</h2>
  <div class="muted" style="font-size: 16px; margin-bottom: 16px; color: #cbd5e1;">
    AI-powered security alert classification using advanced CatBoost ML models
  </div>
  <div class="chips">
    <span class="chip">📊 CSV Input</span>
    <span class="chip">🎯 Multi-Class Prediction</span>
    <span class="chip">⚡ CatBoost Models</span>
    <span class="chip">🔍 Text + Structured Analysis</span>
  </div>
</div>
"""

INSTRUCTIONS = """
<div style="background: rgba(59,130,246,0.08); border-left: 4px solid #3b82f6; padding: 20px; border-radius: 12px; margin: 16px 0;">
  <h3 style="color: #60a5fa; margin-top: 0; font-size: 18px; font-weight: 700;">📋 How to Use</h3>
  <ol class="steps">
    <li>Upload a CSV containing your SIEM alert data with columns like: <strong>Date</strong>, <strong>Time</strong>, <strong>Protocol/Port</strong>, <strong>Source Host</strong>, <strong>Destination IP</strong>, <strong>Executive Summary</strong>, <strong>Severity</strong></li>
    <li>Click the <strong>Run Predictions</strong> button to classify alerts</li>
    <li>View the predictions in the preview table and download the complete results</li>
  </ol>
  <div style="margin-top: 16px; padding: 12px; background: rgba(147,51,234,0.1); border-radius: 8px; border: 1px solid rgba(167,139,250,0.3);">
    💡 <strong>Tip:</strong> Try the sample CSV to see how it works instantly!
  </div>
</div>
"""

FOOTER = """
<div style="text-align: center; padding: 24px; border-top: 2px solid rgba(59,130,246,0.3); margin-top: 40px;">
  <div style="font-size: 16px; font-weight: 600; color: #94a3b8; margin-bottom: 8px;">
    Powered by <span style="color: #60a5fa;">Gradio</span> + <span style="color: #a78bfa;">CatBoost</span>
  </div>
  <div style="font-size: 13px; color: #64748b;">
    🚀 Automated CI/CD via GitHub Actions • 🤗 Deployed on Hugging Face Spaces
  </div>
</div>
"""

WORKFLOW_3D = """
<div class="workflow-container">
  <div class="data-particles"></div>
  <div class="data-particles"></div>
  <div class="data-particles"></div>
  <div class="data-particles"></div>
  <div class="data-particles"></div>
  
  <h2 class="workflow-title">🔄 ML Pipeline Workflow</h2>
  
  <div class="pipeline-flow">
    <div class="pipeline-stage">
      <span class="stage-icon">📊</span>
      <div class="stage-title">Data Ingestion</div>
      <div class="stage-desc">SIEM alerts with date, time, IPs, protocols, and executive summaries</div>
    </div>
    
    <span class="flow-arrow">→</span>
    
    <div class="pipeline-stage">
      <span class="stage-icon">🧹</span>
      <div class="stage-title">Preprocessing</div>
      <div class="stage-desc">DateTime parsing, IP features, protocol splitting, text normalization</div>
    </div>
    
    <span class="flow-arrow">→</span>
    
    <div class="pipeline-stage">
      <span class="stage-icon">⚙️</span>
      <div class="stage-title">Feature Engineering</div>
      <div class="stage-desc">TF-IDF+SVD text embeddings, keyword extraction, temporal features</div>
    </div>
    
    <span class="flow-arrow">→</span>
    
    <div class="pipeline-stage">
      <span class="stage-icon">🤖</span>
      <div class="stage-title">CatBoost Models</div>
      <div class="stage-desc">3 classifiers: Status, Category, Action (68 features total)</div>
    </div>
    
    <span class="flow-arrow">→</span>
    
    <div class="pipeline-stage">
      <span class="stage-icon">🎯</span>
      <div class="stage-title">Predictions</div>
      <div class="stage-desc">Malicious/Legitimate, 5 categories, 5 action types</div>
    </div>
  </div>
  
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-value">68</div>
      <div class="metric-label">Features</div>
    </div>
    
    <div class="metric-card">
      <div class="metric-value">~90%</div>
      <div class="metric-label">Accuracy</div>
    </div>
    
    <div class="metric-card">
      <div class="metric-value">&lt;50ms</div>
      <div class="metric-label">Inference</div>
    </div>
  </div>
</div>
"""


THEME = gr.themes.Soft(
	primary_hue="blue",
	secondary_hue="purple",
	neutral_hue="slate",
	font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
).set(
	body_background_fill="*neutral_950",
	body_background_fill_dark="*neutral_950",
	button_primary_background_fill="linear-gradient(135deg, *primary_500, *secondary_500)",
	button_primary_background_fill_hover="linear-gradient(135deg, *primary_600, *secondary_600)",
	button_primary_text_color="white",
	block_background_fill="*neutral_900",
	block_border_color="*primary_500",
	block_border_width="2px",
	input_background_fill="*neutral_800",
)


def build_demo():
	data_dir = PROJECT_ROOT / "data"
	sample_csv = data_dir / "test_alerts.csv"
	examples = [[str(sample_csv)]] if sample_csv.exists() else None

	with gr.Blocks(title="🛡️ SOC Alert Classifier | ML-Powered Security") as demo:
		gr.HTML(DESCRIPTION)
		
		# 3D Workflow Visualization
		gr.HTML(WORKFLOW_3D)

		with gr.Row(equal_height=False):
			with gr.Column(scale=1, min_width=380):
				gr.HTML(INSTRUCTIONS)
				
				with gr.Group():
					file_input = gr.File(
						label="📁 Upload Alert CSV",
						file_types=[".csv"],
						type="filepath",
						elem_classes=["file-upload"]
					)
					
					run_btn = gr.Button(
						"🚀 Run Predictions",
						variant="primary",
						size="lg",
						elem_classes=["primary-button"]
					)
				
				if examples:
					gr.Examples(
						label="💼 Sample Dataset",
						examples=examples,
						inputs=file_input,
						examples_per_page=3
					)
				
				gr.Markdown(
					"<div style='margin-top: 16px; padding: 12px; background: rgba(59,130,246,0.1); border-radius: 8px; border: 1px solid rgba(59,130,246,0.3);'>"
					"<span style='font-size: 13px; color: #94a3b8;'>💡 <strong>Quick Start:</strong> Use the sample CSV above to test the classifier instantly</span>"
					"</div>"
				)

			with gr.Column(scale=2):
				status_box = gr.HTML(
					_status_html("⏳ Ready to classify alerts. Upload a CSV to begin..."),
					elem_classes=["status-box"]
				)
				
				with gr.Group():
					preview = gr.Dataframe(
						label="📊 Prediction Results Preview (First 50 rows)",
						interactive=False,
						wrap=True,
						elem_classes=["results-table"]
					)
				
				download = gr.File(
					label="⬇️ Download Complete Results",
					type="filepath",
					elem_classes=["download-file"]
				)

		run_btn.click(
			fn=handle_file_upload,
			inputs=file_input,
			outputs=[status_box, preview, download]
		)

		with gr.Accordion("⚙️ Advanced: Pull Models from Hugging Face Hub", open=False):
			gr.Markdown(
				"""
				<div style="padding: 16px; background: rgba(15,23,42,0.6); border-radius: 8px;">
				If deploying this Space without local models, configure the following secrets:
				
				- **`HF_MODEL_REPO`**: Repository ID containing model artifacts (e.g., `username/soc-models`)
				- **`HF_TOKEN`**: Access token for private repositories (optional)
				
				Models are expected under `catboost_models/` and `catboost_action_improved/` directories.
				</div>
				"""
			)

		gr.HTML(FOOTER)

	return demo


demo = build_demo()


if __name__ == "__main__":
	# Disable SSR to avoid asyncio event loop close warnings on shutdown in some environments
	# Bind explicitly for local runs with a stable host/port (overridable via env)
	_server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
	_server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
	demo.launch(
		server_name=_server_name,
		server_port=_server_port,
		theme=THEME,
		css=CUSTOM_CSS,
		ssr_mode=False
	)
