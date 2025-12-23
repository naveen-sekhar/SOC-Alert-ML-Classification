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
PROJECT_ROOT = APP_DIR.parents[1]  # repo root (project/)

# Ensure project root and src/ are importable
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
	if str(p) not in sys.path:
		sys.path.insert(0, str(p))

# Work from project root so predict_alerts_module finds artifacts via relative paths
os.chdir(PROJECT_ROOT)

from hf.space_app.download_artifacts import ensure_artifacts

# Download artifacts if repo id provided; silently skip if not configured
ensure_artifacts()

import predict_alerts_module as pam  # noqa: E402  # after sys.path adjustments


# ---------------------------------------------------------------------------
# Helpers & styling
# ---------------------------------------------------------------------------
PRED_COLS = ["Status", "Category", "Action Taken"]

CUSTOM_CSS = """
body {background: radial-gradient(circle at 20% 20%, #0f172a 0, #0b1220 25%, #0a0f1b 50%, #090d18 100%); color: #e2e8f0;}
.hero {border: 1px solid #1f2937; background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(14,165,233,0.1)); border-radius: 18px; padding: 18px 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.35);}
.chips {display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px;}
.chip {padding: 6px 12px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.04); font-size: 12px; letter-spacing: .2px;}
.card {border: 1px solid #1f2937; background: rgba(255,255,255,0.02); border-radius: 14px; padding: 14px; box-shadow: 0 8px 30px rgba(0,0,0,0.25);}
.status-pill {display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 10px; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); color: #34d399; font-weight: 600;}
.status-pill.error {background: rgba(248,113,113,0.12); border-color: rgba(248,113,113,0.4); color: #fca5a5;}
.steps {padding-left: 16px; line-height: 1.6; color: #cbd5e1;}
.gr-button {font-weight: 600; letter-spacing: .2px;}
.footer {color: #94a3b8; font-size: 13px; text-align: center; margin-top: 6px;}
.muted {color: #94a3b8; font-size: 13px;}
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
  <h2 style="margin-bottom:4px;">SOC Alert Classifier</h2>
  <div class="muted">Upload alerts CSV → get Status, Category, and Action Taken predictions (CatBoost).</div>
  <div class="chips" style="margin-top:10px;">
	<span class="chip">Input: CSV</span>
	<span class="chip">Outputs: Status / Category / Action</span>
	<span class="chip">Models: CatBoost</span>
	<span class="chip">Text + Structured features</span>
  </div>
</div>
"""

INSTRUCTIONS = """
**How to use**
<ol class="steps">
  <li>Upload a CSV with columns like: Date, Time, Protocol/Port, Source Host, Destination IP, Executive Summary, Severity.</li>
  <li>Click <strong>Run predictions</strong>.</li>
  <li>Preview the first rows and download the full results CSV.</li>
</ol>
"""

FOOTER = "Made with Gradio + CatBoost"


THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="cyan", neutral_hue="slate")


def build_demo():
	data_dir = PROJECT_ROOT / "data"
	sample_csv = data_dir / "test_alerts.csv"
	examples = [[str(sample_csv)]] if sample_csv.exists() else None

	with gr.Blocks(title="SOC Alert Classifier", analytics_enabled=False) as demo:
		gr.HTML(DESCRIPTION)

		with gr.Row():
			with gr.Column(scale=1, min_width=340):
				gr.Markdown(INSTRUCTIONS)
				file_input = gr.File(label="1) Upload alerts CSV", file_types=[".csv"], type="filepath")
				run_btn = gr.Button("2) Run predictions", variant="primary")
				gr.Markdown("<span class='muted'>Tip: use the sample CSV if you just want to try it out.</span>", elem_classes=["muted"])

			with gr.Column(scale=2):
				status_box = gr.HTML(_status_html("Waiting for a file..."))
				preview = gr.Dataframe(label="Preview (first rows with predictions)", interactive=False, wrap=True)
				download = gr.File(label="Download predictions CSV")

		run_btn.click(fn=handle_file_upload, inputs=file_input, outputs=[status_box, preview, download])

		if examples:
			gr.Examples(label="Sample CSV", examples=examples, inputs=file_input)

		with gr.Accordion("Need to pull models from the Hub?", open=False):
			gr.Markdown(
				"Set Space secrets `HF_MODEL_REPO` (and `HF_TOKEN` if private). Files are expected under `catboost_models/` and `catboost_action_improved/`."
			)

		gr.Markdown(f"<div class='footer'>{FOOTER}</div>")

	return demo


demo = build_demo()


if __name__ == "__main__":
	demo.launch(css=CUSTOM_CSS, theme=THEME)
