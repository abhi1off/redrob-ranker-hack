"""
Gradio front-end for the RedRob candidate ranker.

Upload a candidates.jsonl file, hit Run, and download the ranked CSV.
"""

import sys
import tempfile
import shutil
from pathlib import Path

import gradio as gr
import pandas as pd

# Make sure src/ is importable whether app.py is run from the repo root
# (local dev) or from the Space runner (where CWD may differ).
_APP_DIR = Path(__file__).resolve().parent
_SRC_DIR = _APP_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fast_ranker import run_pipeline  # noqa: E402  (after sys.path fix)
from jd import JD  # noqa: E402

_JD_RAW_PATH = _APP_DIR / "jd_raw_text.txt"


def _load_jd_raw() -> str:
    if _JD_RAW_PATH.exists():
        return _JD_RAW_PATH.read_text(encoding="utf-8")
    return ""


def rank_candidates(
    jsonl_file: str | None,
    top_n: int,
) -> tuple[pd.DataFrame | None, str | None, str]:
    """
    Core callback wired to the Gradio interface.

    Returns
    -------
    dataframe  : preview table shown in the UI
    csv_path   : path handed to gr.File for download
    status_msg : plain-text status / error message
    """
    if jsonl_file is None:
        return None, None, "Please upload a candidates.jsonl file first."

    candidates_path = Path(jsonl_file)
    if not candidates_path.exists():
        return None, None, f"Uploaded file not found at: {candidates_path}"

    # Inject raw JD text for TF-IDF / BM25
    jd = dict(JD)
    jd["raw_text_for_bm25"] = _load_jd_raw()

    # Write outputs to a temp directory so concurrent runs don't collide
    tmp_dir = Path(tempfile.mkdtemp(prefix="redrob_"))
    try:
        output_json = tmp_dir / f"top{top_n}_candidates.json"

        results = run_pipeline(
            candidates_path=candidates_path,
            jd=jd,
            top_n=top_n,
            output_path=output_json,
        )
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, None, f"Pipeline error: {exc}"

    csv_path = output_json.with_suffix(".csv")
    if not csv_path.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, None, "Pipeline ran but CSV was not created."

    df = pd.read_csv(csv_path)
    status = (
        f"Done. Ranked {len(results):,} candidates from your file. "
        f"Showing top {len(df)}."
    )
    return df, str(csv_path), status


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

_JD_PREVIEW = (
    "**Senior AI Engineer — Founding Team · Redrob AI (Series A)**\n\n"
    "Pune / Noida, India · Hybrid · 5–9 yrs experience\n\n"
    "Ranks candidates by: skill match (0.35) · experience (0.30) · "
    "text similarity (0.25) · location (0.10)\n\n"
    "Hard disqualifiers: <4 yrs exp · outside India without relocation · "
    "Manager/Executive title"
)

with gr.Blocks(title="RedRob Candidate Ranker", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#RedRob Candidate Ranker")
    gr.Markdown(_JD_PREVIEW)

    with gr.Row():
        with gr.Column(scale=1):
            jsonl_input = gr.File(
                label="Upload candidates.jsonl",
                file_types=[".jsonl"],
                type="filepath",
            )
            top_n_slider = gr.Slider(
                minimum=10,
                maximum=500,
                value=100,
                step=10,
                label="Top N to return",
            )
            run_btn = gr.Button("Run pipeline", variant="primary")

        with gr.Column(scale=2):
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
            )
            results_table = gr.Dataframe(
                label="Top candidates (preview)",
                interactive=False,
                wrap=True,
            )
            csv_download = gr.File(
                label="Download full CSV",
                interactive=False,
            )

    run_btn.click(
        fn=rank_candidates,
        inputs=[jsonl_input, top_n_slider],
        outputs=[results_table, csv_download, status_box],
    )

if __name__ == "__main__":
    demo.launch()
