"""
result_printer.py — Console output for ranked candidate results.

Responsibility:
  - Print a human-readable ranked summary table with per-candidate reasoning.
  - No scoring logic, no file I/O — display only.
"""

from __future__ import annotations
import sys
from pathlib import Path


def print_results(results: list[dict], summary_path: Path | None = None) -> None:
    """
    Print a ranked summary table with score breakdown and reasoning
    for each candidate in results (pre-sorted by final_score descending).

    If summary_path is provided, the same output is also written to that file.
    """
    out_fh = open(summary_path, "w", encoding="utf-8") if summary_path else None

    def _print(*args, **kwargs):
        kwargs.pop("flush", None)
        print(*args, **kwargs)
        if out_fh:
            print(*args, file=out_fh, **kwargs)
    _print(f"\n{'='*80}")
    _print(f"  TOP {len(results)} CANDIDATES")
    _print(f"{'='*80}")

    for r in results:
        cs    = r.get("component_scores", {})
        det   = r.get("details", {})
        exp   = det.get("experience_breakdown", {})
        flags = ", ".join(r.get("triggered_disqualifiers", [])) or "none"

        # Top matched skills (contribution > 0), capped at 5 for readability
        matched_skills = [
            f"{s['matched_via']} ({s['proficiency']}, {s['duration_months']}mo)"
            for s in det.get("skill_breakdown", [])
            if s.get("matched") and s.get("contribution", 0) > 0
        ]
        skills_str = ", ".join(matched_skills[:5]) or "none"
        if len(matched_skills) > 5:
            skills_str += f" +{len(matched_skills)-5} more"

        loc_reasons = " | ".join(det.get("location_reasons", [])) or "—"

        # Education validation — pulled from disqualifier_flags
        edu_flag = det.get("disqualifier_flags", {}).get("suspicious_education", {})
        if edu_flag.get("triggered"):
            edu_str = f"SUSPICIOUS — {edu_flag.get('reason', 'unknown issue')}"
        else:
            edu_str = edu_flag.get("reason", "—")

        _print(
            f"\n#{r['rank']:>3}  {r['candidate_id']}  →  score: {r['final_score']:.4f}"
            f"  (penalty: ×{r['penalty_multiplier']:.2f})",
        )
        _print(
            f"     Components — skill: {cs.get('skill_score', 0):.2f}  "
            f"exp: {cs.get('experience_score', 0):.2f}  "
            f"loc: {cs.get('location_score', 0):.2f}  "
            f"text: {cs.get('text_similarity_score', 0):.2f}",
        )
        _print(
            f"     Experience — {exp.get('total_years', '-')} yrs total | "
            f"{exp.get('applied_ai_years_estimate', '-')} applied AI/ML yrs | "
            f"hands-on score: {exp.get('hands_on_score', '-')}",
        )
        _print(f"     Skills matched — {skills_str}")
        _print(f"     Location       — {loc_reasons}")
        _print(f"     Education      — {edu_str}")
        if flags != "none":
            _print(f"     Penalties      — {flags}")

    _print(f"\n{'='*80}\n")
    if out_fh:
        out_fh.close()
        print(f"Summary written to: {summary_path}", flush=True)
