from __future__ import annotations
import csv
from pathlib import Path


def _build_reasoning(r: dict) -> str:
    """Compact 1-line justification pulled from the same fields as the txt summary."""
    det = r.get("details", {})
    exp = det.get("experience_breakdown", {})
    cs  = r.get("component_scores", {})

    parts = []

    # Experience
    yrs = exp.get("total_years")
    ai_yrs = exp.get("applied_ai_years_estimate")
    if yrs is not None:
        exp_str = f"{yrs} yrs exp"
        if ai_yrs is not None:
            exp_str += f" ({ai_yrs} AI/ML)"
        parts.append(exp_str)

    # Top 2 matched skills only
    matched_skills = [
        s["matched_via"] for s in det.get("skill_breakdown", [])
        if s.get("matched") and s.get("contribution", 0) > 0
    ]
    if matched_skills:
        parts.append("skilled in " + ", ".join(matched_skills[:2]))

    # Location — take first reason, shortened
    loc_reasons = det.get("location_reasons", [])
    if loc_reasons:
        loc = loc_reasons[0]
        # keep it short: strip trailing parenthetical detail if too long
        if len(loc) > 40:
            loc = loc.split("(")[0].strip()
        parts.append(loc.lower())

    # Education — only mention if suspicious (worth flagging)
    edu_flag = det.get("disqualifier_flags", {}).get("suspicious_education", {})
    if edu_flag.get("triggered"):
        parts.append(f"education concern: {edu_flag.get('reason', 'unverified')}")

    # Penalties
    flags = r.get("triggered_disqualifiers", [])
    if flags:
        parts.append("penalized for " + ", ".join(flags[:2]))

    reasoning = "; ".join(parts)
    # hard cap so it stays compact
    if len(reasoning) > 180:
        reasoning = reasoning[:177].rsplit(" ", 1)[0] + "..."
    return reasoning


def write_csv_summary(results: list[dict], csv_path: Path) -> None:
    """Write ranked results as a single CSV: candidate_id,rank,score,reasoning"""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['final_score']:.3f}",
                _build_reasoning(r),
            ])
    print(f"CSV written to: {csv_path}")
