"""
check_overlap.py — Verify that no candidate in the final ranked results was
also hard-disqualified by disqualifier.py.

This is a QA sanity-check tool. If overlap exists it means fast_ranker's
inline rules (candidate_loader.fast_disqualify) are less strict than
disqualifier.py's config-driven rules and a candidate slipped through.

Usage:
    cd src
    python check_overlap.py
    python check_overlap.py --ranked ../resources/top100_candidates.json
    python check_overlap.py --ranked ../resources/top100_candidates.json \\
                            --disqualified ../resources/candidates_disqualified.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent


def load_disqualified_ids(jsonl_path: Path) -> set[str]:
    """Stream-load candidates_disqualified.jsonl and return the set of candidate_ids."""
    ids: set[str] = set()
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cid = c.get("candidate_id")
                if cid:
                    ids.add(cid)
            except json.JSONDecodeError:
                continue
    return ids


def check_overlap(
    ranked_path: Path,
    disqualified_path: Path,
) -> list[dict]:
    """
    Return list of candidates that appear in both ranked results and disqualified.

    Each entry has: candidate_id, rank, final_score, penalty_multiplier,
    triggered_disqualifiers (from fast_ranker's scoring).
    """
    with open(ranked_path, encoding="utf-8") as fh:
        ranked: list[dict] = json.load(fh)

    print(f"  Loading disqualified IDs from {disqualified_path} ...", flush=True)
    disqualified_ids = load_disqualified_ids(disqualified_path)
    print(f"  {len(disqualified_ids):,} disqualified candidates loaded.", flush=True)

    overlaps = []
    for result in ranked:
        cid = result.get("candidate_id")
        if cid in disqualified_ids:
            overlaps.append({
                "candidate_id": cid,
                "rank": result.get("rank"),
                "final_score": result.get("final_score"),
                "penalty_multiplier": result.get("penalty_multiplier"),
                "triggered_disqualifiers": result.get("triggered_disqualifiers", []),
            })
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check overlap between ranked results and candidates_disqualified.jsonl."
    )
    parser.add_argument(
        "--ranked",
        type=Path,
        default=BASE_DIR / "resources" / "top100_candidates.json",
        help="Path to ranked results JSON (default: resources/top100_candidates.json)",
    )
    parser.add_argument(
        "--disqualified",
        type=Path,
        default=BASE_DIR / "resources" / "candidates_disqualified.jsonl",
        help="Path to candidates_disqualified.jsonl (default: resources/candidates_disqualified.jsonl)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  OVERLAP CHECK — ranked results vs. disqualified candidates")
    print("=" * 70)
    print(f"  Ranked:       {args.ranked}")
    print(f"  Disqualified: {args.disqualified}\n")

    overlaps = check_overlap(args.ranked, args.disqualified)

    if not overlaps:
        print(
            "\n✓  No overlap — all ranked candidates are absent from the "
            "disqualified list. Pipeline is consistent.\n"
        )
        return

    print(f"\n⚠  OVERLAP DETECTED: {len(overlaps)} candidate(s) appear in BOTH files!\n")
    print(f"  This means fast_ranker's inline rules let these through, but")
    print(f"  disqualifier.py's config-driven rules would have rejected them.\n")
    print(f"{'Rank':>6}  {'Candidate ID':<20}  {'Score':>8}  {'Penalty':>8}  Flags")
    print("-" * 80)
    for o in sorted(overlaps, key=lambda x: x["rank"] or 999):
        flags = ", ".join(o["triggered_disqualifiers"]) or "none"
        print(
            f"{o['rank']:>6}  {o['candidate_id']:<20}  "
            f"{o['final_score']:>8.4f}  {o['penalty_multiplier']:>8.4f}  {flags}"
        )
    print()


if __name__ == "__main__":
    main()
