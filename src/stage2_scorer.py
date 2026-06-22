"""
stage2_scorer.py — Stage-2 full detailed scoring with parallelism.

Responsibility:
  - Define the picklable worker function for ProcessPoolExecutor.
  - Run full score_candidate() on a shortlist of candidates in parallel.
  - Return results sorted by final_score with 1-based rank attached.

Must be a top-level module (not nested) so worker functions are picklable
on Windows, where multiprocessing uses spawn (not fork).
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure src/ is importable in worker processes (Windows spawn creates a fresh
# interpreter that inherits sys.path only if explicitly set).
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from matcher import build_skill_alias_map, score_candidate


# ---------------------------------------------------------------------------
# Worker (module-level so it is picklable)
# ---------------------------------------------------------------------------

def _score_one(args: tuple) -> dict:
    """
    Score a single candidate with the full 5-component pipeline.

    args = (candidate_dict, jd_dict, precomputed_tfidf_score, alias_map)
    """
    candidate, jd, tfidf_score, alias_map = args
    return score_candidate(
        candidate,
        jd,
        alias_map=alias_map,
        precomputed_text_score=tfidf_score,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def full_score_parallel(
    shortlist: list[tuple[float, dict, float]],
    jd: dict,
    top_n: int,
    n_workers: int | None = None,
) -> list[dict]:
    """
    Run full scoring on the shortlist in parallel and return top_n results.

    Args:
        shortlist  : list of (coarse_score, candidate_dict, tfidf_score) from coarse_rank()
        jd         : JD dict with raw_text_for_bm25 populated
        top_n      : number of top results to return
        n_workers  : parallel workers (default: min(cpu_count, 8))

    Returns:
        list of result dicts sorted by final_score descending, with `rank` field added.
    """
    alias_map = build_skill_alias_map(jd)
    tasks = [
        (candidate, jd, tfidf_score, alias_map)
        for _, candidate, tfidf_score in shortlist
    ]

    workers = n_workers or min(os.cpu_count() or 4, 8)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_score_one, task) for task in tasks}
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: r["final_score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return results[:top_n]
