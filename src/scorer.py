# Stage-2: run full 5-component scoring on the shortlist in parallel.
# Needs to be a top-level module (not nested) so the worker function
# is picklable on Windows (spawn-based multiprocessing).

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# make src/ importable in worker processes — Windows spawn doesn't inherit sys.path
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from matcher import build_skill_alias_map, score_candidate


# runs once per worker process — cap threads so N workers don't each spin up N threads
def _init_worker() -> None:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"


# worker fn must be at module level to be picklable
def _score_one(args: tuple) -> dict:
    """Score a single candidate with the full pipeline."""
    candidate, jd, tfidf_score, alias_map = args
    return score_candidate(
        candidate,
        jd,
        alias_map=alias_map,
        precomputed_text_score=tfidf_score,
    )


def full_score_parallel(
    shortlist: list[tuple[float, dict, float]],
    jd: dict,
    top_n: int,
    n_workers: int | None = None,
) -> list[dict]:
    """Run full scoring on the shortlist in parallel and return top_n results sorted by score."""
    alias_map = build_skill_alias_map(jd)
    tasks = [
        (candidate, jd, tfidf_score, alias_map)
        for _, candidate, tfidf_score in shortlist
    ]

    workers = n_workers or min(os.cpu_count() or 2, 4)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_score_one, task) for task in tasks}
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: (-r["final_score"], int(r["candidate_id"].split("_")[1])))
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return results[:top_n]
