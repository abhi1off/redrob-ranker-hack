"""
fast_ranker.py -- Orchestrator + CLI for the two-stage 100k candidate ranker.

Responsibility:
  - Wire together the individual pipeline modules.
  - Expose run_pipeline() for programmatic use.
  - Provide a CLI via __main__.

Pipeline modules (one responsibility each):
  candidate_loader  -- load JSONL + hard disqualification
  coarse_scorer     -- Stage-1 skill coverage + TF-IDF ranking
  stage2_scorer     -- Stage-2 full scoring in parallel
  result_printer    -- console summary table

Usage:
    cd src
    python fast_ranker.py
    python fast_ranker.py --top_n 50 --shortlist 3000
    python fast_ranker.py --candidates ../resources/candidates.jsonl --top_n 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from candidate_loader import load_candidates, fast_disqualify
from coarse_scorer import coarse_rank
from corpus_vectorizer import CorpusVectorizer
from stage2_scorer import full_score_parallel
from result_printer import print_results
from jd_schema import JD

SHORTLIST_SIZE = None
DEFAULT_TOP_N  = 100


def run_pipeline(
    candidates_path: Path,
    jd: dict,
    top_n: int = DEFAULT_TOP_N,
    shortlist_size: int | None = SHORTLIST_SIZE,
    output_path: Path | None = None,
    n_workers: int | None = None,
) -> list[dict]:
    """Full two-stage ranking pipeline."""
    t0 = time.perf_counter()

    print(f"[Stage 1] Loading candidates from {candidates_path} ...", flush=True)
    all_candidates = load_candidates(candidates_path)
    n_total = len(all_candidates)
    print(f"  Loaded {n_total:,} candidates  ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Build TF-IDF on the FULL corpus before filtering so IDF is computed
    # across all 100k candidates (not just the ~23k survivors), giving
    # meaningful term rarity weights that distinguish candidates.
    t1 = time.perf_counter()
    print(f"  Building TF-IDF index on full corpus of {n_total:,} ...", flush=True)
    cv = CorpusVectorizer()
    cv.fit(all_candidates)
    jd_text = jd.get("raw_text_for_bm25", "")
    tfidf_all = cv.score_all(jd_text)   # array aligned with all_candidates
    print(f"  TF-IDF done  ({time.perf_counter()-t1:.1f}s)", flush=True)

    # Hard disqualify — preserve each survivor's tfidf score by original index
    t2 = time.perf_counter()
    survived_pairs = [(i, c) for i, c in enumerate(all_candidates) if not fast_disqualify(c)]
    survived       = [c for _, c in survived_pairs]
    survived_tfidf = [float(tfidf_all[i]) for i, _ in survived_pairs]
    print(
        f"  Hard-disqualified {n_total - len(survived):,} | {len(survived):,} remain"
        f"  ({time.perf_counter()-t2:.1f}s)",
        flush=True,
    )

    t3 = time.perf_counter()
    print(f"  Coarse scoring {len(survived):,} candidates ...", flush=True)
    shortlist = coarse_rank(survived, jd, shortlist_size=shortlist_size, precomputed_tfidf=survived_tfidf)
    print(
        f"  Coarse done; shortlist = {len(shortlist):,}"
        f"  ({time.perf_counter()-t3:.1f}s)",
        flush=True,
    )

    t4 = time.perf_counter()
    print(f"[Stage 2] Full scoring of {len(shortlist):,} candidates ...", flush=True)
    top_results = full_score_parallel(shortlist, jd, top_n=top_n, n_workers=n_workers)
    print(
        f"  Full scoring done  ({time.perf_counter()-t4:.1f}s)\n"
        f"[Done] Total wall-clock: {time.perf_counter()-t0:.1f}s"
        f"  |  Returning top {len(top_results)}",
        flush=True,
    )

    if output_path is None:
        output_path = BASE_DIR / "resources" / f"top{top_n}_candidates.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(top_results, fh, indent=2)
    print(f"Results saved to: {output_path}", flush=True)

    summary_path = output_path.with_suffix(".txt")
    print_results(top_results, summary_path=summary_path)
    return top_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast two-stage candidate ranker (no LLM).")
    parser.add_argument("--candidates", type=Path, default=BASE_DIR / "resources" / "candidates.jsonl")
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--shortlist", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    jd_raw_path = BASE_DIR / "jd_raw_text.txt"  # BASE_DIR = project root (set at module level)
    print(f"Loading JD raw text from: {jd_raw_path}")
    with open(jd_raw_path, encoding="utf-8") as fh:
        JD["raw_text_for_bm25"] = fh.read()
    run_pipeline(
        candidates_path=args.candidates,
        jd=JD,
        top_n=args.top_n,
        shortlist_size=args.shortlist or None,
        output_path=args.output,
        n_workers=args.workers,
    )
