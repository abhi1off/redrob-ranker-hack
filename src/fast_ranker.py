# Two-stage pipeline for ranking 100k candidates. Runs from CLI or imported as a module.
# Stage 1: hard disqualify + TF-IDF coarse filter
# Stage 2: full 5-component scoring in parallel
#
# Usage:
#   python fast_ranker.py
#   python fast_ranker.py --top_n 50 --shortlist 3000

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loader import load_candidates, fast_disqualify
from coarse import coarse_rank
from vectorizer import CorpusVectorizer
from scorer import full_score_parallel
from printer import write_csv_summary
from overlap import check_overlap
from jd import JD

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
    # full pipeline: load -> disqualify -> tfidf -> coarse rank -> full score
    t0 = time.perf_counter()

    print(f"[Stage 1] Loading candidates from {candidates_path} ...", flush=True)
    all_candidates = load_candidates(candidates_path)
    n_total = len(all_candidates)
    print(f"  Loaded {n_total:,} candidates  ({time.perf_counter()-t0:.1f}s)", flush=True)

    # drop obvious rejects before building the TF-IDF index
    t1 = time.perf_counter()
    survived = [c for c in all_candidates if not fast_disqualify(c)]
    print(
        f"  Hard-disqualified {n_total - len(survived):,} | {len(survived):,} remain"
        f"  ({time.perf_counter()-t1:.1f}s)",
        flush=True,
    )

    # build TF-IDF on survivors only so we don't waste time on rejects
    t2 = time.perf_counter()
    print(f"  Building TF-IDF index on {len(survived):,} survivors ...", flush=True)
    cv = CorpusVectorizer()
    cv.fit(survived)
    jd_text = jd.get("raw_text_for_bm25", "")
    survived_tfidf = [float(x) for x in cv.score_all(jd_text)]
    print(f"  TF-IDF done  ({time.perf_counter()-t2:.1f}s)", flush=True)

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

    summary_path = output_path.with_suffix(".csv")
    write_csv_summary(top_results, summary_path)

    # run overlap check if the disqualified file is present
    disqualified_path = candidates_path.with_name(
        candidates_path.stem + "_disqualified" + candidates_path.suffix
    )
    if disqualified_path.exists():
        print(f"\n[Overlap Check] Comparing ranked results against {disqualified_path.name} ...", flush=True)
        overlaps = check_overlap(output_path, disqualified_path)
        if not overlaps:
            print("  No overlap — pipeline is consistent.", flush=True)
        else:
            print(f"  WARNING: {len(overlaps)} candidate(s) appear in both ranked and disqualified!", flush=True)
    else:
        print(
            f"\n[Overlap Check] Skipped — {disqualified_path.name} not found next to input file.",
            flush=True,
        )

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
    jd_raw_path = BASE_DIR / "jd_raw_text.txt"
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
