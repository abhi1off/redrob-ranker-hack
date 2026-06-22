# redrob-ranker-hack

Candidate ranking pipeline for the Redrob AI hackathon. Picks the top N candidates from a pool of 100k based on a job description — **no LLM calls anywhere**.

---

## Setup

**Create and activate a virtual environment (Windows)**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Install dependencies**

```powershell
pip install -r requirements.txt
```

Place `candidates.jsonl` in the `resources/` directory (git-ignored).

---

## Pipeline overview

There are two independent pipelines. Use **fast_ranker** for 100k-scale ranking.

```
candidates.jsonl
      │
      ├──[Option A: Pre-filter then small-batch score]──────────────────────
      │         │
      │         ▼
      │   disqualifier.py   ── config-driven hard rules ──▶ candidates_qualified.jsonl
      │                                                      candidates_disqualified.jsonl
      │         │
      │         ▼
      │   run_match.py      ── full 5-component score  ──▶ resources/scored_candidates.json
      │
      └──[Option B: Fast ranker — recommended for 100k]──────────────────────
                │
                ▼
          fast_ranker.py
            Stage 1: inline hard disqualify (candidate_loader.py)
            Stage 1: TF-IDF on FULL 100k corpus → coarse rank (coarse_scorer.py)
            Stage 2: full 5-component parallel scoring (stage2_scorer.py)
                │
                ▼
          resources/top100_candidates.json  (structured)
          resources/top100_candidates.txt   (human-readable summary)
```

> **Note:** `fast_ranker.py` reads **directly from `candidates.jsonl`** — it does not need the `disqualifier.py` pre-filtering step. The two pipelines are independent.

---

## Does fast_ranker use disqualifier.py?

**No.** They apply similar hard rules but are separate tools:

| Tool | Input | Rules defined in | Output |
|---|---|---|---|
| `disqualifier.py` | `candidates.jsonl` | `disqualifier_conditions.json` (JSON config) | `candidates_qualified.jsonl`, `candidates_disqualified.jsonl` |
| `candidate_loader.py` (used by fast_ranker) | `candidates.jsonl` | Hardcoded in Python for speed | Filtered list in memory, no file written |

You can run `check_overlap.py` to verify that no candidate that `disqualifier.py` rejected slipped through into the fast_ranker top results.

---

## Step 1 — Build the qualified candidate list (Option A only)

Runs config-driven disqualification rules and splits `candidates.jsonl`.

```bash
cd src
python disqualifier.py
```

Outputs:
- `resources/candidates_qualified.jsonl`
- `resources/candidates_disqualified.jsonl`

---

## Step 2a — Small-batch scoring (`run_match.py`)

Scores `candidates_qualified.jsonl` using the full 5-component pipeline (skill match, experience, location, disqualifier penalties, BM25 text similarity). Suitable for hundreds to low thousands of candidates.

```bash
cd src
python run_match.py
```

Output: `resources/scored_candidates.json`

---

## Step 2b — Fast large-scale ranking (`fast_ranker.py`) ← recommended

Two-stage pipeline designed for **100k candidates in under 5 minutes**. Reads directly from `candidates.jsonl`.

```bash
cd src
python fast_ranker.py
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--candidates` | `resources/candidates.jsonl` | Path to input JSONL |
| `--top_n` | `100` | Number of top candidates to return |
| `--shortlist` | all survivors | Stage-1 pool size forwarded to Stage-2 (leave unset to score all) |
| `--output` | `resources/top<N>_candidates.json` | Output JSON path |
| `--workers` | `min(cpu_count, 8)` | Parallel workers for Stage-2 |

### Examples

```bash
# Default: top 100 from candidates.jsonl
python fast_ranker.py

# Top 50, pass only best 3000 to Stage 2 (faster, lower recall)
python fast_ranker.py --top_n 50 --shortlist 3000

# Custom input/output
python fast_ranker.py --candidates ../resources/candidates.jsonl --output ../resources/results.json

# Limit parallelism on a low-core machine
python fast_ranker.py --workers 2
```

Outputs:
- `resources/top100_candidates.json` — full structured results
- `resources/top100_candidates.txt` — human-readable ranked summary

### How it works

| Stage | Scope | What happens |
|---|---|---|
| 1 — Hard disqualify | all 100k | Field checks: yoe < 4, wrong country, bad title, wrong city — drops ~60% instantly |
| 1 — TF-IDF index | **full 100k corpus** | `TfidfVectorizer` fit on all candidates before filtering so IDF reflects real term rarity |
| 1 — Coarse score | survivors | `0.35 × skill_coverage + 0.25 × tfidf`; survivors sorted and capped at `--shortlist` |
| 2 — Full scoring | shortlist | 5-component scorer (skill + exp + location + disqualifiers + text) across CPU cores |

---

## Overlap check

Verify that no disqualified candidate slipped into the ranked results:

```bash
cd src
python check_overlap.py

# Custom paths
python check_overlap.py --ranked ../resources/top100_candidates.json \
                        --disqualified ../resources/candidates_disqualified.jsonl
```

If there is overlap it means `candidate_loader.fast_disqualify` (used by fast_ranker) is less strict than `disqualifier.py` for some edge case — a useful signal to tighten the inline rules.

---

## Scoring components

The full scorer (`matcher.py`) produces a weighted score from 5 components:

| Component | Weight | What it measures |
|---|---|---|
| Skill match | 0.35 | Weighted JD skill coverage × proficiency × recency × endorsements |
| Experience | 0.30 | Years total, applied AI/ML years proxy, hands-on recency |
| Location / logistics | 0.10 | City match, notice period, work-mode preference |
| Text similarity | 0.25 | Corpus TF-IDF (Stage 1 pre-computed) or single-doc BM25 fallback |

**Disqualifier penalties** are applied multiplicatively on top:

| Severity | Multiplier | Examples |
|---|---|---|
| `hard_reject` | ×0.0 | Pure research career, no production experience |
| `likely_reject` | ×0.35 | Only recent LLM wrapper skills, no pre-LLM ML |
| `soft_reject` | ×0.75 | Suspicious education credentials, low external signals |

---

## Project structure

```
resources/
  candidates.jsonl               ← input (git-ignored)
  candidates_qualified.jsonl     ← output of disqualifier.py (Option A)
  candidates_disqualified.jsonl  ← rejected by disqualifier.py (Option A)
  scored_candidates.json         ← output of run_match.py (Option A)
  top100_candidates.json         ← output of fast_ranker.py (structured)
  top100_candidates.txt          ← output of fast_ranker.py (human summary)

src/
  jd_schema.py                   ← structured JD definition (skills, weights, cities, logistics)
  matcher.py                     ← 5-component scoring engine; used by both pipelines
  text_skill_extractor.py        ← regex-based skill inference from free text fields
  education_validator.py         ← heuristic checks for suspicious education credentials
  corpus_vectorizer.py           ← TF-IDF index builder (scikit-learn); sparse matrix scoring
  candidate_loader.py            ← JSONL loader + fast inline hard-disqualify; used by fast_ranker
  coarse_scorer.py               ← Stage-1: JD skill weight map + fast coverage + TF-IDF combine
  stage2_scorer.py               ← Stage-2: ProcessPoolExecutor parallel full scoring
  result_printer.py              ← console + file summary table with per-candidate reasoning
  fast_ranker.py                 ← thin orchestrator + CLI; wires together the pipeline modules
  disqualifier.py                ← config-driven pre-filter; reads disqualifier_conditions.json
  disqualifier_conditions.json   ← JSON rule config for disqualifier.py
  run_match.py                   ← small-batch scoring script for candidates_qualified.jsonl
  ranker.py                      ← CLI wrapper for disqualifier step
  check_overlap.py               ← QA tool: finds candidates in both ranked results and disqualified

jd_raw_text.txt                  ← raw JD text (used for TF-IDF / BM25)
requirements.txt                 ← scikit-learn, numpy
```

