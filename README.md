# redrob-ranker-hack

Ranks candidates from a 100k pool against a job description. No LLM calls — pure scoring logic.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Put `candidates.jsonl` in `resources/` (git-ignored).

---

## How to run

Two-stage pipeline, reads directly from `resources/candidates.jsonl`:

```bash
cd src
python fast_ranker.py
```

Common flags:
```bash
python fast_ranker.py --top_n 50                                 # return top 50
python fast_ranker.py --candidates path/to/candidates.jsonl      # custom input path
```

Outputs:
- `resources/top100_candidates.json` — full results
- `resources/top100_candidates.csv` — readable summary

**How it works:**
1. Drop obvious rejects (wrong city, bad title, <4 yrs exp) — cuts ~60% immediately
2. Build TF-IDF on remaining candidates, coarse score with `0.35 × skill + 0.25 × tfidf`
3. Run full 5-component scoring on the shortlist in parallel

---

## Scoring breakdown

| Component | Weight | Notes |
|---|---|---|
| Skill match | 0.35 | JD skill coverage × proficiency × recency × endorsements |
| Experience | 0.30 | Total years, applied AI/ML estimate, hands-on recency |
| Location | 0.10 | City match, notice period, work mode |
| Text similarity | 0.25 | Corpus TF-IDF (fast_ranker) or BM25 fallback (match.py) |

Disqualifier penalties multiply the raw score:

| Severity | Multiplier |
|---|---|
| `hard_reject` | ×0.0 |
| `likely_reject` | ×0.35 |
| `soft_reject` | ×0.75 |

---

## File overview

```
src/
  fast_ranker.py      main pipeline orchestrator + CLI
  matcher.py          5-component scoring engine
  loader.py           JSONL loader + fast inline disqualify
  coarse.py           stage-1: skill coverage + TF-IDF
  scorer.py           stage-2: parallel full scoring
  vectorizer.py       TF-IDF index (scikit-learn)
  printer.py          ranked results summary output
  skill_extractor.py  regex skill signals from free text
  edu_validator.py    education credential sanity checks
  jd.py               structured JD definition
  overlap.py          QA: check ranked vs disqualified

resources/
  candidates.jsonl           input (git-ignored)
  top100_candidates.json     output of fast_ranker.py
  top100_candidates.csv      human-readable summary
```

