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

## Two ways to run

### Option A — small batch

Pre-filter with rule-based disqualification, then score:

```bash
cd src
python disqualifier.py   # splits candidates.jsonl → qualified / disqualified
python match.py          # scores qualified candidates, outputs scored_candidates.json
```

### Option B — fast 100k ranker (recommended)

Two-stage pipeline, reads directly from `candidates.jsonl`:

```bash
cd src
python fast_ranker.py
```

Common flags:
```bash
python fast_ranker.py --top_n 50 --shortlist 3000   # faster, slightly lower recall
python fast_ranker.py --workers 2                    # limit CPU usage
```

Outputs:
- `resources/top100_candidates.json` — full results
- `resources/top100_candidates.txt` — readable summary

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

## QA check

If you ran both pipelines, verify no disqualified candidate snuck into the top results:

```bash
cd src
python overlap.py
```

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
  disqualifier.py     config-driven pre-filter (Option A)
  rules.json          disqualification rule config
  match.py            small-batch scoring script (Option A)
  ranker.py           CLI wrapper for disqualifier
  overlap.py          QA: check ranked vs disqualified

resources/
  candidates.jsonl               input (git-ignored)
  candidates_qualified.jsonl     output of disqualifier.py
  candidates_disqualified.jsonl  rejected by disqualifier.py
  scored_candidates.json         output of match.py
  top100_candidates.json         output of fast_ranker.py
  top100_candidates.txt          human-readable summary
```

