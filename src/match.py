import json
from jd import JD
from matcher import score_candidate, build_skill_alias_map, score_candidates
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR.parent / "resources" / "candidates.jsonl"

candidates_qualified = []
with open(file_path) as f:
    for line in f:
        candidates_qualified.append(json.loads(line))

# Load the full JD raw text for BM25 component
with open(BASE_DIR.parent / "resources" / "jd_raw_text.txt") as f:
    JD["raw_text_for_bm25"] = f.read()

print(f"qualified candidates: {len(candidates_qualified)}")
scored_candidates = score_candidates(candidates_qualified, jd= JD, include_details=True, top_n=100)

output_path = BASE_DIR.parent / "resources" / "scored_candidates.json"
with open(output_path, "w") as f:
    json.dump(scored_candidates, f, indent=2)

print(f"Saved results to: {output_path}")
print(json.dumps(scored_candidates, indent=2))
