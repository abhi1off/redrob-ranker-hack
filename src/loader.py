# Load candidates.jsonl and apply fast hard-disqualification rules.

import json
import re
from pathlib import Path

# Cities the JD considers acceptable. Bangalore is intentionally excluded
# but candidates willing to relocate still pass.
_ACCEPTABLE_CITIES = {
    "noida", "pune", "hyderabad", "mumbai", "delhi ncr",
}
_BAD_TITLE_RE = re.compile(r"(manager|executive)", re.IGNORECASE)


def load_candidates(jsonl_path: Path) -> list[dict]:
    """Read a JSONL file and return a list of candidate dicts."""
    candidates = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def fast_disqualify(candidate: dict) -> bool:
    """
    Return True if the candidate should be dropped immediately.
    Only does dict lookups and one regex — no scoring.

    Rules:
      - less than 4 years experience
      - outside India and not willing to relocate
      - city not in acceptable list and not willing to relocate
      - current title contains manager or executive
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    if profile.get("years_of_experience", 0) < 4:
        return True

    if (
        profile.get("country", "").lower() != "india"
        and not signals.get("willing_to_relocate", False)
    ):
        return True

    city = profile.get("location", "").split(",")[0].strip().lower()
    if city not in _ACCEPTABLE_CITIES and not signals.get("willing_to_relocate", False):
        return True

    if _BAD_TITLE_RE.search(profile.get("current_title", "")):
        return True

    return False


def load_and_filter(jsonl_path: Path) -> tuple[list[dict], int]:
    """Load all candidates and drop disqualified ones. Returns (survivors, total_loaded)."""
    all_candidates = load_candidates(jsonl_path)
    survived = [c for c in all_candidates if not fast_disqualify(c)]
    return survived, len(all_candidates)
