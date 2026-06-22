"""
candidate_loader.py — JSONL loading and fast hard-disqualification.

Responsibility:
  - Stream-load candidates.jsonl into memory.
  - Apply instant field-level disqualification (no regex, no scoring).
    Mirrors the hard rules in disqualifier_conditions.json.
"""

import json
import re
from pathlib import Path

# Bangalore is intentionally excluded: it is not in the JD's preferred or acceptable
# cities. Bangalore candidates who are willing to relocate still pass because the
# second condition below requires BOTH city mismatch AND unwilling to relocate.
_ACCEPTABLE_CITIES = {
    "noida", "pune", "hyderabad", "mumbai", "delhi ncr",
}
_BAD_TITLE_RE = re.compile(r"(manager|executive)", re.IGNORECASE)


def load_candidates(jsonl_path: Path) -> list[dict]:
    """Stream-load a JSONL file into a list of dicts."""
    candidates = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def fast_disqualify(candidate: dict) -> bool:
    """
    Return True if the candidate should be immediately dropped.
    Fast: only dict lookups + one regex, no external calls.

    Rules mirrored from disqualifier_conditions.json:
      - years_of_experience < 4
      - outside India and not willing to relocate
      - city not in acceptable list and not willing to relocate
      - current title matches manager / executive
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
    """
    Load all candidates and drop hard-disqualified ones.

    Returns:
        (survived_candidates, total_loaded)
        survived_candidates preserves original list order.
    """
    all_candidates = load_candidates(jsonl_path)
    survived = [c for c in all_candidates if not fast_disqualify(c)]
    return survived, len(all_candidates)
