import re
from datetime import date, datetime
import json
from collections import Counter

def collect_bot_signatures(candidates_path):
    phrase_counts = Counter()

    with open(candidates_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                summary = candidate.get("profile", {}).get("summary", "")
                if summary:
                    normalized = " ".join(summary.split())
                    sentences = [
                        s.strip()
                        for s in re.split(r'[.!?]', normalized)
                        if len(s.strip()) > 150
                    ]
                    for sentence in sentences:
                        phrase_counts[sentence] += 1
            except json.JSONDecodeError:
                continue

    bot_signatures = {phrase for phrase, count in phrase_counts.items() if count > 5}
    print(f"remove_disqualified_candidates::Discovered {len(bot_signatures)} unique bot text signatures.")
    return bot_signatures