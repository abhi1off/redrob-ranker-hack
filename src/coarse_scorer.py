"""
coarse_scorer.py — Stage-1 coarse scoring: skill coverage + TF-IDF similarity.

Responsibility:
  - Build a JD skill weight map for fast lookup.
  - Score each candidate by skill coverage (presence/absence, no proficiency).
  - Combine skill score + TF-IDF score into a single coarse score.
  - Return candidates sorted by coarse score, optionally capped at shortlist_size.

This module is intentionally lightweight — no regex, no full scoring pipeline.
"""

from __future__ import annotations

from corpus_vectorizer import CorpusVectorizer


# ---------------------------------------------------------------------------
# JD skill weight map
# ---------------------------------------------------------------------------

def build_jd_skill_weight_map(jd: dict) -> dict:
    """
    Build {lowercase_name_or_alias: (canonical_skill_name, weight)}.
    Canonical name deduplicates alias matches in fast_skill_score().
    """
    weight_map: dict[str, tuple[str, float]] = {}
    for skill in jd["required_skills"]:
        w = skill["weight"]
        canonical = skill["name"].lower()
        for name in [skill["name"]] + skill.get("aliases", []):
            weight_map[name.lower()] = (canonical, w)
    return weight_map


# ---------------------------------------------------------------------------
# Fast skill coverage score
# ---------------------------------------------------------------------------

def fast_skill_score(candidate: dict, jd_skill_weights: dict) -> float:
    """
    Sum weights of JD skills the candidate has (presence only, no proficiency).
    Deduplicates aliases via canonical skill name.

    Returns value in [0, 1].
    """
    # Denominator: sum unique JD skill weights (not unique weight values)
    seen_canonicals: dict[str, float] = {}
    for canonical, w in jd_skill_weights.values():
        seen_canonicals[canonical] = w
    total_weight = sum(seen_canonicals.values())

    achieved = 0.0
    matched_canonicals: set[str] = set()
    for skill in candidate.get("skills", []):
        name_lower = skill.get("name", "").lower()
        if name_lower in jd_skill_weights:
            canonical, jd_weight = jd_skill_weights[name_lower]
            if canonical not in matched_canonicals:
                matched_canonicals.add(canonical)
                achieved += jd_weight
    return min(achieved / (total_weight or 1.0), 1.0)


# ---------------------------------------------------------------------------
# Coarse ranking
# ---------------------------------------------------------------------------

COARSE_SKILL_WEIGHT  = 0.35
COARSE_TFIDF_WEIGHT  = 0.25


def coarse_rank(
    candidates: list[dict],
    jd: dict,
    shortlist_size: int | None = None,
    precomputed_tfidf: list[float] | None = None,
) -> list[tuple[float, dict, float]]:
    """
    Score all candidates using the fast coarse pipeline and return them sorted.

    Steps:
      1. Build JD skill weight map.
      2. Score each candidate's skill coverage.
      3. Fit corpus-wide TF-IDF index (or use precomputed_tfidf if provided).
      4. Combine: coarse_score = SKILL_WEIGHT x skill + TFIDF_WEIGHT x tfidf.
      5. Sort descending; cap at shortlist_size if given.

    Args:
        precomputed_tfidf: optional list of floats aligned with candidates,
                           computed on the full 100k corpus for accurate IDF.
                           If None, TF-IDF is built from candidates only.

    Returns:
        list of (coarse_score, candidate_dict, tfidf_score) tuples, sorted
        by coarse_score descending.
    """
    jd_skill_weights = build_jd_skill_weight_map(jd)

    # Skill scores
    skill_scores = [fast_skill_score(c, jd_skill_weights) for c in candidates]

    # TF-IDF scores — use pre-computed full-corpus scores when available
    if precomputed_tfidf is not None:
        tfidf_arr = precomputed_tfidf
    else:
        cv = CorpusVectorizer()
        cv.fit(candidates)
        jd_text = jd.get("raw_text_for_bm25", "")
        tfidf_arr = cv.score_all(jd_text)

    # Combine
    coarse: list[tuple[float, dict, float]] = []
    for i, candidate in enumerate(candidates):
        s_skill = skill_scores[i]
        s_tfidf = float(tfidf_arr[i])
        score   = COARSE_SKILL_WEIGHT * s_skill + COARSE_TFIDF_WEIGHT * s_tfidf
        coarse.append((score, candidate, s_tfidf))

    coarse.sort(key=lambda x: x[0], reverse=True)
    return coarse[:shortlist_size] if shortlist_size else coarse
