# Stage-1 coarse scoring: fast skill coverage check + TF-IDF similarity.
# No proficiency weighting here, that's done in stage 2.

from __future__ import annotations

from vectorizer import CorpusVectorizer


def build_jd_skill_weight_map(jd: dict) -> dict:
    """Build a flat lookup: lowercase skill name/alias -> (canonical name, weight)."""
    weight_map: dict[str, tuple[str, float]] = {}
    for skill in jd["required_skills"]:
        w = skill["weight"]
        canonical = skill["name"].lower()
        for name in [skill["name"]] + skill.get("aliases", []):
            weight_map[name.lower()] = (canonical, w)
    return weight_map


def fast_skill_score(candidate: dict, jd_skill_weights: dict) -> float:
    """Check which JD skills the candidate has and sum their weights. Returns 0-1."""
    # get total possible weight (deduplicated by canonical name)
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


COARSE_SKILL_WEIGHT = 0.35
COARSE_TFIDF_WEIGHT = 0.25


def coarse_rank(
    candidates: list[dict],
    jd: dict,
    shortlist_size: int | None = None,
    precomputed_tfidf: list[float] | None = None,
) -> list[tuple[float, dict, float]]:
    """
    Score all candidates and return them sorted by coarse score.
    Pass precomputed_tfidf (from the full corpus) for better IDF accuracy.
    Returns list of (coarse_score, candidate, tfidf_score) tuples.
    """
    jd_skill_weights = build_jd_skill_weight_map(jd)

    # Skill scores
    skill_scores = [fast_skill_score(c, jd_skill_weights) for c in candidates]

    # use precomputed scores if available, otherwise build TF-IDF from scratch
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
