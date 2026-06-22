"""
Lightweight gazetteer-based skill extractor over free-text fields
(summary, career_history descriptions, headline).

Purpose: candidates frequently describe relevant work without listing
the corresponding term in `skills[]` (e.g. "migrated keyword search to
embedding-based retrieval" without a "Hybrid Search" skill entry, or
"offline-online correlation analysis" without "Ranking Evaluation").

This module finds such phrase-level signals and returns "soft" skill
entries that the matcher can merge with the structured skills list.
Soft matches get a lower proficiency ceiling (capped at "intermediate")
since text mentions aren't independently verified the way endorsed
skills are.

No LLM. Pure keyword/phrase matching (case-insensitive substring search).
For 100k scale, swap in flashtext.KeywordProcessor for speed, this uses
plain substring search for clarity/portability.
"""

import re


# Maps JD-canonical skill name -> list of text patterns (regex, case-insensitive)
# that, if found in candidate free text, indicate evidence of that skill
# even if absent from skills[].
TEXT_SIGNAL_PATTERNS = {
    "Python": [r"\bpython\b", r"\bpyspark\b", r"\bpandas\b"],
    "Hybrid Search": [
        r"hybrid (search|retrieval)",
        r"keyword[\s\-]based.{0,40}embedding",
        r"embedding[\s\-]based retrieval",
        r"migrat\w* .{0,40}(keyword|lexical).{0,40}(embedding|vector|dense|semantic)",
        r"(dense|sparse) (and|\+|\/) (sparse|dense)",
    ],
    "BM25": [
        r"\bbm25\b", r"\belasticsearch\b", r"\bopensearch\b", r"lexical (search|retrieval)",
        r"keyword[\s\-](based|search)",
    ],
    "Ranking Evaluation": [
        r"\bndcg\b", r"\bmrr\b", r"\bmap\b@", r"offline[\s\-]online correlation",
        r"offline[\s\-]to[\s\-]online", r"a\/b test", r"\bab test", r"evaluation framework",
        r"relevance labeling", r"offline benchmark", r"predicted .{0,20}(test|experiment) outcomes",
    ],
    "Learning to Rank": [
        r"learning[\s\-]to[\s\-]rank", r"\bltr\b", r"\bxgboost\b", r"\blightgbm\b",
        r"ranking model", r"hand[\s\-]tuned scoring function",
    ],
    "Embeddings": [
        r"\bembedding", r"sentence[\s\-]transformer", r"dense retrieval", r"vector representation",
    ],
    "Vector Databases": [
        r"\bfaiss\b", r"\bpinecone\b", r"\bweaviate\b", r"\bqdrant\b", r"\bmilvus\b",
        r"vector (database|index|store)",
    ],
    "LLM-based re-ranking": [
        r"llm[\s\-]based re[\s\-]?rank", r"re[\s\-]?ranking with (an? )?llm", r"cross[\s\-]encoder",
    ],
    "Fine-tuning LLMs": [
        r"\blora\b", r"\bqlora\b", r"\bpeft\b", r"fine[\s\-]tun\w* .{0,30}(llm|model|transformer)",
    ],
    "Distributed Systems": [
        r"distributed system", r"large[\s\-]scale inference", r"horizontal scal",
        r"\bkafka\b", r"\bspark\b", r"\bairflow\b",
    ],
    "HR-tech / Recruiting / Marketplace": [
        r"recruit\w*", r"job (search|matching)", r"candidate.{0,20}(search|matching|ranking)",
        r"marketplace",
    ],
    "Open Source Contributions": [
        r"open[\s\-]source", r"\bgithub\b.{0,20}(contribut|maintain|project)",
    ],
}

# Compile once
_COMPILED_PATTERNS = {
    skill: [re.compile(p, re.IGNORECASE) for p in patterns]
    for skill, patterns in TEXT_SIGNAL_PATTERNS.items()
}


# ----------------------------------------------------------------------
# SKILL IMPLICATION RULES
# ----------------------------------------------------------------------
# If a candidate has ANY of the "trigger" skills in skills[] (case-insensitive),
# infer the "implied" skill is also present, even if not separately listed.
#
# Rationale: practitioners frequently list the tools/libraries they use but
# not the underlying language/foundation those tools require. A candidate
# listing scikit-learn, MLflow, PySpark, or Hugging Face Transformers is
# using Python; listing these as a "Python" skill separately is rare.
#
# Implied proficiency = MIN of (trigger skill's proficiency, "intermediate")
# i.e. capped at intermediate since this is inference, not direct evidence.
SKILL_IMPLICATION_RULES = {
    "Python": {
        "triggers": [
            "scikit-learn", "pyspark", "pandas", "mlflow", "hugging face transformers",
            "sentence transformers", "xgboost", "lightgbm", "pytorch", "tensorflow",
            "numpy", "bentoml", "fastapi", "flask",
        ],
        "max_proficiency": "advanced",  # if trigger is advanced/expert, Python likely is too
    },
}


def _proficiency_rank(p):
    order = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    return order.get((p or "").lower(), 1)


def _cap_proficiency(p, cap):
    return p if _proficiency_rank(p) <= _proficiency_rank(cap) else cap


# ----------------------------------------------------------------------
# INDUSTRY-BASED INFERENCE
# ----------------------------------------------------------------------
# Some JD skill categories are domain categories rather than tools (e.g.
# "HR-tech / Recruiting / Marketplace"). If the candidate's current or past
# employer industry is a recognized marketplace/two-sided-platform industry,
# infer relevant exposure even without an exact text-pattern match.
MARKETPLACE_INDUSTRIES = {
    "food delivery", "e-commerce", "ride-hailing", "transportation",
    "marketplace", "travel", "logistics", "hr tech", "recruiting", "staffing",
}


def infer_marketplace_exposure(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    industries = {profile.get("current_industry", "").lower()}
    industries.update(c.get("industry", "").lower() for c in career)

    matched_industries = industries & MARKETPLACE_INDUSTRIES
    return bool(matched_industries), sorted(matched_industries)


def apply_skill_implications(skills):
    """
    Given a skills list, return additional inferred skill entries based on
    SKILL_IMPLICATION_RULES. Does not mutate input. Returns list of new
    entries only (caller merges).
    """
    existing_names_lower = {s["name"].lower() for s in skills}
    inferred = []

    for implied_name, rule in SKILL_IMPLICATION_RULES.items():
        if implied_name.lower() in existing_names_lower:
            continue  # already explicitly present

        best_trigger = None
        for s in skills:
            if s["name"].lower() in rule["triggers"]:
                if best_trigger is None or _proficiency_rank(s.get("proficiency")) > _proficiency_rank(best_trigger.get("proficiency")):
                    best_trigger = s

        if best_trigger is not None:
            inferred_proficiency = _cap_proficiency(best_trigger.get("proficiency", "intermediate"),
                                                       rule["max_proficiency"])
            inferred.append({
                "name": implied_name,
                "proficiency": inferred_proficiency,
                "endorsements": 0,
                "duration_months": best_trigger.get("duration_months", 12),
                "source": "skill_implication",
                "implied_from": best_trigger["name"],
            })

    return inferred


def extract_text_signals(candidate):
    """
    Returns dict: {skill_name: {"found": bool, "matched_patterns": [...], "evidence": [...]}}
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    text_blocks = [profile.get("headline", ""), profile.get("summary", "")]
    text_blocks += [c.get("description", "") for c in career]
    text_blocks += [c.get("title", "") for c in career]

    results = {}
    for skill, patterns in _COMPILED_PATTERNS.items():
        matched_patterns = []
        evidence = []
        for pattern in patterns:
            for block in text_blocks:
                if pattern.search(block):
                    matched_patterns.append(pattern.pattern)
                    # capture a short snippet around the match for audit
                    m = pattern.search(block)
                    start = max(0, m.start() - 30)
                    end = min(len(block), m.end() + 30)
                    evidence.append(block[start:end].strip())
                    break  # one evidence snippet per pattern is enough
        results[skill] = {
            "found": bool(matched_patterns),
            "matched_patterns": matched_patterns,
            "evidence": evidence[:2],  # cap evidence snippets
        }
    return results


def merge_text_signals_into_skills(candidate, max_soft_proficiency="intermediate"):
    """
    Returns a NEW skills list = original skills[] + soft entries for any
    TEXT_SIGNAL_PATTERNS skill not already present in skills[], found via
    extract_text_signals, plus skill-implication entries and
    marketplace-industry inference.

    Soft entries get:
      - proficiency = max_soft_proficiency (default "intermediate")
      - endorsements = 0
      - duration_months = a conservative estimate (12 months) since we can't
        derive tenure from a text mention alone
      - source = "text_signal" / "skill_implication" / "industry_inference"
        (so downstream code / audits can distinguish these from verified skills)
    """
    existing_skills = candidate.get("skills", [])
    existing_names_lower = {s["name"].lower() for s in existing_skills}

    signals = extract_text_signals(candidate)
    merged = list(existing_skills)  # shallow copy

    # 1. Apply skill-implication rules (e.g. scikit-learn -> Python)
    implied = apply_skill_implications(existing_skills)
    for entry in implied:
        if entry["name"].lower() not in existing_names_lower:
            merged.append(entry)
            existing_names_lower.add(entry["name"].lower())

    # 2. Apply text-signal extraction (e.g. "keyword-search-based" -> BM25)
    for skill_name, info in signals.items():
        if info["found"] and skill_name.lower() not in existing_names_lower:
            merged.append({
                "name": skill_name,
                "proficiency": max_soft_proficiency,
                "endorsements": 0,
                "duration_months": 12,
                "source": "text_signal",
                "evidence": info["evidence"],
            })
            existing_names_lower.add(skill_name.lower())

    # 3. Industry-based inference for domain categories (e.g. marketplace -> HR-tech/Marketplace)
    marketplace_target = "HR-tech / Recruiting / Marketplace"
    if marketplace_target.lower() not in existing_names_lower:
        has_marketplace, matched_industries = infer_marketplace_exposure(candidate)
        if has_marketplace:
            merged.append({
                "name": marketplace_target,
                "proficiency": "intermediate",
                "endorsements": 0,
                "duration_months": 12,
                "source": "industry_inference",
                "evidence": [f"Industry match: {', '.join(matched_industries)}"],
            })

    return merged, signals
