# Scoring engine for matching candidates against the JD. No LLM calls.
# Produces a weighted score from 5 components:
#   skill match, experience, location, text similarity, and disqualifier penalties.

import math
import re
from datetime import datetime, date

from skill_extractor import merge_text_signals_into_skills
from edu_validator import validate_education


PROFICIENCY_WEIGHT = {"beginner": 1.0, "intermediate": 2.0, "advanced": 3.0, "expert": 3.5}

# Skill name -> set of aliases, built once from JD for fast lookup.
def build_skill_alias_map(jd):
    alias_map = {}
    for skill in jd["required_skills"]:
        names = [skill["name"]] + skill.get("aliases", [])
        for n in names:
            alias_map[n.lower()] = skill
    return alias_map


# 1. DISQUALIFIER CHECKS

def check_disqualifiers(candidate, jd):
    """Return a dict of flag_id -> {triggered, severity, reason} for each disqualifier rule."""
    flags = {}
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    skill_names = {s["name"].lower() for s in skills}

    # pure research career: all entries are research/academic, no industry work
    research_keywords = {"research", "academia", "university", "lab"}
    all_industries = {c.get("industry", "").lower() for c in career}
    looks_pure_research = bool(all_industries) and all_industries.issubset(research_keywords)
    flags["pure_research_no_production"] = {
        "triggered": looks_pure_research,
        "severity": "hard_reject",
        "reason": "All career_history entries are research/academic industries"
                  if looks_pure_research else "Has non-research industry experience",
    }

    # langchain-only: only recent LLM wrapper skills with no pre-LLM ML foundation
    total_years = profile.get("years_of_experience", 0)
    llm_recent_skills = [
        s for s in skills
        if s["name"].lower() in {"langchain", "fine-tuning llms", "lora", "qlora", "peft"}
        and s.get("duration_months", 0) < 12
    ]
    pre_llm_skills = [
        s for s in skills
        if s["name"].lower() in {"sql", "spark", "airflow", "etl", "statistical modeling",
                                  "nlp", "data engineering", "hadoop", "kafka"}
        and s.get("duration_months", 0) >= 24
    ]
    only_recent_llm_wrapper = bool(llm_recent_skills) and not pre_llm_skills and total_years < 2
    flags["langchain_only_recent"] = {
        "triggered": only_recent_llm_wrapper,
        "severity": "likely_reject",
        "reason": "Only recent (<12mo) LLM-wrapper skills, no pre-LLM ML production signal"
                  if only_recent_llm_wrapper else "Has pre-LLM ML production experience or sufficient tenure",
    }

    # stale IC: currently in a non-coding role (architect/lead/manager) for 18+ months
    non_coding_titles = {"manager", "architect", "lead", "director", "head"}
    current_role = next((c for c in career if c.get("is_current")), None)
    stale_ic = False
    if current_role:
        title_lower = current_role.get("title", "").lower()
        if any(t in title_lower for t in non_coding_titles) and current_role.get("duration_months", 0) > 18:
            stale_ic = True
    flags["stale_ic_18mo"] = {
        "triggered": stale_ic,
        "severity": "likely_reject",
        "reason": "Current role appears non-coding (architect/lead/manager) for >18 months"
                  if stale_ic else "Current role appears hands-on / within recency window",
    }

    # title chaser: 2+ short stints with escalating seniority titles
    short_stints = [c for c in career if c.get("duration_months", 999) <= 18]
    escalation_words = ["senior", "staff", "principal", "lead"]
    escalating = False
    titles = [c.get("title", "").lower() for c in career]
    if len(short_stints) >= 2:
        # crude check: do later jobs contain higher-seniority words than earlier ones?
        levels = []
        for t in titles:
            level = max([i for i, w in enumerate(escalation_words) if w in t], default=-1)
            levels.append(level)
        escalating = levels == sorted(levels) and levels[-1] > levels[0] if levels else False
    title_chaser = len(short_stints) >= 2 and escalating
    flags["title_chaser"] = {
        "triggered": title_chaser,
        "severity": "soft_reject",
        "reason": "Multiple short stints (<=18mo) with escalating seniority titles"
                  if title_chaser else "No clear title-chasing pattern",
    }

    # framework-only: trendy tools with no systems/infra depth and short tenure
    trendy = {"langchain", "tailwind", "flask"}
    systems = {"spark", "airflow", "kafka", "sql", "distributed systems", "milvus", "faiss",
               "elasticsearch", "apache beam"}
    has_trendy = any(s["name"].lower() in trendy for s in skills)
    has_systems = any(s["name"].lower() in systems for s in skills)
    framework_only = has_trendy and not has_systems and total_years < 2
    flags["framework_enthusiast_only"] = {
        "triggered": framework_only,
        "severity": "soft_reject",
        "reason": "Only trendy-framework skills, no systems/infra depth, short tenure"
                  if framework_only else "Has systems/infra depth or sufficient tenure",
    }

    # pure consulting career: every company in career history is a services firm
    consulting_firms_lower = {f.lower() for f in jd.get("consulting_firms", [])}
    company_names_lower = [c.get("company", "").lower() for c in career]
    all_consulting = bool(company_names_lower) and all(
        any(cf in cn for cf in consulting_firms_lower) for cn in company_names_lower
    )
    flags["pure_services_career"] = {
        "triggered": all_consulting,
        "severity": "soft_reject",
        "reason": "Entire career_history is at consulting/services firms with no product-company experience"
                  if all_consulting else "Has product-company experience (or mixed history)",
    }

    # CV/speech focus with no NLP/IR exposure
    cv_speech_skills = {"image classification", "gans", "speech recognition", "tts",
                         "computer vision", "robotics", "object detection"}
    nlp_ir_skills = {"nlp", "bm25", "embeddings", "milvus", "elasticsearch", "information retrieval",
                      "retrieval", "ranking", "search"}
    has_cv_speech = any(s["name"].lower() in cv_speech_skills for s in skills)
    has_nlp_ir = any(s["name"].lower() in nlp_ir_skills for s in skills)
    cv_no_nlp = has_cv_speech and not has_nlp_ir
    flags["cv_speech_robotics_no_nlp_ir"] = {
        "triggered": cv_no_nlp,
        "severity": "soft_reject",
        "reason": "Has CV/speech skills but no NLP/IR/retrieval/ranking exposure"
                  if cv_no_nlp else "Has NLP/IR exposure (or no CV/speech focus)",
    }

    # no external validation signal (5+ years but very low github activity)
    redrob = candidate.get("redrob_signals", {})
    github_score = redrob.get("github_activity_score", 0)
    closed_source_flag = total_years >= 5 and github_score < 2.0
    flags["closed_source_no_external_validation"] = {
        "triggered": closed_source_flag,
        "severity": "soft_reject",
        "reason": f"5+ yrs experience but github_activity_score very low ({github_score}), "
                  "no proxy signal of external validation"
                  if closed_source_flag else "Has some external-validation proxy signal or <5yrs experience",
    }

    # --- suspicious_education ---
    edu_issues = validate_education(candidate)
    flags["suspicious_education"] = {
        "triggered": bool(edu_issues),
        "severity": "soft_reject",
        "reason": "; ".join(edu_issues) if edu_issues else "Education credentials appear consistent",
    }

    return flags


SEVERITY_PENALTY = {
    "hard_reject": 0.0,      # zeroes out the score entirely
    "likely_reject": 0.35,   # multiplies score by this
    "soft_reject": 0.75,     # mild multiplicative penalty
}


# 2. SKILL MATCH SCORE

def skill_match_score(candidate, jd, alias_map):
    skills, text_signals = merge_text_signals_into_skills(candidate)
    candidate_skill_map = {s["name"].lower(): s for s in skills}

    total_weight = sum(s["weight"] for s in jd["required_skills"])
    achieved = 0.0
    breakdown = []

    for jd_skill in jd["required_skills"]:
        names_to_check = [jd_skill["name"].lower()] + [a.lower() for a in jd_skill.get("aliases", [])]
        matched_skill = None
        for n in names_to_check:
            if n in candidate_skill_map:
                matched_skill = candidate_skill_map[n]
                break

        if matched_skill is None:
            breakdown.append({"jd_skill": jd_skill["name"], "matched": False, "contribution": 0.0})
            continue

        prof_weight = PROFICIENCY_WEIGHT.get(matched_skill.get("proficiency", "beginner"), 1.0)
        recency_factor = min(matched_skill.get("duration_months", 0) / 36.0, 1.5)
        endorsement_factor = 1 + 0.1 * math.log1p(matched_skill.get("endorsements", 0))

        # normalize proficiency contribution to [0,1] before applying jd weight
        # (cap prof_norm at 1.0 so "expert" doesn't exceed the per-skill ceiling)
        prof_norm = min(prof_weight / 3.0, 1.0)
        per_skill_multiplier = min(prof_norm * min(recency_factor, 1.0) * min(endorsement_factor, 1.3), 1.0)
        contribution = jd_skill["weight"] * per_skill_multiplier

        # inferred/unverified skills get dampened contributions
        source = matched_skill.get("source", "structured")
        if source == "skill_implication":
            contribution *= 0.85
        elif source == "industry_inference":
            contribution *= 0.5
        elif source == "text_signal":
            contribution *= 0.3   # halved: text regex matches are weak evidence

        achieved += contribution

        breakdown.append({
            "jd_skill": jd_skill["name"],
            "matched": True,
            "matched_via": matched_skill["name"],
            "source": source,
            "proficiency": matched_skill.get("proficiency"),
            "duration_months": matched_skill.get("duration_months"),
            "endorsements": matched_skill.get("endorsements"),
            "contribution": round(contribution, 4),
        })

    score = achieved / total_weight if total_weight else 0.0
    return min(score, 1.0), breakdown


# 3. EXPERIENCE MATCH SCORE

def experience_match_score(candidate, jd):
    profile = candidate.get("profile", {})
    total_years = profile.get("years_of_experience", 0)
    exp_req = jd["experience"]

    min_y, max_y = exp_req["min_years"], exp_req["max_years"]
    ideal_min, ideal_max = exp_req["ideal_total_years"]

    # score peaks in the ideal range, decays outside but band is soft (no hard zero)
    if ideal_min <= total_years <= ideal_max:
        years_score = 1.0
    elif min_y <= total_years <= max_y:
        years_score = 0.85
    else:
        # decay outside the 5-9 band; still a soft floor since band_is_soft=True
        if total_years < min_y:
            gap = min_y - total_years
        else:
            gap = total_years - max_y
        # lower floor for very over-experienced candidates (15+ yrs -> ~0.15)
        years_score = max(0.15, 1.0 - 0.15 * gap)

    # "applied AI/ML years" proxy: sum duration_months of AI/ML-flavored skills / 12,
    # capped at total_years
    ai_ml_skill_keywords = {
        "nlp", "image classification", "fine-tuning llms", "lora", "qlora", "peft", "gans",
        "speech recognition", "tts", "milvus", "weights & biases", "bentoml",
        "embeddings", "llm", "ranking", "qdrant", "faiss", "weaviate", "pinecone",
        "opensearch", "elasticsearch", "bm25", "information retrieval", "retrieval",
        "vector databases", "hybrid search", "sentence transformers",
        "learning to rank", "ranking evaluation",
    }
    ai_ml_months = max(
        (s.get("duration_months", 0) for s in candidate.get("skills", [])
         if s["name"].lower() in ai_ml_skill_keywords),
        default=0,
    )
    applied_ai_years = min(ai_ml_months / 12.0, total_years)
    ideal_ai_min, ideal_ai_max = exp_req["ideal_applied_ai_ml_years"]
    if ideal_ai_min <= applied_ai_years <= ideal_ai_max:
        ai_years_score = 1.0
    elif applied_ai_years > 0:
        ai_years_score = max(0.4, 1.0 - 0.2 * abs(applied_ai_years - ideal_ai_min))
    else:
        ai_years_score = 0.2  # some applied AI signal expected

    # hands-on recency check: current role title shouldn't be pure-management
    career = candidate.get("career_history", [])
    current_role = next((c for c in career if c.get("is_current")), None)
    hands_on_score = 1.0
    if current_role:
        title_lower = current_role.get("title", "").lower()
        if any(t in title_lower for t in ["manager", "director", "head"]):
            hands_on_score = 0.3
        elif "architect" in title_lower or "lead" in title_lower:
            hands_on_score = 0.6

    combined = 0.4 * years_score + 0.4 * ai_years_score + 0.2 * hands_on_score
    return combined, {
        "total_years": total_years,
        "years_score": round(years_score, 3),
        "applied_ai_years_estimate": round(applied_ai_years, 2),
        "ai_years_score": round(ai_years_score, 3),
        "hands_on_score": hands_on_score,
        "combined": round(combined, 3),
    }


# 4. LOCATION / LOGISTICS MATCH SCORE

def location_logistics_score(candidate, jd):
    profile = candidate.get("profile", {})
    redrob = candidate.get("redrob_signals", {})
    loc = jd["location"]

    candidate_city_raw = profile.get("location", "")
    # candidate location is often "Hyderabad, Telangana" — compare on first token
    candidate_city_primary = candidate_city_raw.split(",")[0].strip()

    def city_in(city_list):
        city_list_lower = {c.lower() for c in city_list}
        return (candidate_city_raw.lower() in city_list_lower
                or candidate_city_primary.lower() in city_list_lower)

    candidate_city = candidate_city_raw  # for messages
    score = 0.0
    reasons = []

    if city_in(loc["preferred_cities"]):
        score += 0.5
        reasons.append(f"In preferred city ({candidate_city})")
    elif city_in(loc["acceptable_cities"]):
        score += 0.35
        reasons.append(f"In acceptable city ({candidate_city})")
    else:
        if redrob.get("willing_to_relocate") and loc.get("relocation_offered"):
            score += 0.2
            reasons.append("Not in target city but willing to relocate; relocation offered")
        else:
            score += 0.0
            reasons.append(f"Not in target/acceptable city ({candidate_city}), not willing to relocate")

    # notice period
    notice = redrob.get("notice_period_days", 999)
    if notice <= jd["logistics"]["notice_period_days_ideal_max"]:
        score += 0.3
        reasons.append(f"Notice period {notice}d within ideal <=30d")
    elif notice <= 60:
        score += 0.15
        reasons.append(f"Notice period {notice}d above ideal but within buyout/scope range")
    elif notice <= 90:
        score -= 0.10
        reasons.append(f"Notice period {notice}d high — beyond buyout window")
    else:
        score -= 0.20
        reasons.append(f"Notice period {notice}d very high — unplaceable within buyout window")

    # work mode preference vs hybrid requirement
    if redrob.get("preferred_work_mode") in (None, "hybrid", "onsite"):
        score += 0.2
        reasons.append("Work-mode preference compatible with hybrid")
    else:
        reasons.append("Work-mode preference (remote) may not fit hybrid requirement")

    return min(score, 1.0), reasons


# 5. BM25 TEXT SIMILARITY

def simple_bm25_score(candidate, jd_text, k1=1.5, b=0.75):
    """
    Single-document BM25-like score against JD text.
    IDF is a uniform prior here (no corpus). At 100k scale, fast_ranker
    passes a corpus TF-IDF score instead, so this only runs for small batches.
    """
    def tokenize(text):
        return re.findall(r"[a-zA-Z][a-zA-Z0-9\-\+#]*", text.lower())

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    candidate_text = " ".join([
        profile.get("headline", ""),
        profile.get("summary", ""),
        " ".join(c.get("description", "") for c in career),
        " ".join(c.get("title", "") for c in career),
    ])

    cand_tokens = tokenize(candidate_text)
    jd_tokens = tokenize(jd_text)
    jd_token_set = set(jd_tokens)

    if not cand_tokens or not jd_token_set:
        return 0.0

    doc_len = len(cand_tokens)
    avg_len = max(doc_len, 1)  # single-doc approximation

    from collections import Counter
    tf = Counter(cand_tokens)

    score = 0.0
    matched_terms = 0
    for term in jd_token_set:
        f = tf.get(term, 0)
        if f == 0:
            continue
        matched_terms += 1
        idf = 1.0  # uniform prior; replace with corpus IDF at scale
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * doc_len / avg_len)
        score += idf * (numerator / denominator)

    # normalize roughly to [0,1] by dividing by a soft cap
    norm_score = min(score / (0.3 * len(jd_token_set) + 1e-9), 1.0)
    return norm_score


# 6. FINAL COMBINED SCORE

WEIGHTS = {
    "skill": 0.35,
    "experience": 0.30,
    "location": 0.10,
    "text_similarity": 0.25,
}


def score_candidate(candidate, jd, alias_map=None, precomputed_text_score: float | None = None):
    if alias_map is None:
        alias_map = build_skill_alias_map(jd)

    flags = check_disqualifiers(candidate, jd)
    skill_score, skill_breakdown = skill_match_score(candidate, jd, alias_map)
    exp_score, exp_breakdown = experience_match_score(candidate, jd)
    loc_score, loc_reasons = location_logistics_score(candidate, jd)
    # use corpus TF-IDF if available (from fast_ranker), otherwise fall back to BM25
    text_score = (
        float(precomputed_text_score)
        if precomputed_text_score is not None
        else simple_bm25_score(candidate, jd.get("raw_text_for_bm25") or "")
    )

    raw_score = (
        WEIGHTS["skill"] * skill_score
        + WEIGHTS["experience"] * exp_score
        + WEIGHTS["location"] * loc_score
        + WEIGHTS["text_similarity"] * text_score
    )

    # multiply penalties together (hard_reject -> x0.0 zeroes it out)
    penalty_multiplier = 1.0
    triggered_flags = []
    for flag_id, flag in flags.items():
        if flag["triggered"]:
            triggered_flags.append(flag_id)
            penalty_multiplier *= SEVERITY_PENALTY[flag["severity"]]

    final_score = raw_score * penalty_multiplier

    return {
        "candidate_id": candidate.get("candidate_id"),
        "final_score": round(final_score, 4),
        "raw_score_before_penalties": round(raw_score, 4),
        "penalty_multiplier": round(penalty_multiplier, 4),
        "triggered_disqualifiers": triggered_flags,
        "component_scores": {
            "skill_score": round(skill_score, 4),
            "experience_score": round(exp_score, 4),
            "location_score": round(loc_score, 4),
            "text_similarity_score": round(text_score, 4),
        },
        "details": {
            "skill_breakdown": skill_breakdown,
            "experience_breakdown": exp_breakdown,
            "location_reasons": loc_reasons,
            "disqualifier_flags": flags,
        },
    }


# 7. BATCH SCORING

def score_candidates(candidates, jd, top_n=None, include_details=True):
    """Score a list of candidates against a JD. Returns sorted results with rank attached."""
    alias_map = build_skill_alias_map(jd)

    results = []
    for candidate in candidates:
        result = score_candidate(candidate, jd, alias_map=alias_map)
        if not include_details:
            result.pop("details", None)
        results.append(result)

    # Sort by final_score descending
    results.sort(key=lambda r: r["final_score"], reverse=True)

    # Add 1-based rank
    for i, result in enumerate(results):
        result["rank"] = i + 1

    if top_n is not None:
        results = results[:top_n]

    return results