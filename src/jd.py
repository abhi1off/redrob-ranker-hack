# Structured JD definition for the Redrob Senior AI Engineer role.
# Hand-authored from the raw JD text since the JD has rich hard-disqualifiers
# and judgment calls that are hard to extract reliably with regex.

JD = {
    "job_id": "JD_REDROB_SR_AI_ENG_FOUNDING",
    "title": "Senior AI Engineer — Founding Team",
    "company": {
        "name": "Redrob AI",
        "stage": "Series A",
        "industry": "HR Tech / Recruiting Platform (AI-native talent intelligence)",
        "team_growth": {"current": 4, "target_in_12mo": 12},
    },

    # ------------------------------------------------------------------
    # LOCATION
    # ------------------------------------------------------------------
    "location": {
        "preferred_cities": ["Pune", "Noida"],
        "acceptable_cities": ["Hyderabad", "Pune", "Mumbai", "Delhi NCR", "Noida"],
        "tier_constraint": "tier_1_indian_cities",  # "Open to relocation candidates from Tier-1 Indian cities"
        "work_mode": "hybrid",
        "remote_allowed": False,
        "international_allowed": "case_by_case_no_visa_sponsorship",
        "relocation_offered": True,  # they're open to relocating Tier-1 city candidates
    },

    # ------------------------------------------------------------------
    # EXPERIENCE
    # ------------------------------------------------------------------
    "experience": {
        "min_years": 5,
        "max_years": 9,
        "band_is_soft": True,  # "we'll seriously consider candidates outside the band if other signals are strong"
        "ideal_total_years": [6, 8],
        "ideal_applied_ai_ml_years": [4, 5],
        "seniority_level": "senior_ic_with_mentorship",
        "must_be_hands_on": True,  # "writes code" - last 18mo must include production coding
        "hands_on_recency_months_max": 18,
    },

    # ------------------------------------------------------------------
    # SKILLS - must_have / nice_to_have, weighted
    # ------------------------------------------------------------------
    "required_skills": [
        # Tier 1: must-have, production-deployed
        {"name": "Embeddings", "category": "retrieval", "importance": "must_have",
         "weight": 1.0, "production_required": True,
         "aliases": ["sentence-transformers", "OpenAI embeddings", "BGE", "E5"]},
        {"name": "Vector Databases", "category": "retrieval", "importance": "must_have",
         "weight": 1.0, "production_required": True,
         "aliases": ["Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
                      "Elasticsearch", "FAISS"]},
        {"name": "Hybrid Search", "category": "retrieval", "importance": "must_have",
         "weight": 0.9, "production_required": True},
        {"name": "Python", "category": "engineering", "importance": "must_have",
         "weight": 0.9, "production_required": True},
        {"name": "Ranking Evaluation", "category": "evaluation", "importance": "must_have",
         "weight": 1.0, "production_required": False,
         "aliases": ["NDCG", "MRR", "MAP", "offline-online correlation", "A/B testing"]},
        {"name": "BM25", "category": "retrieval", "importance": "must_have",
         "weight": 0.6, "production_required": False},
        {"name": "LLM-based re-ranking", "category": "llm", "importance": "must_have",
         "weight": 0.5, "production_required": False},

        # Tier 2: nice-to-have, won't reject without
        {"name": "Fine-tuning LLMs", "category": "llm", "importance": "nice_to_have",
         "weight": 0.5, "production_required": False,
         "aliases": ["LoRA", "QLoRA", "PEFT"]},
        {"name": "Learning to Rank", "category": "ranking", "importance": "nice_to_have",
         "weight": 0.5, "production_required": False,
         "aliases": ["XGBoost ranker", "LightGBM ranker", "LambdaMART", "neural ranking"]},
        {"name": "HR-tech / Recruiting / Marketplace", "category": "domain", "importance": "nice_to_have",
         "weight": 0.3, "production_required": False},
        {"name": "Distributed Systems", "category": "infra", "importance": "nice_to_have",
         "weight": 0.3, "production_required": False,
         "aliases": ["large-scale inference optimization"]},
        {"name": "Open Source Contributions", "category": "credibility", "importance": "nice_to_have",
         "weight": 0.3, "production_required": False},
    ],

    # ------------------------------------------------------------------
    # HARD DISQUALIFIERS - if any true, candidate should be filtered/heavily penalized
    # ------------------------------------------------------------------
    "disqualifiers": [
        {
            "id": "pure_research_no_production",
            "description": "Career entirely in academic/research labs, no production deployment",
            "severity": "hard_reject",
        },
        {
            "id": "langchain_only_recent",
            "description": ("AI experience is primarily <12mo of LangChain/OpenAI wrapper "
                             "projects, with no substantial pre-LLM-era ML production experience"),
            "severity": "likely_reject",
            "exception": "substantial pre-LLM ML production experience present",
        },
        {
            "id": "stale_ic_18mo",
            "description": "Senior engineer who hasn't written production code in last 18 months "
                            "(moved fully into architecture/tech-lead, non-coding)",
            "severity": "likely_reject",
        },
        {
            "id": "title_chaser",
            "description": "Career trajectory shows title-chasing - company switch every ~1.5 years "
                            "chasing Senior->Staff->Principal",
            "severity": "soft_reject",
        },
        {
            "id": "framework_enthusiast_only",
            "description": "Profile dominated by LangChain tutorials / 'how I used X framework' "
                            "demo projects with no systems-level depth",
            "severity": "soft_reject",
        },
        {
            "id": "pure_services_career",
            "description": ("Entire career at consulting/services firms (TCS, Infosys, Wipro, "
                             "Accenture, Cognizant, Capgemini, etc.) with NO product-company experience"),
            "severity": "soft_reject",
            "exception": "currently at services firm but has prior product-company experience -> OK",
        },
        {
            "id": "cv_speech_robotics_no_nlp_ir",
            "description": "Primary expertise is CV/speech/robotics WITHOUT significant NLP/IR exposure",
            "severity": "soft_reject",
        },
        {
            "id": "closed_source_no_external_validation",
            "description": ("5+ years entirely on closed-source proprietary systems with zero "
                             "external validation: no papers, talks, OSS contributions"),
            "severity": "soft_reject",
        },
    ],

    "consulting_firms": ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
                          "Mindtree", "HCL", "Tech Mahindra", "LTI", "Mphasis"],

    # ------------------------------------------------------------------
    # COMP / LOGISTICS
    # ------------------------------------------------------------------
    "logistics": {
        "notice_period_days_ideal_max": 30,
        "notice_period_days_buyout_max": 30,
        "notice_period_above_threshold": "still in scope, higher bar",
    },

    # ------------------------------------------------------------------
    # "READING BETWEEN THE LINES" - ideal profile, used for soft scoring
    # ------------------------------------------------------------------
    "ideal_profile": {
        "total_experience_years": [6, 8],
        "applied_ai_ml_years": [4, 5],
        "company_type_required": "product_company",  # not pure services
        "must_have_shipped": "end-to-end ranking/search/recommendation system to real users at scale",
        "location_fit": ["Pune", "Noida"],
        "platform_activity_signal_required": True,  # active on Redrob / clearly job-seeking
    },

    "raw_text_for_bm25": None,  # filled programmatically with full JD text
}
