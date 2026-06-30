# Check education credentials for obvious issues (impossible durations,
# programs that didn't exist yet, tier mismatches).
# Returns a list of issue strings; empty = no problems found.

from __future__ import annotations


# ---------------------------------------------------------------------------
# 1. Expected degree duration (years)
# ---------------------------------------------------------------------------
_DEGREE_DURATION: dict[str, tuple[int, int]] = {
    # (min_years, max_years)
    # Diploma lateral-entry students complete B.E./B.Tech in 3 years (skip first year),
    # so min is 3, not 4.
    "b.e.":     (3, 5),
    "b.tech":   (3, 5),
    "b.sc":     (3, 4),
    "b.sc.":    (3, 4),
    "bsc":      (3, 4),
    "b.com":    (3, 4),
    "b.com.":   (3, 4),
    "bca":      (3, 4),
    "b.ca":     (3, 4),
    "mca":      (3, 3),
    # M.Tech standard = 2 years; part-time / sponsored / thesis-extension programmes
    # are officially allowed up to 3 years at most Indian universities.
    "m.tech":   (2, 3),
    "m.e.":     (2, 3),
    "m.sc":     (2, 3),
    "m.sc.":    (2, 3),
    "msc":      (2, 3),
    "mba":      (2, 2),
    "m.b.a.":   (2, 2),
    "pgdm":     (2, 2),
    "ph.d":     (3, 7),
    "phd":      (3, 7),
    "ph.d.":    (3, 7),
}


def _check_duration(degree: str, start_year: int | None, end_year: int | None) -> str | None:
    """Return an issue string if the degree duration is outside expected range."""
    if start_year is None or end_year is None:
        return None
    duration = end_year - start_year
    if duration <= 0:
        return f"Education end_year ({end_year}) is not after start_year ({start_year})"

    degree_key = degree.strip().lower().rstrip(".")
    # Try progressively shorter prefixes to match e.g. "b.e" -> "b.e."
    for key, (min_y, max_y) in _DEGREE_DURATION.items():
        if degree_key.startswith(key) or key.startswith(degree_key):
            if duration < min_y:
                return (
                    f"{degree} typically takes {min_y}–{max_y} years but "
                    f"this record shows only {duration} year(s) ({start_year}–{end_year})"
                )
            if duration > max_y + 2:   # +2 to allow for gap years / repeat years
                return (
                    f"{degree} typically takes {min_y}–{max_y} years but "
                    f"this record shows {duration} year(s) ({start_year}–{end_year})"
                )
    return None


# ---------------------------------------------------------------------------
# 2. Institution → known programs + earliest plausible graduation year
# ---------------------------------------------------------------------------
# Structure: institution_key → {program_key: earliest_graduation_year}
# institution_key: lowercase substring that would appear in the institution name
# program_key: lowercase substring that would appear in the field_of_study
_INSTITUTION_PROGRAMS: dict[str, dict[str, int]] = {
    # COEP (College of Engineering Pune)
    "coep": {
        "computer science":         1970,
        "cs":                       1970,
        "information technology":   2000,
        "it":                       2000,
        "mechanical":               1950,
        "civil":                    1950,
        "electrical":               1950,
        "electronics":              1960,
        "e&tc":                     1960,
        "instrumentation":          1975,
        "production":               1975,
        "data science":             2023,   # B.E. Data Science started only ~2022
        "artificial intelligence":  2022,
        "ai":                       2022,
    },
    # IIT Bombay
    "iit bombay": {
        "computer science":         1960,
        "electrical":               1960,
        "mechanical":               1960,
        "data science":             2020,
        "artificial intelligence":  2020,
    },
    # IIT Delhi
    "iit delhi": {
        "computer science":         1965,
        "electrical":               1965,
        "data science":             2020,
    },
    # BITS Pilani
    "bits pilani": {
        "computer science":         1970,
        "electrical":               1970,
        "data science":             2021,
    },
    "bits": {
        "computer science":         1970,
        "data science":             2021,
    },
    # VIT
    "vit": {
        "computer science":         1990,
        "it":                       2000,
        "data science":             2020,
        "artificial intelligence":  2020,
    },
    # SRM
    "srm": {
        "computer science":         1990,
        "data science":             2020,
    },
    # NIT Trichy
    "nit trichy": {
        "computer science":         1970,
        "data science":             2022,
    },
    # Generic NIT
    "nit ": {
        "data science":             2022,
        "artificial intelligence":  2022,
    },
    # Pune University (affiliated colleges)
    "pune university": {
        "data science":             2022,
    },
}

# Tier truth table for known institutions (lowercase substring → actual tier)
_INSTITUTION_TIER: dict[str, str] = {
    "iit":          "tier_1",
    "bits pilani":  "tier_1",
    "nit trichy":   "tier_1",
    "nit surathkal":"tier_1",
    "nit warangal": "tier_1",
    "coep":         "tier_1",
    "dtu":          "tier_1",
    "nsit":         "tier_1",
    "iiit hyderabad":"tier_1",
    "iiit bangalore":"tier_1",
    "vit":          "tier_2",
    "srm":          "tier_2",
    "manipal":      "tier_2",
    "amity":        "tier_3",
}


def _check_program_year(
    institution: str,
    field_of_study: str,
    end_year: int | None,
) -> str | None:
    """Return an issue if the program couldn't have existed at that institution by end_year."""
    if end_year is None:
        return None
    inst_lower = institution.lower()
    fos_lower  = field_of_study.lower()

    for inst_key, programs in _INSTITUTION_PROGRAMS.items():
        if inst_key in inst_lower:
            for prog_key, earliest in programs.items():
                if prog_key in fos_lower:
                    if end_year < earliest:
                        return (
                            f"{institution} did not offer '{field_of_study}' "
                            f"before ~{earliest} (graduated {end_year})"
                        )
    return None


def _check_tier(institution: str, claimed_tier: str | None) -> str | None:
    """Return an issue if the claimed tier contradicts the known tier."""
    if not claimed_tier:
        return None
    inst_lower = institution.lower()
    for inst_key, actual_tier in _INSTITUTION_TIER.items():
        if inst_key in inst_lower:
            if claimed_tier != actual_tier:
                return (
                    f"{institution} is known to be {actual_tier} "
                    f"but profile claims {claimed_tier}"
                )
    return None


# ---------------------------------------------------------------------------
# 3. Career–education timeline consistency
# ---------------------------------------------------------------------------
def _check_timeline(education: list[dict], career: list[dict]) -> str | None:
    """
    The earliest career start date should not predate the most recent education
    end year by more than 1 year (internships are fine, so allow 1 year gap).
    """
    end_years = [e.get("end_year") for e in education if e.get("end_year")]
    if not end_years:
        return None
    latest_edu_end = max(end_years)

    career_starts = []
    for c in career:
        sd = c.get("start_date", "")
        if sd and len(sd) >= 4:
            try:
                career_starts.append(int(sd[:4]))
            except ValueError:
                pass
    if not career_starts:
        return None

    earliest_career = min(career_starts)
    if earliest_career < latest_edu_end - 1:
        return (
            f"Career started {earliest_career} but latest education ended {latest_edu_end} "
            f"— timeline overlap suggests fabricated or backdated entry"
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate_education(candidate: dict) -> list[str]:
    """
    Run all education checks on a candidate dict.

    Returns:
        List of issue strings. Empty = no problems detected.
    """
    issues: list[str] = []
    education: list[dict] = candidate.get("education", [])
    career:    list[dict] = candidate.get("career_history", [])

    for edu in education:
        degree       = edu.get("degree", "")
        institution  = edu.get("institution", "")
        field        = edu.get("field_of_study", "")
        start_year   = edu.get("start_year")
        end_year     = edu.get("end_year")
        claimed_tier = edu.get("tier")

        issue = _check_duration(degree, start_year, end_year)
        if issue:
            issues.append(issue)

        issue = _check_program_year(institution, field, end_year)
        if issue:
            issues.append(issue)

        issue = _check_tier(institution, claimed_tier)
        if issue:
            issues.append(issue)

    timeline_issue = _check_timeline(education, career)
    if timeline_issue:
        issues.append(timeline_issue)

    return issues
