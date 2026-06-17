"""
profile_verifier.py — ProfileVerifier

Step 2 of the pipeline: sits between hard disqualification and ML embeddings.

Takes qualified_candidates.jsonl (output of disqualifier),
applies integrity + credibility rules from integrity_rules.json,
and outputs a pre-score map:  { candidate_id → { score, flags, penalties } }

Every candidate starts at 1.0. Rules deduct penalties. Floor is 0.0.
This score is later used as a MULTIPLIER on the ML semantic score.

Usage:
    python -m src.profile_verifier \
        --input  data/qualified_candidates.jsonl \
        --output data/credibility_scores.json \
        --rules  data/integrity_rules.json

    # Or use as a library:
    from src.profile_integrity_scorer import ProfileVerifier
    verifier = ProfileVerifier("src/profile_integrity_conditions.json")
    results  = verifier.score_all("resources/candidates_qualified.jsonl")
"""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Any


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PenaltyEvent:
    rule_id:   str
    severity:  str          # high / medium / low / info
    penalty:   float
    message:   str


@dataclass
class VerificationResult:
    candidate_id: str
    pre_score:    float                          # 0.0 – 1.0
    penalties:    list[PenaltyEvent] = field(default_factory=list)
    flags:        list[str]          = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "pre_score":    round(self.pre_score, 4),
            "total_deducted": round(1.0 - self.pre_score, 4),
            "penalty_count": len(self.penalties),
            "flags": self.flags,
            "penalty_detail": [
                {
                    "rule_id":  p.rule_id,
                    "severity": p.severity,
                    "penalty":  p.penalty,
                    "message":  p.message,
                }
                for p in self.penalties
            ],
        }


# ─── ProfileVerifier ──────────────────────────────────────────────────────────

class ProfileVerifier:
    def __init__(self, rules_path, bot_signatures: set = None):
        self.bot_signatures = bot_signatures or set()
        self.rules_path = Path(rules_path)
        with open(self.rules_path, encoding="utf-8") as f:
            self.config = json.load(f)

        self.tech_release_years: dict[str, int] = self.config.get("technology_release_years", {})
        self.integrity_rules:    list[dict]     = self.config.get("integrity_rules", [])
        self.credibility_rules:  list[dict]     = self.config.get("credibility_rules", [])
        self._current_year = datetime.now().year

    # ── Public API ────────────────────────────────────────────────────────────

    def score_candidate(self, cand: dict) -> VerificationResult:
        result = VerificationResult(candidate_id=cand["candidate_id"], pre_score=1.0)

        # Bot phrase check first — uses dynamic signatures
        if self.bot_signatures:
            self._check_bot_phrases(cand, result)

        for rule in self.integrity_rules:
            self._apply_rule(rule, cand, result)

        for rule in self.credibility_rules:
            self._apply_rule(rule, cand, result)

        result.pre_score = max(0.0, result.pre_score)
        return result

    def score_all(self, jsonl_path: str | Path) -> dict[str, dict]:
        """
        Score every candidate in a .jsonl file.
        Returns { candidate_id: VerificationResult.to_dict() }
        """
        results = {}
        path = Path(jsonl_path)
        total = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cand = json.loads(line)
                vr = self.score_candidate(cand)
                results[vr.candidate_id] = vr.to_dict()
                total += 1

        print(f"[ProfileVerifier] Scored {total} candidates from {path.name}")
        self._print_summary(results)

        output_file = Path("resources/candidates_scored.jsonl")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # FIX: Iterate over results.values() to dump only the internal dictionaries
        with open(output_file, "w", encoding="utf-8") as f:
            for candidate_data in results.values():
                f.write(json.dumps(candidate_data) + "\n")
        
        return results

    # ── Rule dispatcher ───────────────────────────────────────────────────────

    def _apply_rule(self, rule: dict, cand: dict, result: VerificationResult):
        check = rule["check"]
        handler = self._HANDLERS.get(check)
        if handler is None:
            return   # unknown check — skip silently
        handler(self, rule, cand, result)

    # ── Individual check implementations ──────────────────────────────────────

    def _check_cert_impossible_date(self, rule, cand, result):
        for cert in cand.get("certifications", []):
            name = cert.get("name", "")
            year = cert.get("year")
            if not year:
                continue
            # Check each known tech keyword in cert name
            for tech, release_year in self.tech_release_years.items():
                if tech.lower() in name.lower() and int(year) < release_year:
                    msg = rule["flag_message"].format(
                        name=name, year=year, release_year=release_year
                    )
                    self._deduct(result, rule, msg)
                    break   # one penalty per cert max

    def _check_skill_duration_gt_total_career_months(self, rule, cand, result):
        career_months = self._total_career_months(cand)
        if career_months <= 0:
            return
        penalties = 0
        max_penalty = rule.get("max_penalty", rule["penalty"])

        for skill in cand.get("skills", []):
            dur = skill.get("duration_months", 0) or 0
            if dur > career_months + 3:   # 3-month tolerance
                if rule.get("per_occurrence") and penalties * rule["penalty"] >= max_penalty:
                    break
                msg = rule["flag_message"].format(
                    name=skill["name"], duration_months=dur, career_months=career_months
                )
                self._deduct(result, rule, msg)
                penalties += 1

    def _check_claimed_yoe_minus_computed_yoe_gt_threshold(self, rule, cand, result):
        claimed = float(cand.get("profile", {}).get("years_of_experience", 0) or 0)
        computed = self._total_career_months(cand) / 12.0
        threshold = rule.get("threshold", 2.0)
        if claimed - computed > threshold:
            msg = rule["flag_message"].format(
                claimed_yoe=round(claimed, 1), computed_yoe=round(computed, 1)
            )
            self._deduct(result, rule, msg)

    def _check_career_start_before_last_degree_end(self, rule, cand, result):
        edu_list = cand.get("education", [])
        career   = cand.get("career_history", [])
        if not edu_list or not career:
            return

        last_grad_year = max(
            (e.get("end_year") or 0) for e in edu_list
        )
        if not last_grad_year:
            return

        tolerance_months = rule.get("overlap_tolerance_months", 6)
        earliest_career_date = None
        earliest_company = ""

        for job in career:
            start_str = job.get("start_date", "")
            if not start_str:
                continue
            try:
                start = datetime.strptime(start_str[:10], "%Y-%m-%d").date()
                if earliest_career_date is None or start < earliest_career_date:
                    earliest_career_date = start
                    earliest_company = job.get("company", "")
            except ValueError:
                continue

        if earliest_career_date is None:
            return

        # Compare: career start more than tolerance_months before grad year Jan
        grad_cutoff = date(last_grad_year, 1, 1)
        months_before = (grad_cutoff.year - earliest_career_date.year) * 12 + \
                        (grad_cutoff.month - earliest_career_date.month)

        if months_before > tolerance_months:
            msg = rule["flag_message"].format(
                company=earliest_company,
                start_date=str(earliest_career_date),
                grad_year=last_grad_year
            )
            self._deduct(result, rule, msg)

    def _check_degree_duration_below_minimum(self, rule, cand, result):
        minimums = rule.get("degree_minimums", {})
        penalties = 0
        max_penalty = rule.get("max_penalty", rule["penalty"])

        for edu in cand.get("education", []):
            degree    = edu.get("degree", "")
            start_yr  = edu.get("start_year")
            end_yr    = edu.get("end_year")
            if not (start_yr and end_yr):
                continue
            actual_years = end_yr - start_yr
            min_years    = minimums.get(degree)
            if min_years and actual_years < min_years:
                if penalties * rule["penalty"] >= max_penalty:
                    break
                msg = rule["flag_message"].format(
                    degree=degree,
                    institution=edu.get("institution", ""),
                    actual_years=actual_years,
                    min_years=min_years
                )
                self._deduct(result, rule, msg)
                penalties += 1

    def _check_education_timeline_overlap(self, rule, cand, result):
        edu_list = cand.get("education", [])
        if len(edu_list) < 2:
            return
        # Sort by start year
        sorted_edu = sorted(edu_list, key=lambda e: e.get("start_year") or 0)
        for i in range(len(sorted_edu) - 1):
            a = sorted_edu[i]
            b = sorted_edu[i + 1]
            end_a   = a.get("end_year") or 0
            start_b = b.get("start_year") or 9999
            if end_a > start_b:
                msg = rule["flag_message"].format(
                    degree_a=a.get("degree", ""),
                    end_a=end_a,
                    degree_b=b.get("degree", ""),
                    start_b=start_b
                )
                self._deduct(result, rule, msg)

    def _check_assessment_score_below_threshold_for_proficiency(self, rule, cand, result):
        assessments   = cand.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
        thresholds    = rule.get("thresholds", {"advanced": 65, "expert": 75})
        penalties     = 0
        max_penalty   = rule.get("max_penalty", rule["penalty"])

        for skill in cand.get("skills", []):
            name        = skill.get("name", "")
            proficiency = skill.get("proficiency", "").lower()
            score       = assessments.get(name)

            if score is None:
                continue   # no assessment taken — handled by credibility rule

            min_score = thresholds.get(proficiency)
            if min_score and float(score) < min_score:
                if penalties * rule["penalty"] >= max_penalty:
                    break
                msg = rule["flag_message"].format(
                    name=name, proficiency=proficiency,
                    score=float(score), threshold=min_score
                )
                self._deduct(result, rule, msg)
                penalties += 1

    def _check_bot_phrases(self, cand, result):
        summary = " ".join((cand.get("profile", {}).get("summary") or "").split())
        matches = [sig for sig in self.bot_signatures if sig in summary]
        count   = len(matches)
        if count == 0:
            return
        
        # Graduated penalty from rules JSON
        rule      = next((r for r in self.integrity_rules if r["id"] == "bot_signature_summary"), None)
        thresholds = rule.get("thresholds", {"1": 0.05, "2": 0.12, "3": 0.25}) if rule else {}
        max_penalty = float(rule.get("max_penalty", 0.25)) if rule else 0.25
        
        penalty_key = str(min(count, max(int(k) for k in thresholds.keys())))
        penalty     = min(float(thresholds.get(penalty_key, 0.25)), max_penalty)
        
        result.pre_score -= penalty
        result.flags.append(f"[bot_signature_summary] {count} template phrase(s) in summary.")

    def _check_skill_duration_gt_relevant_company_tenure(self, rule, cand, result):
        """
        For each skill, find the maximum single-company tenure and compare.
        Rationale: a skill can't realistically have been used for longer than
        the longest job the candidate held (with some tolerance).
        """
        career   = cand.get("career_history", [])
        if not career:
            return
        max_tenure       = max((j.get("duration_months") or 0) for j in career)
        tolerance        = rule.get("tolerance_months", 6)
        penalties        = 0
        max_penalty      = rule.get("max_penalty", rule["penalty"])

        for skill in cand.get("skills", []):
            dur = skill.get("duration_months", 0) or 0
            if dur > max_tenure + tolerance:
                if penalties * rule["penalty"] >= max_penalty:
                    break
                msg = rule["flag_message"].format(
                    name=skill["name"],
                    duration_months=dur,
                    max_tenure=max_tenure
                )
                self._deduct(result, rule, msg)
                penalties += 1

    def _check_headline_current_title_vs_career_history_title_mismatch(self, rule, cand, result):
        profile_title = (cand.get("profile", {}).get("current_title") or "").lower().strip()
        current_jobs  = [j for j in cand.get("career_history", []) if j.get("is_current")]
        if not current_jobs or not profile_title:
            return
        history_title = current_jobs[0].get("title", "").lower().strip()
        # Simple token overlap check — not embedding-based to keep it fast
        p_tokens = set(profile_title.split())
        h_tokens = set(history_title.split())
        overlap  = len(p_tokens & h_tokens) / max(len(p_tokens | h_tokens), 1)
        if overlap < 0.3:   # less than 30% word overlap
            msg = rule["flag_message"].format(
                profile_title=profile_title, history_title=history_title
            )
            self._deduct(result, rule, msg)

    # ── Credibility checks ────────────────────────────────────────────────────

    def _check_verified_email_false_and_verified_phone_false(self, rule, cand, result):
        signals = cand.get("redrob_signals", {})
        if not signals.get("verified_email") and not signals.get("verified_phone"):
            self._deduct(result, rule, rule["flag_message"])

    def _check_advanced_skill_zero_endorsements_no_assessment(self, rule, cand, result):
        assessments = cand.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
        penalties   = 0
        max_penalty = rule.get("max_penalty", rule["penalty"])

        for skill in cand.get("skills", []):
            name        = skill.get("name", "")
            proficiency = skill.get("proficiency", "").lower()
            endorsements = skill.get("endorsements", 0) or 0
            if proficiency in ("advanced", "expert") and \
               endorsements == 0 and \
               name not in assessments:
                if penalties * rule["penalty"] >= max_penalty:
                    break
                msg = rule["flag_message"].format(name=name, proficiency=proficiency)
                self._deduct(result, rule, msg)
                penalties += 1

    def _check_all_skills_identical_proficiency(self, rule, cand, result):
        skills      = cand.get("skills", [])
        min_count   = rule.get("min_skill_count", 5)
        if len(skills) < min_count:
            return
        proficiencies = [s.get("proficiency", "") for s in skills if s.get("proficiency")]
        if len(set(proficiencies)) == 1:
            msg = rule["flag_message"].format(
                count=len(proficiencies), proficiency=proficiencies[0]
            )
            self._deduct(result, rule, msg)

    def _check_summary_title_domain_embedding_distance(self, rule, cand, result):
        """
        Lightweight keyword-based domain check (no embedding at this stage).
        Flags if summary mentions a completely different domain than the title.
        """
        profile = cand.get("profile", {})
        summary = (profile.get("summary") or "").lower()
        title   = (profile.get("current_title") or "").lower()

        domain_keywords = {
            "engineering": {"engineer", "developer", "coding", "software", "backend", "frontend"},
            "data_science": {"data science", "machine learning", "ml", "model", "analytics"},
            "management": {"manager", "management", "strategy", "operations", "executive"},
            "sales": {"sales", "revenue", "pipeline", "crm", "quota", "leads"},
            "support": {"support", "customer", "tickets", "helpdesk", "escalation"},
        }

        def detect_domain(text):
            for domain, keywords in domain_keywords.items():
                if any(kw in text for kw in keywords):
                    return domain
            return "unknown"

        title_domain   = detect_domain(title)
        summary_domain = detect_domain(summary)

        if title_domain != "unknown" and summary_domain != "unknown" \
                and title_domain != summary_domain:
            msg = rule["flag_message"].format(current_title=title)
            self._deduct(result, rule, msg)

    def _check_profile_completeness_below_threshold(self, rule, cand, result):
        score     = cand.get("redrob_signals", {}).get("profile_completeness_score", 100) or 100
        threshold = rule.get("threshold", 40)
        if float(score) < threshold:
            msg = rule["flag_message"].format(score=score, threshold=threshold)
            self._deduct(result, rule, msg)

    def _check_applications_gt_threshold_and_response_rate_lt_threshold(self, rule, cand, result):
        signals = cand.get("redrob_signals", {})
        apps    = signals.get("applications_submitted_30d", 0) or 0
        rr      = signals.get("recruiter_response_rate", 1.0)
        if rr < 0:
            rr = 1.0
        app_thresh = rule.get("application_threshold", 15)
        rr_thresh  = rule.get("response_rate_threshold", 0.20)
        if apps > app_thresh and rr < rr_thresh:
            msg = rule["flag_message"].format(apps=apps, rr=rr)
            self._deduct(result, rule, msg)

    def _check_single_company_career(self, rule, cand, result):
        career    = cand.get("career_history", [])
        min_years = rule.get("min_years", 3)
        if len(career) != 1:
            return
        duration_years = (career[0].get("duration_months") or 0) / 12
        if duration_years >= min_years:
            msg = rule["flag_message"].format(
                years=round(duration_years, 1),
                company=career[0].get("company", "")
            )
            self._deduct(result, rule, msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _deduct(self, result: VerificationResult, rule: dict, message: str):
        penalty = rule["penalty"]
        result.pre_score -= penalty
        result.penalties.append(PenaltyEvent(
            rule_id  = rule["id"],
            severity = rule.get("severity", "medium"),
            penalty  = penalty,
            message  = message,
        ))
        result.flags.append(f"[{rule['id']}] {message}")

    def _total_career_months(self, cand: dict) -> int:
        return sum(
            (j.get("duration_months") or 0)
            for j in cand.get("career_history", [])
        )

    def _print_summary(self, results: dict):
        scores  = [r["pre_score"] for r in results.values()]
        flagged = sum(1 for r in results.values() if r["penalty_count"] > 0)
        perfect = sum(1 for r in results.values() if r["pre_score"] == 1.0)
        low     = sum(1 for r in results.values() if r["pre_score"] < 0.6)
        print(f"  Perfect (1.0):   {perfect}")
        print(f"  Flagged:         {flagged}")
        print(f"  Low (<0.6):      {low}")
        if scores:
            print(f"  Avg pre_score:   {sum(scores)/len(scores):.3f}")
            print(f"  Min pre_score:   {min(scores):.3f}")

    # ── Handler registry ──────────────────────────────────────────────────────

    _HANDLERS = {
        "cert_year_before_tech_release":
            _check_cert_impossible_date,
        "skill_duration_gt_total_career_months":
            _check_skill_duration_gt_total_career_months,
        "claimed_yoe_minus_computed_yoe_gt_threshold":
            _check_claimed_yoe_minus_computed_yoe_gt_threshold,
        "career_start_before_last_degree_end":
            _check_career_start_before_last_degree_end,
        "degree_duration_below_minimum":
            _check_degree_duration_below_minimum,
        "education_timeline_overlap":
            _check_education_timeline_overlap,
        "assessment_score_below_threshold_for_proficiency":
            _check_assessment_score_below_threshold_for_proficiency,
        "skill_duration_gt_relevant_company_tenure":
            _check_skill_duration_gt_relevant_company_tenure,
        "headline_current_title_vs_career_history_title_mismatch":
            _check_headline_current_title_vs_career_history_title_mismatch,
        "verified_email_false_and_verified_phone_false":
            _check_verified_email_false_and_verified_phone_false,
        "advanced_skill_zero_endorsements_no_assessment":
            _check_advanced_skill_zero_endorsements_no_assessment,
        "all_skills_identical_proficiency":
            _check_all_skills_identical_proficiency,
        "summary_title_domain_embedding_distance":
            _check_summary_title_domain_embedding_distance,
        "profile_completeness_below_threshold":
            _check_profile_completeness_below_threshold,
        "applications_gt_threshold_and_response_rate_lt_threshold":
            _check_applications_gt_threshold_and_response_rate_lt_threshold,
        "single_company_career":
            _check_single_company_career,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ProfileVerifier — score qualified candidates before ML pipeline."
    )
    parser.add_argument("--input",  default="data/qualified_candidates.jsonl",
                        help="Qualified candidates JSONL file")
    parser.add_argument("--output", default="data/credibility_scores.json",
                        help="Output JSON: { candidate_id → score + flags }")
    parser.add_argument("--rules",  default="data/integrity_rules.json",
                        help="Integrity + credibility rules JSON")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Optionally hard-drop candidates below this pre_score")
    args = parser.parse_args()

    verifier = ProfileVerifier(args.rules)
    results  = verifier.score_all(args.input)

    # Optional: drop candidates below min_score threshold
    if args.min_score > 0:
        before = len(results)
        results = {k: v for k, v in results.items() if v["pre_score"] >= args.min_score}
        print(f"[ProfileVerifier] Dropped {before - len(results)} candidates below "
              f"pre_score {args.min_score}. Remaining: {len(results)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[ProfileVerifier] Scores written → {out_path}")


if __name__ == "__main__":
    main()