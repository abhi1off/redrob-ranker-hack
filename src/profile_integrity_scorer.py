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
    from src.profile_verifier import ProfileVerifier
    verifier = ProfileVerifier("data/integrity_rules.json")
    results  = verifier.score_all("data/qualified_candidates.jsonl")
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PenaltyEvent:
    rule_id:  str
    severity: str
    penalty:  float
    message:  str


@dataclass
class VerificationResult:
    candidate_id: str
    pre_score:    float
    penalties:    list[PenaltyEvent] = field(default_factory=list)
    flags:        list[str]          = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id":   self.candidate_id,
            "pre_score":      round(self.pre_score, 4),
            "total_deducted": round(1.0 - self.pre_score, 4),
            "penalty_count":  len(self.penalties),
            "flags":          self.flags,
            "penalty_detail": [
                {"rule_id": p.rule_id, "severity": p.severity,
                 "penalty": p.penalty, "message": p.message}
                for p in self.penalties
            ],
        }


# ─── ProfileVerifier ──────────────────────────────────────────────────────────

class ProfileVerifier:
    def __init__(self, rules_path: str):
        with open(rules_path, encoding="utf-8") as f:
            config = json.load(f)
        self.integrity_rules:  list[dict] = config.get("integrity_rules", [])
        self.credibility_rules: list[dict] = config.get("credibility_rules", [])

    # ── Public API ────────────────────────────────────────────────────────────

    def score_candidate(self, cand: dict) -> VerificationResult:
        result = VerificationResult(candidate_id=cand["candidate_id"], pre_score=1.0)
        for rule in self.integrity_rules + self.credibility_rules:
            handler = self._HANDLERS.get(rule["check"])
            if handler:
                handler(self, rule, cand, result)
        result.pre_score = max(0.0, result.pre_score)
        return result

    def score_all(self, jsonl_path: str) -> dict[str, dict]:
        results = {}
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cand = json.loads(line)
                vr = self.score_candidate(cand)
                results[vr.candidate_id] = vr.to_dict()

        print(f"[ProfileVerifier] Scored {len(results)} candidates")
        self._print_summary(results)

        output_file = Path("resources/candidates_scored.jsonl")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for candidate_data in results.values():
                f.write(json.dumps(candidate_data) + "\n")

        return results

    # ── Integrity checks ──────────────────────────────────────────────────────

    def _check_claimed_yoe_minus_computed_yoe_gt_threshold(self, rule, cand, result):
        claimed  = float(cand.get("profile", {}).get("years_of_experience") or 0)
        computed = self._total_career_months(cand) / 12.0
        if claimed - computed > rule.get("threshold", 2.0):
            self._deduct(result, rule, rule["flag_message"].format(
                claimed_yoe=round(claimed, 1), computed_yoe=round(computed, 1)
            ))

    def _check_degree_duration_below_minimum(self, rule, cand, result):
        minimums  = rule.get("degree_minimums", {})
        penalties = 0
        max_pen   = rule.get("max_penalty", rule["penalty"])
        for edu in cand.get("education", []):
            start, end = edu.get("start_year"), edu.get("end_year")
            if not (start and end):
                continue
            min_yrs = minimums.get(edu.get("degree", ""))
            actual  = end - start
            if min_yrs and actual < min_yrs:
                if penalties * rule["penalty"] >= max_pen:
                    break
                self._deduct(result, rule, rule["flag_message"].format(
                    degree=edu.get("degree", ""), institution=edu.get("institution", ""),
                    actual_years=actual, min_years=min_yrs
                ))
                penalties += 1

    def _check_degree_duration_above_maximum(self, rule, cand, result):
        maximums  = rule.get("degree_maximums", {})
        penalties = 0
        max_pen   = rule.get("max_penalty", rule["penalty"])
        for edu in cand.get("education", []):
            start, end = edu.get("start_year"), edu.get("end_year")
            if not (start and end):
                continue
            max_yrs = maximums.get(edu.get("degree", ""))
            actual  = end - start
            if max_yrs and actual > max_yrs:
                if penalties * rule["penalty"] >= max_pen:
                    break
                self._deduct(result, rule, rule["flag_message"].format(
                    degree=edu.get("degree", ""), institution=edu.get("institution", ""),
                    actual_years=actual, max_years=max_yrs
                ))
                penalties += 1

    def _check_education_timeline_overlap(self, rule, cand, result):
        edu_list = cand.get("education", [])
        if len(edu_list) < 2:
            return
        sorted_edu = sorted(edu_list, key=lambda e: e.get("start_year") or 0)
        for i in range(len(sorted_edu) - 1):
            a, b = sorted_edu[i], sorted_edu[i + 1]
            if (a.get("end_year") or 0) > (b.get("start_year") or 9999):
                self._deduct(result, rule, rule["flag_message"].format(
                    degree_a=a.get("degree", ""), end_a=a.get("end_year"),
                    degree_b=b.get("degree", ""), start_b=b.get("start_year")
                ))

    # ── Credibility checks ────────────────────────────────────────────────────

    def _check_last_active_date_lt_signup_date(self, rule, cand, result):
        signals = cand.get("redrob_signals", {})
        signup  = signals.get("signup_date")
        active  = signals.get("last_active_date")
        if not (signup and active):
            return
        try:
            if date.fromisoformat(active) < date.fromisoformat(signup):
                self._deduct(result, rule, rule["flag_message"].format(
                    last_active=active, signup=signup
                ))
        except ValueError:
            pass

    def _check_verified_email_false_and_verified_phone_false(self, rule, cand, result):
        signals = cand.get("redrob_signals", {})
        if not signals.get("verified_email") and not signals.get("verified_phone"):
            self._deduct(result, rule, rule["flag_message"])

    def _check_all_skills_identical_proficiency(self, rule, cand, result):
        skills = cand.get("skills", [])
        if len(skills) < rule.get("min_skill_count", 5):
            return
        profs = [s.get("proficiency", "") for s in skills if s.get("proficiency")]
        if len(set(profs)) == 1:
            self._deduct(result, rule, rule["flag_message"].format(
                count=len(profs), proficiency=profs[0]
            ))

    def _check_summary_title_domain_embedding_distance(self, rule, cand, result):
        profile = cand.get("profile", {})
        summary = (profile.get("summary") or "").lower()
        title   = (profile.get("current_title") or "").lower()

        domain_keywords = {
            "engineering": {"engineer", "developer", "coding", "software", "backend", "frontend"},
            "data_science": {"data science", "machine learning", "ml", "model", "analytics"},
            "management":  {"manager", "management", "strategy", "operations", "executive"},
            "sales":       {"sales", "revenue", "pipeline", "crm", "quota", "leads"},
            "support":     {"support", "customer", "tickets", "helpdesk", "escalation"},
        }

        def detect_domain(text):
            for domain, keywords in domain_keywords.items():
                if any(kw in text for kw in keywords):
                    return domain
            return "unknown"

        t_domain = detect_domain(title)
        s_domain = detect_domain(summary)
        if t_domain != "unknown" and s_domain != "unknown" and t_domain != s_domain:
            self._deduct(result, rule, rule["flag_message"].format(
                current_title=title
            ))

    def _check_profile_completeness_below_threshold(self, rule, cand, result):
        score = float(cand.get("redrob_signals", {}).get("profile_completeness_score") or 100)
        if score < rule.get("threshold", 40):
            self._deduct(result, rule, rule["flag_message"].format(
                score=score, threshold=rule.get("threshold", 40)
            ))

    def _check_applications_gt_threshold_and_response_rate_lt_threshold(self, rule, cand, result):
        signals = cand.get("redrob_signals", {})
        apps    = signals.get("applications_submitted_30d") or 0
        rr      = signals.get("recruiter_response_rate") or 1.0
        if rr < 0:
            rr = 1.0
        if apps > rule.get("application_threshold", 15) and rr < rule.get("response_rate_threshold", 0.20):
            self._deduct(result, rule, rule["flag_message"].format(apps=apps, rr=rr))

    def _check_single_company_career(self, rule, cand, result):
        career = cand.get("career_history", [])
        if len(career) != 1:
            return
        years = (career[0].get("duration_months") or 0) / 12
        if years >= rule.get("min_years", 3):
            self._deduct(result, rule, rule["flag_message"].format(
                years=round(years, 1), company=career[0].get("company", "")
            ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _deduct(self, result: VerificationResult, rule: dict, message: str):
        result.pre_score -= rule["penalty"]
        result.penalties.append(PenaltyEvent(
            rule_id=rule["id"], severity=rule.get("severity", "medium"),
            penalty=rule["penalty"], message=message,
        ))
        result.flags.append(f"[{rule['id']}] {message}")

    def _total_career_months(self, cand: dict) -> int:
        return sum((j.get("duration_months") or 0) for j in cand.get("career_history", []))

    def _print_summary(self, results: dict):
        scores = [r["pre_score"] for r in results.values()]
        print(f"  Perfect (1.0): {sum(1 for s in scores if s == 1.0)}")
        print(f"  Flagged:       {sum(1 for r in results.values() if r['penalty_count'] > 0)}")
        print(f"  Low (<0.6):    {sum(1 for s in scores if s < 0.6)}")
        if scores:
            print(f"  Avg pre_score: {sum(scores)/len(scores):.3f}")
            print(f"  Min pre_score: {min(scores):.3f}")

    # ── Handler registry ──────────────────────────────────────────────────────

    _HANDLERS = {
        "claimed_yoe_minus_computed_yoe_gt_threshold":
            _check_claimed_yoe_minus_computed_yoe_gt_threshold,
        "degree_duration_below_minimum":
            _check_degree_duration_below_minimum,
        "degree_duration_above_maximum":
            _check_degree_duration_above_maximum,
        "education_timeline_overlap":
            _check_education_timeline_overlap,
        "last_active_date_lt_signup_date":
            _check_last_active_date_lt_signup_date,
        "verified_email_false_and_verified_phone_false":
            _check_verified_email_false_and_verified_phone_false,
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
    parser.add_argument("--input",     default="data/qualified_candidates.jsonl")
    parser.add_argument("--output",    default="data/credibility_scores.json")
    parser.add_argument("--rules",     default="data/integrity_rules.json")
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    verifier = ProfileVerifier(args.rules)
    results  = verifier.score_all(args.input)

    if args.min_score > 0:
        before  = len(results)
        results = {k: v for k, v in results.items() if v["pre_score"] >= args.min_score}
        print(f"[ProfileVerifier] Dropped {before - len(results)} below {args.min_score}. "
              f"Remaining: {len(results)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    import json as _json
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(results, f, indent=2)
    print(f"[ProfileVerifier] Scores written → {out_path}")


if __name__ == "__main__":
    main()

