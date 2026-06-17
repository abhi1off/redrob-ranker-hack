import re
from datetime import date, datetime
import json

def get_field(candidate, field):

    if field == "last_active_days":
        raw = candidate["redrob_signals"].get("last_active_date")
        if raw is None:
            return None
        return (date.today() - datetime.strptime(raw, "%Y-%m-%d").date()).days
    
    mapping = {
        "years_of_experience":
            candidate["profile"]["years_of_experience"],

        "country":
            candidate["profile"]["country"],

        "location": 
            candidate["profile"]["location"].split(",")[0].strip().lower(),

        "current_title":
            candidate["profile"]["current_title"],

        "willing_to_relocate":
            candidate["redrob_signals"]["willing_to_relocate"],

        "open_to_work":
            candidate["redrob_signals"]["open_to_work_flag"],

        "applications_submitted_30d":
            candidate["redrob_signals"]["applications_submitted_30d"],

        "avg_response_time_hours":
            candidate["redrob_signals"]["avg_response_time_hours"],

        
    }

    return mapping.get(field)

def evaluate(value, operator, expected, rule=None):

    if value is None:
        return False
    
    if operator == "==":
        if isinstance(value, str) and isinstance(expected, str):
            return value.lower() == expected.lower()
        return value == expected

    if operator == "!=":
        if isinstance(value, str) and isinstance(expected, str):
            return value.lower() != expected.lower()
        return value != expected

    if operator == "<":
        return value < expected

    if operator == "<=":
        return value <= expected

    if operator == ">":
        return value > expected

    if operator == ">=":
        return value >= expected

    if operator == "IN":
        if isinstance(value, str):
            return value.lower() in {str(x).lower() for x in expected}
        return value in expected

    if operator == "NOT_IN":
        if isinstance(value, str):
            return value.lower() not in {str(x).lower() for x in expected}
        return value not in expected

    if operator == "REGEX":
        if re.search(expected, str(value), re.I) is not None:
            if rule and "exceptions" in rule:
                if any(eng_word in str(value) for eng_word in rule["exceptions"]):
                    return False
            return True     
    return False

    raise ValueError(f"Unsupported operator {operator}")

def all_companies_in_service_list(candidate, allowed):
    companies = set()

    # current company from profile
    current = candidate["profile"].get("current_company")
    if current:
        companies.add(current.strip().lower())

    # every company from career history
    for job in candidate.get("career_history", []):
        company = job.get("company")
        if company:
            companies.add(company.strip().lower())

    if not companies:
        return False  # no company data → don't disqualify

    allowed_set = {c.lower() for c in allowed}
    return all(company in allowed_set for company in companies)

def check_rule_group(candidate, rules):
    for rule in rules:
        # ALL_IN is special: needs both profile + career_history
        if rule.get("operator") == "ALL_IN":
            fired = all_companies_in_service_list(candidate, rule["value"])

        elif "field" in rule:
            value = get_field(candidate, rule["field"])
            fired = evaluate(value, rule["operator"], rule["value"])

        elif "conditions" in rule:
            results = [
                evaluate(get_field(candidate, c["field"]), c["operator"], c["value"])
                for c in rule["conditions"]
            ]
            fired = all(results) if rule["logic"] == "AND" else any(results)

        else:
            continue

        if fired:
            return True, rule["action"]

    return False, None

def is_disqualified_by_conditions(candidate, config):
    # 1. Hard disqualification rules
    fired, action = check_rule_group(candidate, config["disqualification_rules"])
    if fired:
        return True

    # 2. Activity rules — first match wins
    fired, action = check_rule_group(candidate, config["activity_rules"])
    if fired:
        return action == "DISQUALIFY"

    return False  # default: keep

def remove_disqualified_candidates(candidates_path, config):
    print("remove_disqualified_candidates::Starting", candidates_path)

    qualified_path = candidates_path.replace(".jsonl", "_qualified.jsonl")
    disqualified_path = candidates_path.replace(".jsonl", "_disqualified.jsonl")

    qualified_count = 0
    disqualified_count = 0


    with open(candidates_path, "r", encoding="utf-8") as infile, \
         open(qualified_path, "w", encoding="utf-8") as qfile, \
         open(disqualified_path, "w", encoding="utf-8") as dfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue

            if is_disqualified_by_conditions(candidate, config):
                dfile.write(line + "\n")
                disqualified_count += 1
            else:
                qfile.write(line + "\n")
                qualified_count += 1

    print(f"remove_disqualified_candidates::Done")
    print(f"  Qualified    → {qualified_path} ({qualified_count})")
    print(f"  Disqualified → {disqualified_path} ({disqualified_count})")

    return qualified_path, disqualified_path