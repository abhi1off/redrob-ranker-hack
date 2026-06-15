from datetime import datetime

def remove_disqualified_candidates(candidates_path):
    # Remove the candidates not matching basic requirements
    # should create a new list with only the filetered candidates, DO NOT EDIT MAIN RESOURCE
    print("remove_disqualified_candidates::Starting",candidates_path)

def is_disqualified_by_conditions(candidate_json):
    # Helper for each candidate
    # Can return only "bool" to keep or remove in the list
    print("is_disqualified_by_conditions::Starting",candidate_json)
    signals = candidate_json.get("redrob_signals", {})
