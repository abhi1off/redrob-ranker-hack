import argparse
import json

from pathlib import Path
from src.disqualifier import remove_disqualified_candidates
from src.profile_integrity_scorer import ProfileVerifier
from src.bot_signature_map import collect_bot_signatures

def build_candidates(candidates_path: Path):
    print("starting build candidates...", candidates_path)
    
    with open("src/disqualifier_conditions.json", "r") as f:
        config = json.load(f)
    
    remove_disqualified_candidates(str(candidates_path), config)

    bot_signatures = collect_bot_signatures("resources/candidates_qualified.jsonl")

    verifier = ProfileVerifier("src/profile_integrity_conditions.json",bot_signatures)
    results  = verifier.score_all("resources/candidates_qualified.jsonl")


def rank_candidates(jd_path: Path, output_path: Path):
    print("starting rank candidates...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
 
    p_build = sub.add_parser("build", help="candidates.jsonl")
    p_build.add_argument("--candidates", default="resources/candidates.jsonl")
 
    p_rank = sub.add_parser("rank", help="Rank candidates for a JD")
    p_rank.add_argument("--jd",     default="resources/job_description.md")
    p_rank.add_argument("--output", default="output/results.csv")
 
    args = parser.parse_args()
 
    if args.cmd == "build":
        build_candidates(Path(args.candidates))
    elif args.cmd == "rank":
        rank_candidates(Path(args.output))
    else:
        parser.print_help()
