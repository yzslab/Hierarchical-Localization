import os
from pathlib import Path
from hloc import match_features, distibuted_tasks


import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("json")
distibuted_tasks.configure_arg_parser_v2(parser)
args = parser.parse_args()


with open(args.json, "r") as f:
    conf = json.load(f)

output_dir = Path(os.path.dirname(args.json))
sfm_pairs = output_dir / "pairs.txt"
match_features.main(
    conf["conf"],
    sfm_pairs,
    conf["feature"],
    output_dir,
    world_size=args.n_processes,
    global_rank=args.process_id,
)

# print("python merge_matches.py {}".format(output_dir))
