import os
import sys
import pycolmap

pairs = os.path.join(sys.argv[1], "imported_pairs.txt")
if not os.path.exists(pairs):
    pairs = os.path.join(sys.argv[1], "pairs.txt")
print(pairs)
with pycolmap.ostream():
    pycolmap.verify_matches(
        os.path.join(sys.argv[1], "sfm_aliked+lightglue", "database.db"),
        pairs,
        options=dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.1)),
    )
