import os
import sys
import pycolmap

with pycolmap.ostream():
    pycolmap.verify_matches(
        os.path.join(sys.argv[1], "sfm_aliked+lightglue", "database.db"),
        os.path.join(sys.argv[1], "pairs.txt"),
        options=dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.1)),
    )
