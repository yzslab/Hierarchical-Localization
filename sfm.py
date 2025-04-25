import os
import argparse
import json
import pycolmap

from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    pairs_from_retrieval,
    pairs_from_exhaustive,
    pairs_from_poses,
    pairs_from_covisibility,
)

parser = argparse.ArgumentParser()
parser.add_argument("--image-dir", "-i", required=True)
# parser.add_argument("--mask-dir", "-m", default=None)  # TODO: mask
parser.add_argument("--output-dir", "-o", required=True)
parser.add_argument("--camera-model", "-c", default="OPENCV")
parser.add_argument("--num-matched", "-n", type=int, default=32)
parser.add_argument("--feature-conf", type=str, default="aliked-n16", help=", ".join(list(extract_features.confs.keys())))
parser.add_argument("--resize", type=int, default=None)
parser.add_argument("--matching-method", type=str, default="netvlad")
parser.add_argument("--matcher-conf", type=str, default="aliked+lightglue", help=", ".join(list(match_features.confs.keys())))
parser.add_argument("--fast-lightglue", action="store_true", default=False)
parser.add_argument("--mapper-min-num-matches", type=int, default=32)
parser.add_argument("--match-threshold", type=float, default=0.85)
parser.add_argument("--sparse-model-dir", required=False, default=None)
parser.add_argument("--image-list", type=str, required=False, default=None)
parser.add_argument("--single-camera", action="store_true", default=False)
parser.add_argument("--pair-min-score", type=float, default=0.25)
parser.add_argument("--import-only", action="store_true", default=False)
parser.add_argument("--extract-only", action="store_true", default=False)
parser.add_argument("--skip-extraction", action="store_true", default=False)
args = parser.parse_args()

# get configs
retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs[args.feature_conf]
matcher_conf = match_features.confs[args.matcher_conf]

if args.match_threshold is not None:
    if matcher_conf["model"]["name"] == "superglue":
        matcher_conf["model"]["match_threshold"] = args.match_threshold
    elif matcher_conf["model"]["name"] == "lightglue":
        matcher_conf["model"]["filter_threshold"] = args.match_threshold

# avoid false matching
if not args.fast_lightglue and matcher_conf["model"]["name"] == "lightglue":
    matcher_conf["model"]["depth_confidence"] = -1
    matcher_conf["model"]["width_confidence"] = -1

if args.resize is not None:
    feature_conf["preprocessing"]["resize_max"] = args.resize

print("feature_conf={}".format(feature_conf))
print("matcher_conf={}".format(matcher_conf))

# setup dirs

image_path = Path(args.image_dir)
# mask_path = args.mask_dir
# if mask_path is not None:
# mask_path = Path(mask_path)
output_path = Path(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

images = image_path
outputs = output_path
sfm_pairs = outputs / 'pairs.txt'
sfm_dir = outputs / 'sfm_{}+{}'.format(feature_conf["model"]["name"], matcher_conf["model"]["name"])

sfm_image_list = None
if args.image_list is not None:
    sfm_image_list = []
    with open(args.image_list, "r") as f:
        for row in f:
            row = row.rstrip("\n")
            if row == "":
                continue
            sfm_image_list.append(Path(row))

if not args.skip_extraction:
    # pipeline
    print("Building image pairs...")
    if args.matching_method == "netvlad" or args.matching_method == "netvlad+sift":
        # Find image pairs via image retrieval
        """
        We extract global descriptors with NetVLAD and find for each image the most similar ones. 
        For smaller dataset we can instead use exhaustive matching via hloc/pairs_from_exhaustive.py, 
        which would find n(n-1)/2 images pairs
        """
        retrieval_path = extract_features.main(retrieval_conf, images, outputs, image_list=sfm_image_list)
        netvlad_pairs = pairs_from_retrieval.main(
            retrieval_path,
            sfm_pairs,
            num_matched=args.num_matched,
            min_score=args.pair_min_score,
        )
    elif args.matching_method == "exhaustive":
        if sfm_image_list is not None:
            image_list = sfm_image_list
        else:
            image_list = list(images.rglob("*.jpg")) + \
                list(images.rglob("*.JPG")) + \
                list(images.rglob("*.jpeg")) + \
                list(images.rglob("*.JPEG"))

        pairs_from_exhaustive.main(
            output=sfm_pairs,
            image_list=[str(i).replace(str(images), "").strip("/") for i in image_list],
        )

    if args.matching_method == "sift" or args.matching_method == "netvlad+sift":
        sift_pairs = pairs_from_covisibility.main(
            model=Path(args.sparse_model_dir),
            output=sfm_pairs,
            num_matched=args.num_matched,
        )
    elif args.matching_method == "sfm":
        pairs_from_poses.main(
            model=Path(args.sparse_model_dir),
            output=sfm_pairs,
            num_matched=args.num_matched,
        )
    # else:
    #     raise ValueError("unsupported matching method {}".format(args.matching_method))

    if args.matching_method == "netvlad+sift":
        valid_pairs = {}
        for i in netvlad_pairs:
            valid_pairs[tuple(sorted(i))] = True
        for i in sift_pairs:
            valid_pairs[tuple(sorted(i))] = True
        with open(sfm_pairs, "w") as f:
            for i in sorted(list(valid_pairs.keys())):
                f.write("{} {}\n".format(*i))

    # Extract and match local features
    print("Extracting features...")
    feature_path = extract_features.main(feature_conf, images, outputs, image_list=sfm_image_list)
else:
    feature_path = Path(output_path, feature_conf["output"] + ".h5")

if args.extract_only:
    with open(outputs / "matching.json", "w") as f:
        json.dump({
            "conf": matcher_conf,
            "feature": feature_conf['output'],
        }, f, indent=4, ensure_ascii=False)
    print(outputs / "matching.json")
    exit(0)

print("Matching...")
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

# 3D reconstruction
model = reconstruction.main(
    sfm_dir,
    images,
    sfm_pairs,
    feature_path,
    match_path,
    image_list=sfm_image_list,
    camera_mode=pycolmap.CameraMode.SINGLE if args.single_camera else pycolmap.CameraMode.AUTO,
    image_options=dict(camera_model=args.camera_model),
    verbose=True,
    mapper_options={
        "min_num_matches": args.mapper_min_num_matches,
    },
    import_only=args.import_only,
)
if args.import_only:
    os.makedirs(sfm_dir, exist_ok=True)
    print(" \\\n    ".join([
        "colmap",
        "mapper",
        "--database_path={}".format(sfm_dir / "database.db"),
        "--image_path={}".format(images),
        "--output_path={}".format(sfm_dir),
        "--Mapper.min_num_matches={}".format(args.mapper_min_num_matches),
        "--Mapper.ba_use_gpu=1",
    ]))
else:
    print(sfm_dir)
