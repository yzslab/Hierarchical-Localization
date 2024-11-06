import os
import argparse
import pycolmap

from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, \
    pairs_from_exhaustive, pairs_from_poses, pairs_from_covisibility

parser = argparse.ArgumentParser()
parser.add_argument("--image-dir", "-i", required=True)
# parser.add_argument("--mask-dir", "-m", default=None)  # TODO: mask
parser.add_argument("--output-dir", "-o", required=True)
parser.add_argument("--camera-model", "-c", default="OPENCV")
parser.add_argument("--num-matched", "-n", type=int, default=64)
parser.add_argument("--matching-method", type=str, default="netvlad")
parser.add_argument("--matcher-conf", type=str, default="superglue", help=", ".join(list(match_features.confs.keys())))
parser.add_argument("--mapper-min-num-matches", type=int, default=15)
parser.add_argument("--sparse-model-dir", required=False, default=None)
parser.add_argument("--image-list", type=str, required=False, default=None)
parser.add_argument("--single-camera", action="store_true", default=False)
parser.add_argument("--indoor", action="store_true", default=False)
parser.add_argument("--pair-min-score", type=float, default=0.25)
parser.add_argument("--import-only", action="store_true", default=False)
args = parser.parse_args()

image_path = Path(args.image_dir)
mask_path = args.mask_dir
if mask_path is not None:
    mask_path = Path(mask_path)
output_path = Path(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

# Setup
"""
In this notebook, we will run SfM reconstruction from scratch on a set of images. 
First, we define some paths.
"""
images = image_path
outputs = output_path
sfm_pairs = outputs / 'pairs.txt'
sfm_dir = outputs / 'sfm_superpoint+superglue'

sfm_image_list = None
if args.image_list is not None:
    sfm_image_list = []
    with open(args.image_list, "r") as f:
        for row in f:
            row = row.rstrip("\n")
            if row == "":
                continue
            sfm_image_list.append(Path(row))

retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen' if not args.indoor else 'superpoint_inloc']
matcher_conf = match_features.confs[args.matcher_conf]

if args.matching_method == "netvlad":
    # Find image pairs via image retrieval
    """
    We extract global descriptors with NetVLAD and find for each image the most similar ones. 
    For smaller dataset we can instead use exhaustive matching via hloc/pairs_from_exhaustive.py, 
    which would find n(n-1)/2 images pairs
    """
    retrieval_path = extract_features.main(retrieval_conf, images, outputs, image_list=sfm_image_list)
    pairs_from_retrieval.main(
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
elif args.matching_method == "sift":
    pairs_from_covisibility.main(
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
else:
    raise ValueError("unsupported matching method {}".format(args.matching_method))

# Extract and match local features
feature_path = extract_features.main(feature_conf, images, outputs, image_list=sfm_image_list)
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

# 3D reconstruction
"""
Run COLMAP on the features and matches.
"""
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
print(sfm_dir)
