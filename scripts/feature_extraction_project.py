import argparse

from pathlib import Path

from basty.utils.io import read_config
from basty.project.feature_extraction import FeatureExtraction

parser = argparse.ArgumentParser(
    description="Initialize project  based on a given main configuration."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
parser.add_argument(
    "--reset-project",
    action=argparse.BooleanOptionalAction,
    help="Option to create a new project.",
)
parser.add_argument(
    "--compute-pose-values",
    action=argparse.BooleanOptionalAction,
    help="Option to compute pose values.",
)
parser.add_argument(
    "--compute-spatiotemporal-features",
    action=argparse.BooleanOptionalAction,
    help="Option to compute spatiotemporal values.",
)
parser.add_argument(
    "--compute-postural-dynamics",
    action=argparse.BooleanOptionalAction,
    help="Option to compute postural dynamics.",
)
parser.add_argument(
    "--compute-behavioral-representations",
    action=argparse.BooleanOptionalAction,
    help="Option to compute behavioral representations.",
)
parser.add_argument(
    "--compute-all",
    action=argparse.BooleanOptionalAction,
    help="Option to compute all.",
)

args = parser.parse_args()


def backup_old_project(main_cfg_path):
    main_cfg = read_config(main_cfg_path)
    project_path = main_cfg.get("project_path")
    old_proj_dir = Path(project_path)
    if old_proj_dir.exists():
        suffix = ".old"
        while old_proj_dir.with_suffix(suffix).exists():
            suffix = suffix + ".old"
        old_proj_dir.replace(old_proj_dir.with_suffix(suffix))


if __name__ == "__main__":
    """
    Add suffix '.old' to the project created by previous tests.
    """
    if args.reset_project:
        backup_old_project(args.main_cfg_path)

    extract_delta = True
    extract_snap = True

    pose_prep_kwargs = {
        "compute_oriented_pose": True,
        "compute_egocentric_frames": False,
        "save_likelihood": False,
        "local_outlier_threshold": 9,
        "local_outlier_winsize": 15,
        "decreasing_llh_winsize": 30,
        "decreasing_llh_lower_z": -4.5,
        "low_llh_threshold": 0.075,
        "median_filter_winsize": 6,
        "boxcar_filter_winsize": 4,
        "jump_quantile": 0,
        "interpolation_method": "linear",
        "interpolation_kwargs": {},
        "kalman_filter_kwargs": {},
    }
    behavioral_reprs_kwargs = {"use_cartesian_blent": False, "norm": "l1"}

    ft_ext = FeatureExtraction(
        args.main_cfg_path, **behavioral_reprs_kwargs, **pose_prep_kwargs
    )
    if args.compute_pose_values or args.compute_all:
        ft_ext.compute_pose_values()
    if args.compute_spatiotemporal_features or args.compute_all:
        ft_ext.compute_spatiotemporal_features(delta=extract_delta, snap=extract_snap)
    if args.compute_postural_dynamics or args.compute_all:
        ft_ext.compute_postural_dynamics()
    if args.compute_behavioral_representations or args.compute_all:
        ft_ext.compute_behavioral_representations()
