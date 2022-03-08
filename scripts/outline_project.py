import argparse

from pathlib import Path

from basty.utils.io import read_config
from basty.project.experiment_processing import ExptDormantEpochs, ExptActiveBouts

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
    "--outline-dormant-epochs",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--outline-active-bouts",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--outline-all",
    action=argparse.BooleanOptionalAction,
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

    FPS = 30

    supervised_dormant_epochs = False
    dormant_epochs_kwargs = {
        "datums": [],
        "datums_winsize": FPS,
        "log_scale": False,
        "min_dormant": 300 * FPS,
        "num_gmm_comp": 2,
        "threshold_key": "local_max",
        "threshold_idx": 1,
        "epoch_winsize": 180 * FPS,
        "tol_duration": 90 * FPS,
        "tol_percent": 0.4,
    }

    if args.outline_dormant_epochs or args.outline_all:
        dormant_epochs = ExptDormantEpochs(
            args.main_cfg_path,
            use_supervised_learning=supervised_dormant_epochs,
            **dormant_epochs_kwargs
        )
        dormant_epochs.outline_dormant_epochs()

    supervised_active_bouts = True
    active_bouts_kwargs = {
        "datums_list": [[]],
        "datums_winsize": FPS // 5,
        "log_scale": True,
        "post_processing_winsize": FPS * 2,
        "post_processing_wintype": "boxcar",
    }
    decision_tree_kwargs = {
        "n_estimators": 10,
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_leaf": 10 ** 3,
        "max_features": "sqrt",
        "criterion": "gini",
        "class_weight": "balanced",
    }

    if args.outline_active_bouts or args.outline_all:
        active_bouts = ExptActiveBouts(
            args.main_cfg_path,
            use_supervised_learning=supervised_active_bouts,
            **{**decision_tree_kwargs, **active_bouts_kwargs}
        )
        active_bouts.outline_active_bouts()
