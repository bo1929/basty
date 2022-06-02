import argparse

from utils import log_params

from basty.project.experiment_processing import ExptActiveBouts, ExptDormantEpochs

parser = argparse.ArgumentParser(
    description="Outlining project to compute dormant epochs and active bouts."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
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


if __name__ == "__main__":
    FPS = 30

    supervised_dormant_epochs = False
    dormant_epochs_unsupervised_kwargs = {
        "min_dormant": 300 * FPS,
        "num_gmm_comp": 2,
        "threshold_key": "local_max",
        "threshold_idx": 1,
        "epoch_winsize": 180 * FPS,
        "tol_duration": 90 * FPS,
        "tol_percent": 0.4,
    }
    dormant_epochs_supervised_kwargs = {}
    dormant_epochs_kwargs = {
        "datums": [],
        "datums_winsize": FPS,
        "log_scale": False,
        "scale": False,
        "use_supervised_learning": supervised_dormant_epochs,
        **dormant_epochs_supervised_kwargs,
        **dormant_epochs_unsupervised_kwargs,
    }
    log_params(args.main_cfg_path, "dormant_epochs", dormant_epochs_kwargs)

    if args.outline_dormant_epochs or args.outline_all:
        dormant_epochs = ExptDormantEpochs(args.main_cfg_path, **dormant_epochs_kwargs)
        dormant_epochs.outline_dormant_epochs()

    supervised_active_bouts = True
    active_bouts_unsupervised_kwargs = {
        "num_gmm_comp": 3,
        "threshold_key": "local_min",
        # Indices start from 0.
        "threshold_idx": 1,
    }
    active_bouts_supervised_kwargs = {
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_leaf": 10 ** 3,
        "max_features": "sqrt",
        "criterion": "gini",
        "class_weight": "balanced",
        "label_conversion_dict": {
            0: ["Idle&Other", "HaltereSwitch", "Noise"],
            1: ["Grooming", "ProboscisPump", "Moving", "Feeding"],
        },
    }
    active_bouts_kwargs = {
        "datums_list": [[]],
        "datums_winsize": FPS // 10,
        "scale": False,
        "log_scale": True,
        "coefs_summary_method": "sum",
        "post_processing_winsize": FPS,
        "post_processing_wintype": "boxcar",
        "use_supervised_learning": supervised_active_bouts,
        **active_bouts_supervised_kwargs,
        **active_bouts_unsupervised_kwargs,
    }

    log_params(
        args.main_cfg_path,
        "active_bouts",
        active_bouts_kwargs,
    )

    if args.outline_active_bouts or args.outline_all:
        active_bouts = ExptActiveBouts(args.main_cfg_path, **active_bouts_kwargs)
        active_bouts.outline_active_bouts()
