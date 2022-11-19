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
    supervised_active_bouts = True

    dormant_epochs_unsupervised_kwargs = {
        "min_dormant": 30 * FPS,
        "num_gmm_comp": 2,
        "threshold_key": "local_max",
        # Indices start from 0.
        "threshold_idx": 1,
        "epoch_winsize": 90 * FPS,
        "tol_duration": 30 * FPS,
        "tol_percent": 0.4,
    }
    label_conversion_dict = {
        0: [
            "Idle&Other",
            "HaltereSwitch",
            "Feeding",
            "Grooming",
            "ProboscisPumping",
        ],
        1: [
            "PosturalAdjustment&Moving",
        ],
        2: [
            "Noise",
        ],
    }
    dormant_epochs_supervised_kwargs = {"label_conversion_dict": label_conversion_dict}
    dormant_epochs_kwargs = {
        "datums": [],
        "datums_winsize": FPS // 3,
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

    active_bouts_unsupervised_kwargs = {
        "num_gmm_comp": 3,
        "threshold_key": "local_min",
        # Indices start from 0.
        "threshold_idx": 1,
    }
    label_conversion_dict = {
        0: [
            "Noise",
            "Idle&Other",
            "HaltereSwitch",
        ],
        1: [
            "Feeding",
            "Grooming",
            "ProboscisPumping",
            "PosturalAdjustment&Moving",
        ],
    }
    active_bouts_supervised_kwargs = {"label_conversion_dict": label_conversion_dict}
    active_bouts_kwargs = {
        "datums_list": [[]],
        "datums_winsize": max(FPS // 10, 1),
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
