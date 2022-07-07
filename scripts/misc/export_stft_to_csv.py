import argparse

import basty.project.experiment_processing as experiment_processing
import pandas as pd

parser = argparse.ArgumentParser(
    description="Export spatio-temporal features as csv files."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
args = parser.parse_args()


def export_stft(
    project_obj,
):
    for expt_name, expt_path in project_obj.expt_path_dict.items():
        snap_stft_path = expt_path / "snap_stft.pkl"
        delta_stft_path = expt_path / "delta_stft.pkl"

        if snap_stft_path.exists():
            pd.read_pickle(snap_stft_path).to_csv(expt_path / "snap_stft.csv")

        if delta_stft_path.exists():
            pd.read_pickle(delta_stft_path).to_csv(expt_path / "delta_stft.csv")


if __name__ == "__main__":
    project = experiment_processing.Project(
        args.main_cfg_path,
    )
    export_stft(
        project,
    )
