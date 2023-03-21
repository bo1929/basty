import argparse
from pathlib import Path

import joblib as jl
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import normalize

import basty.project.experiment_processing as experiment_processing
import basty.utils.io as io
import basty.utils.misc as misc

parser = argparse.ArgumentParser(description="Export predictions as a csv.")
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
args = parser.parse_args()


def export_behavior_predictions(project_obj):
    results_dir = project_obj.project_path / "results" / "semisupervised_pair_kNN"
    weights_directories = list(results_dir.glob("predictions*/weights*"))

    label_to_behavior = {}
    for expt_name in project_obj.annotation_path_dict.keys():
        expt_record = jl.load(project_obj.expt_path_dict[expt_name] / "expt_record.z")
        label_to_behavior = {**expt_record.label_to_behavior, **label_to_behavior}

    for weights_dir in tqdm(weights_directories):
        assert weights_dir.is_dir()
        exports_dir = Path(str(weights_dir)).parent / "exports"

        for weights_pred_path in weights_dir.glob("Fly*.npy"):
            expt_name = weights_pred_path.stem
            expt_path = project_obj.expt_path_dict[expt_name]
            expt_record = jl.load(expt_path / "expt_record.z")

            weights_pred = np.load(weights_pred_path)
            weights_pred = uniform_filter1d(weights_pred, size=90, axis=0)  # 120?
            weights_pred = np.round(np.abs(normalize(weights_pred, norm="l1")), 5)

            export_df = pd.DataFrame(
                weights_pred,
                columns=[
                    label_to_behavior.get(i, f"Behavior-{i}")
                    for i in range(weights_pred.shape[1])
                ],
            )

            io.safe_create_dir(exports_dir)
            export_path = exports_dir / f"{expt_name}.csv"
            export_df.to_csv(export_path)


if __name__ == "__main__":
    project = experiment_processing.Project(
        args.main_cfg_path,
    )

    export_behavior_predictions(project)
