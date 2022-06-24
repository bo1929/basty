import argparse
from pathlib import Path

import joblib as jl
import numpy as np
from tqdm import tqdm

import basty.project.experiment_processing as experiment_processing
import basty.utils.io as io
import basty.utils.misc as misc

parser = argparse.ArgumentParser(description="Report and export predictions as a csv.")
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
parser.add_argument("--filter-behaviors", nargs="*")
args = parser.parse_args()


def report_behavior_predictions(project_obj, filter_behaviors):
    results_dir = project_obj.project_path / "results" / "semisupervised_pair_kNN"
    annotations_directories = list(results_dir.glob("predictions*/annotations*"))
    for annotations_dir in tqdm(annotations_directories):
        assert annotations_dir.is_dir()
        reports_dir = Path(str(annotations_dir).replace("annotations", "reports"))

        for annotations_pred_path in annotations_dir.glob("Fly*.npy"):
            expt_name = annotations_pred_path.stem
            expt_path = project_obj.expt_path_dict[expt_name]
            expt_record = jl.load(expt_path / "expt_record.z")

            annotations_pred = np.load(annotations_pred_path)
            label_to_behavior = expt_record.label_to_behavior
            report_df = misc.generate_bout_report(
                annotations_pred, label_to_behavior, filter_behaviors
            )

            io.safe_create_dir(reports_dir)
            if filter_behaviors is not None:
                report_path = (
                    reports_dir / f"{expt_name}-{'_'.join(filter_behaviors)}.csv"
                )
            else:
                report_path = reports_dir / f"{expt_name}.csv"
            report_df.to_csv(report_path)


if __name__ == "__main__":
    project = experiment_processing.Project(
        args.main_cfg_path,
    )

    report_behavior_predictions(project, args.filter_behaviors)
