import argparse
from collections import defaultdict
from pathlib import Path

import joblib as jl
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import basty.project.experiment_processing as experiment_processing
import basty.utils.io as io
import basty.utils.misc as misc

parser = argparse.ArgumentParser(
    description="Evaluate and report details about behavior predictions."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
args = parser.parse_args()


def get_bout_details_report(y_true, y_pred, behavior_to_label):
    bout_details_report = "Details about the bouts:\n"

    def get_bout_details(y):
        intvls = misc.cont_intvls(y)
        number_of_bouts = defaultdict(int)
        duration_of_bouts = defaultdict(list)
        interval_of_bouts = defaultdict(list)
        for i in range(intvls.shape[0] - 1):
            number_of_bouts[y[intvls[i]]] += 1
            duration_of_bouts[y[intvls[i]]].append(intvls[i + 1] - intvls[i])
            interval_of_bouts[y[intvls[i]]].append((intvls[i], intvls[i + 1]))
        return number_of_bouts, duration_of_bouts, interval_of_bouts

    def generate_report(number_of_bouts, duration_of_bouts, label_to_behavior):
        bout_details_report = ""
        for label, count in number_of_bouts.items():
            median_duration = np.median(duration_of_bouts[label])
            q25_duration = np.quantile(duration_of_bouts[label], 0.25)
            q75_duration = np.quantile(duration_of_bouts[label], 0.75)
            bout_details_report += f"\t- {label_to_behavior[label]}: "
            bout_details_report += f"number of bouts {count}, "
            bout_details_report += "Q25, median, & Q75 duration: "
            bout_details_report += (
                f"{q25_duration}, {median_duration} & {q75_duration}\n"
            )
        return bout_details_report

    (
        number_of_bouts_true,
        duration_of_bouts_true,
        interval_of_bouts_true,
    ) = get_bout_details(y_true)
    (
        number_of_bouts_pred,
        duration_of_bouts_pred,
        interval_of_bouts_pred,
    ) = get_bout_details(y_pred)

    label_to_behavior = misc.reverse_dict(behavior_to_label)

    bout_details_report += "= For true annotations;\n"
    bout_details_report += generate_report(
        number_of_bouts_true, duration_of_bouts_true, label_to_behavior
    )

    bout_details_report += "= For predicted annotations;\n"
    bout_details_report += generate_report(
        number_of_bouts_pred, duration_of_bouts_pred, label_to_behavior
    )

    def compute_intersection_score(interval_of_bouts_1, interval_of_bouts_2):
        intersection_score_dict = {}
        for label, intervals_1 in interval_of_bouts_1.items():
            intersect_indicators = []
            intervals_2 = interval_of_bouts_2[label]
            for start_1, end_1 in intervals_1:
                for start_2, end_2 in intervals_2:
                    if start_2 >= start_1 and start_2 <= end_1:
                        found = True
                        break
                    elif start_1 >= start_2 and start_1 <= end_2:
                        found = True
                        break
                    elif start_2 >= end_1:
                        found = False
                        break
                    else:
                        found = False
                intersect_indicators.append(found)
            if not intersection_score_dict:
                intersection_score_dict[label] = np.mean(intersect_indicators)
            else:
                intersection_score_dict[label] = 0
        return intersection_score_dict

    recall_intersection_dict = compute_intersection_score(
        interval_of_bouts_true, interval_of_bouts_pred
    )
    precision_intersection_dict = compute_intersection_score(
        interval_of_bouts_pred, interval_of_bouts_true
    )

    def generate_report(
        recall_intersection_dict, precision_intersection_dict, label_to_behavior
    ):
        bout_details_report = ""
        for label, recall in recall_intersection_dict.items():
            precision = precision_intersection_dict[label]
            bout_details_report += f"\t- {label_to_behavior[label]}: "
            bout_details_report += "recall & precision w.r.t. bout intersection: "
            bout_details_report += f"{round(recall, 2)} & {round(precision, 2)}\n"
        return bout_details_report

    bout_details_report += "= Bout coverage scores;\n"
    bout_details_report += generate_report(
        recall_intersection_dict, precision_intersection_dict, label_to_behavior
    )

    return bout_details_report


def get_confusion_matrix_report(y_true, y_pred, behavior_to_label):
    label_to_behavior = misc.reverse_dict(behavior_to_label)
    labels_unique = np.unique(y_true)
    behaviors_unique = [label_to_behavior[label] for label in labels_unique]
    df = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels_unique),
        columns=behaviors_unique,
        index=behaviors_unique,
    )
    return "\n" + df.to_string()


def get_classification_report(y_true, y_pred, behavior_to_label):
    y_uniq = np.unique(y_pred)
    label_to_behavior = misc.reverse_dict(behavior_to_label)
    target_behaviors = [label_to_behavior[label] for label in y_uniq]
    labels = [behavior_to_label[behavior] for behavior in target_behaviors]
    report = (
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_behaviors,
        )
        + "\n"
    )
    return report


def get_performance_report(y_true, y_pred, behavior_to_label):
    report = ""
    report += get_classification_report(y_true, y_pred, behavior_to_label)
    report += get_bout_details_report(y_true, y_pred, behavior_to_label)
    report += get_confusion_matrix_report(y_true, y_pred, behavior_to_label)
    return report


def evaluate_behavioral_predictions(project_obj):
    results_dir = project_obj.project_path / "results" / "semisupervised_pair_kNN"
    for annotations_dir in results_dir.glob("predictions*/annotations*"):
        print(annotations_dir.name)
        assert annotations_dir.is_dir()
        evaluations_dir = Path(
            str(annotations_dir).replace("annotations", "evaluations")
        )

        for annotations_pred_path in annotations_dir.glob("Fly*.npy"):
            expt_name = annotations_pred_path.stem
            expt_path = project_obj.expt_path_dict[expt_name]
            expt_record = jl.load(expt_path / "expt_record.z")

            annotations_pred = np.load(annotations_pred_path)
            behavior_to_label = expt_record.behavior_to_label
            mask_dormant = expt_record.mask_dormant

            if expt_record.has_annotation:
                annotations = np.load(expt_path / "annotations.npy")

                report_full = get_classification_report(
                    annotations[mask_dormant],
                    annotations_pred[mask_dormant],
                    behavior_to_label,
                )
                report_full += get_performance_report(
                    annotations, annotations_pred, behavior_to_label
                )

                io.safe_create_dir(evaluations_dir)
                io.write_txt(report_full, evaluations_dir / f"{expt_name}.txt")
            else:
                print(f"{expt_name} does not have annotations, skipping...")


if __name__ == "__main__":
    project = experiment_processing.Project(
        args.main_cfg_path,
    )

    evaluate_behavioral_predictions(project)
