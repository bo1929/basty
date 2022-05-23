import argparse

import joblib as jl
import numpy as np

import basty.project.experiment_processing as experiment_processing

parser = argparse.ArgumentParser(
    description="Report details about active and dormant masks."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
parser.add_argument(
    "--evaluate-active-bouts",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--evaluate-dormant-epochs",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--silent",
    action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()


def get_recall_report(recall_dict, behavior_domain):
    report = ""
    for behavior in behavior_domain:
        if recall_dict[behavior] is None:
            report += f"\t\t- {behavior} is not observed.\n"
        else:
            report += (
                f"\t\t- {round(recall_dict[behavior], 2)} of {behavior} is observed.\n"
            )
    return report


def get_evaluation_report(evaluation_dict, behavior_domain):
    report = ""
    report += (
        f"\t= {round(evaluation_dict['masked_percent'], 2)} of frames are masked.\n"
    )
    report += get_recall_report(evaluation_dict["recall_dict"], behavior_domain)
    return report


def get_recall_scores(masked_annotation_counts, annotation_counts, expt_record):
    all_unique, all_counts = annotation_counts
    masked_unique, masked_counts = masked_annotation_counts

    all_count_dict = {}
    masked_count_dict = {}

    for idx, label in enumerate(all_unique):
        behavior = expt_record.label_to_behavior[label]
        all_count_dict[behavior] = all_counts[idx]

    for idx, label in enumerate(masked_unique):
        behavior = expt_record.label_to_behavior[label]
        masked_count_dict[behavior] = masked_counts[idx]

    recall_dict = {}
    for behavior in list(expt_record.label_to_behavior.values()):
        all_count = all_count_dict.get(behavior, 0)
        masked_count = masked_count_dict.get(behavior, 0)
        recall_dict[behavior] = masked_count / all_count if all_count else None
    return recall_dict


def active_bout_evaluation(annotations, annotation_counts, expt_record):
    maskA = expt_record.mask_active
    masked_annotation_counts = np.unique(annotations[maskA], return_counts=True)

    masked_percent = round(np.count_nonzero(maskA) / annotations.shape[0], 2)

    recall_dict = get_recall_scores(
        masked_annotation_counts, annotation_counts, expt_record
    )

    evaluation_dict = {"recall_dict": recall_dict, "masked_percent": masked_percent}
    return evaluation_dict


def dormant_epoch_evaluation(annotations, annotation_counts, expt_record):
    maskD = expt_record.mask_dormant
    masked_annotation_counts = np.unique(annotations[maskD], return_counts=True)

    masked_percent = round(np.count_nonzero(maskD) / annotations.shape[0], 2)

    recall_dict = get_recall_scores(
        masked_annotation_counts, annotation_counts, expt_record
    )

    evaluation_dict = {"recall_dict": recall_dict, "masked_percent": masked_percent}
    return evaluation_dict


def active_bout_in_dormant_epoch_evaluation(
    annotations, annotation_counts, expt_record
):
    maskDA = np.logical_and(expt_record.mask_active, expt_record.mask_dormant)
    masked_annotation_counts = np.unique(annotations[maskDA], return_counts=True)

    masked_percent = round(np.count_nonzero(maskDA) / annotations.shape[0], 2)

    recall_dict = get_recall_scores(
        masked_annotation_counts, annotation_counts, expt_record
    )

    evaluation_dict = {"recall_dict": recall_dict, "masked_percent": masked_percent}
    return evaluation_dict


def evaluate_predicted_masks(
    project_obj,
    evaluate_active_bouts=False,
    evaluate_dormant_epochs=False,
    verbose=True,
):
    all_expt_names = list(project_obj.expt_path_dict.keys())

    for expt_name in all_expt_names:
        expt_path = project_obj.expt_path_dict[expt_name]
        expt_record = jl.load(expt_path / "expt_record.z")
        report = ""
        if expt_record.has_annotation:
            report += "============================================================\n"
            report += f"Evaluation for {expt_name};\n"

            behavior_domain = list(expt_record.label_to_behavior.values())
            annotations = np.load(expt_path / "annotations.npy")
            annotation_counts = np.unique(annotations, return_counts=True)

            if evaluate_active_bouts:
                report += "- Performance report for active bouts:\n"
                evaluation_dict = active_bout_evaluation(
                    annotations, annotation_counts, expt_record
                )
                report += get_evaluation_report(evaluation_dict, behavior_domain)
            if evaluate_dormant_epochs:
                report += "- Performance report for dormant epochs:\n"
                evaluation_dict = dormant_epoch_evaluation(
                    annotations, annotation_counts, expt_record
                )
                report += get_evaluation_report(evaluation_dict, behavior_domain)

            if evaluate_dormant_epochs and evaluate_active_bouts:
                report += "- Performance report for active bouts in dormant epochs:\n"
                evaluation_dict = active_bout_in_dormant_epoch_evaluation(
                    annotations, annotation_counts, expt_record
                )
                report += get_evaluation_report(evaluation_dict, behavior_domain)
            report += "============================================================\n"

        if verbose:
            print(report)


if __name__ == "__main__":
    project = experiment_processing.Project(
        args.main_cfg_path,
    )
    evaluate_predicted_masks(
        project,
        evaluate_active_bouts=args.evaluate_active_bouts,
        evaluate_dormant_epochs=args.evaluate_dormant_epochs,
        verbose=not args.silent,
    )
