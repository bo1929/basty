import argparse
from collections import defaultdict

import joblib as jl
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import basty.project.experiment_processing as experiment_processing
import basty.utils.misc as misc
from basty.utils.postprocessing import PostProcessing

parser = argparse.ArgumentParser(
    description="Predict behaviors using kNN on semisupervised pair embeddings."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
parser.add_argument(
    "--num-neighbors",
    type=int,
    required=True,
    help="Number of neighbors to use in kNN algorithm.",
)
parser.add_argument(
    "--hard-vote",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--normalize-weights",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--silent",
    action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()


def get_mask(embedding_name, expt_record):
    if expt_record.use_annotations_to_mask[embedding_name]:
        maskDAnn = np.logical_and(expt_record.mask_annotated, expt_record.mask_dormant)
        mask = maskDAnn
    else:
        maskDA = np.logical_and(expt_record.mask_active, expt_record.mask_dormant)
        mask = maskDA
    return mask


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
            intersection_score_dict[label] = np.mean(intersect_indicators)
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


def predict_behavioral_bouts(project_obj, verbose, **kwargs):
    annotated_expt_names = list(project_obj.annotation_path_dict.keys())
    all_expt_names = list(project_obj.expt_path_dict.keys())
    unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
    pairs = misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)

    total_weight_dict = defaultdict(list)
    all_behavior_to_label = {}
    arouse_annotation = None

    for expt_name_ann, expt_name_unann in pairs:
        print(f"{expt_name_ann} predicts {expt_name_unann}.")
        expt_path_unann = project_obj.expt_path_dict[expt_name_unann]
        expt_path_ann = project_obj.expt_path_dict[expt_name_ann]

        expt_record_unann = jl.load(expt_path_unann / "expt_record.z")
        expt_record_ann = jl.load(expt_path_ann / "expt_record.z")

        unann_embedding_name = f"semisupervised_pair_embedding_{expt_name_ann}"
        unann_embedding_dir = expt_path_unann / "embeddings"
        X_unann = np.load(unann_embedding_dir / f"{unann_embedding_name}.npy")

        ann_embedding_name = f"semisupervised_pair_embedding_{expt_name_unann}"
        ann_embedding_dir = expt_path_ann / "embeddings"
        X_ann = np.load(ann_embedding_dir / f"{ann_embedding_name}.npy")

        mask_active_unann = get_mask(unann_embedding_name, expt_record_unann)
        mask_arouse_unann = np.logical_not(expt_record_unann.mask_dormant)
        mask_active_ann = get_mask(ann_embedding_name, expt_record_ann)

        all_behavior_to_label = {
            **all_behavior_to_label,
            **expt_record_ann.behavior_to_label,
        }

        num_of_labels = len(all_behavior_to_label.values())
        unann_num_of_frames = mask_active_unann.shape[0]

        if arouse_annotation is not None:
            assert expt_record_ann.arouse_annotation == arouse_annotation
        arouse_annotation = expt_record_ann.arouse_annotation
        arouse_label = all_behavior_to_label[arouse_annotation]

        if expt_record_ann.has_annotation:
            annotations_ann = np.load(expt_path_ann / "annotations.npy")
            y_true_ann = annotations_ann[mask_active_ann]
        else:
            raise ValueError

        neigh = KNeighborsClassifier(
            n_neighbors=kwargs.get("num_neighbors", 10), weights="distance"
        )
        neigh.fit(X_ann, y_true_ann)

        distances, n_neighbors = neigh.kneighbors(X_unann)
        weights = 1 / distances
        neighbors_labels = np.take_along_axis(
            y_true_ann[:, np.newaxis], n_neighbors, axis=0
        )
        w_pred_unann = np.zeros((neighbors_labels.shape[0], num_of_labels))
        for i in range(neighbors_labels.shape[0]):
            for j in range(neighbors_labels.shape[1]):
                w_pred_unann[i, neighbors_labels[i, j]] += weights[i, j]

        if kwargs.get("normalize_weights", False):
            uniq_true_ann, counts_true_ann = np.unique(y_true_ann, return_counts=True)
            for idx, label in enumerate(uniq_true_ann):
                # denom = np.sqrt(counts_true_ann[idx] + 1)
                # denom = np.log2(counts_true_ann[idx] + 1)
                denom = counts_true_ann[idx] + 1
                w_pred_unann[:, label] = w_pred_unann[:, label] / denom

        annotations_w_pred_unann = np.zeros((unann_num_of_frames, num_of_labels))
        annotations_w_pred_unann[mask_active_unann, :] = w_pred_unann
        annotations_w_pred_unann[mask_arouse_unann, arouse_label] = 1.0

        total_weight_dict[expt_name_unann].append(annotations_w_pred_unann)

    for expt_name_unann in unannotated_expt_names:
        expt_path_unann = project_obj.expt_path_dict[expt_name_unann]
        expt_record_unann = jl.load(expt_path_unann / "expt_record.z")

        num_of_labels = len(all_behavior_to_label.values())
        unann_num_of_frames = expt_record_unann.mask_dormant.shape[0]

        if kwargs.get("hard_vote", False):
            annotations_vt_pred_unann = np.zeros((unann_num_of_frames, num_of_labels))
            for i in range(len(total_weight_dict[expt_name_unann])):
                ami = np.argmax(total_weight_dict[expt_name_unann][i], axis=1)
                annotations_vt_pred_unann[np.arange(ami.shape[0]), ami] += 1
            annotations_pred_unann = np.argmax(annotations_vt_pred_unann, axis=1)
        else:
            annotations_wt_pred_unann = np.sum(
                total_weight_dict[expt_name_unann], axis=0
            )
            annotations_pred_unann = np.argmax(annotations_wt_pred_unann, axis=1)

        annotations_pred_unann = PostProcessing.compute_window_majority(
            annotations_pred_unann, kwargs.get("majority_window_size", 0)
        )
        annotations_pred_unann = PostProcessing.postprocess_wrt_durations(
            annotations_pred_unann,
            [
                all_behavior_to_label[behaivor]
                for behaivor in kwargs.get("postprocess_short_behaviors", [])
            ],
            kwargs.get("min_short_duration", 0),
        )
        annotations_pred_unann = PostProcessing.postprocess_wrt_durations(
            annotations_pred_unann,
            [
                all_behavior_to_label[behaivor]
                for behaivor in kwargs.get("postprocess_long_behaviors", [])
            ],
            kwargs.get("max_long_duration", 0),
        )
        annotations_pred_unann[annotations_pred_unann == -1] = 0

        if expt_record_unann.has_annotation and verbose:
            annotations_unann = np.load(expt_path_unann / "annotations.npy")
            report_full = get_performance_report(
                annotations_unann, annotations_pred_unann, all_behavior_to_label
            )
            print("\n", expt_name_unann)
            print(report_full)


if __name__ == "__main__":
    project = experiment_processing.Project(
        args.main_cfg_path,
    )

    prediction_kwargs = {
        "num_neighbors": args.num_neighbors,
        "hard_vote": args.hard_vote,
        "normalize_weights": args.normalize_weights,
    }
    postprocessing_kwargs = {
        "majority_window_size": int(project.fps * 2),
        "min_short_duration": int(project.fps * 1),
        "postprocess_short_behaviors": [
            "ProboscisPump",
            "Moving",
            "Grooming",
            "HaltereSwitch",
        ],
        "max_long_duration": int(project.fps * -10.0),
        "postprocess_long_behaviors": ["HaltereSwitch"],
    }
    predict_behavioral_bouts(
        project, verbose=(not args.silent), **prediction_kwargs, **postprocessing_kwargs
    )
