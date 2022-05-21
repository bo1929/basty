import argparse
from collections import defaultdict

import joblib as jl
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
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
    "--silent",
    action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()


def get_mask_suffix(expt_record):
    return "-DAnn" if expt_record.use_annotations_to_mask else "-DA"


def get_mask(expt_record):
    if expt_record.use_annotations_to_mask:
        maskDAnn = np.logical_and(expt_record.mask_annotated, expt_record.mask_dormant)
        mask = maskDAnn
    else:
        maskDA = np.logical_and(expt_record.mask_active, expt_record.mask_dormant)
        mask = maskDA
    return mask


def get_classification_report(y_true, y_pred, behavior_to_label):
    y_uniq = np.unique(y_pred)
    label_to_behavior = misc.reverse_dict(behavior_to_label)
    target_behaviors = [label_to_behavior[label] for label in y_uniq]
    labels = [behavior_to_label[behavior] for behavior in target_behaviors]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_behaviors,
    )
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

        # expt_record_unann = jl.load(expt_path_unann / "expt_record.z")
        expt_record_ann = jl.load(expt_path_ann / "expt_record.z")

        X_unann = np.load(
            expt_path_unann
            / "embeddings"
            / f"semisupervised_pair_embedding_{expt_name_ann}.npy"
        )
        X_ann = np.load(
            expt_path_ann
            / "embeddings"
            / f"semisupervised_pair_embedding_{expt_name_unann}.npy"
        )

        mask_active_ann = get_mask(expt_record_ann)
        all_behavior_to_label = {
            **all_behavior_to_label,
            **expt_record_ann.behavior_to_label,
        }
        if arouse_annotation is not None:
            assert expt_record_ann.arouse_annotation == arouse_annotation
        arouse_annotation = expt_record_ann.arouse_annotation

        if expt_record_ann.has_annotation:
            annotations_ann = np.load(expt_path_ann / "annotations.npy")
            y_true_ann = annotations_ann[mask_active_ann]
        else:
            raise ValueError

        neigh = KNeighborsClassifier(
            n_neighbors=kwargs["num_neighbors"], weights="distance"
        )
        neigh.fit(X_ann, y_true_ann)

        distances, n_neighbors = neigh.kneighbors(X_unann)
        weights = 1 / distances
        neighbors_labels = np.take_along_axis(
            y_true_ann[:, np.newaxis], n_neighbors, axis=0
        )
        w_pred_unann = np.zeros((neighbors_labels.shape[0], np.max(neigh.classes_) + 1))
        for i in range(neighbors_labels.shape[0]):
            for j in range(neighbors_labels.shape[1]):
                w_pred_unann[i, neighbors_labels[i, j]] += weights[i, j]
        total_weight_dict[expt_name_unann].append(w_pred_unann)

    for expt_name_unann in unannotated_expt_names:
        expt_path_unann = project_obj.expt_path_dict[expt_name_unann]
        expt_record_unann = jl.load(expt_path_unann / "expt_record.z")

        mask_active_unann = get_mask(expt_record_unann)
        mask_arouse_unann = np.logical_not(expt_record_unann.mask_dormant)
        arouse_label = all_behavior_to_label[arouse_annotation]

        if kwargs.get("hard_vote", False):
            v_tpred_unann = np.zeros(
                (np.count_nonzero(mask_active_unann), len(all_behavior_to_label))
            )
            for i in range(len(total_weight_dict[expt_name_unann])):
                ami = np.argmax(total_weight_dict[expt_name_unann][i], axis=1)
                v_tpred_unann[np.arange(ami.shape[0]), ami] += 1
            y_tpred_unann = np.argmax(v_tpred_unann, axis=1)
        else:
            w_tpred_unann = np.sum(total_weight_dict[expt_name_unann], axis=0)
            y_tpred_unann = np.argmax(w_tpred_unann, axis=1)

        annotations_pred_unann = np.zeros(mask_active_unann.shape[0], dtype=int)
        annotations_pred_unann[mask_active_unann] = y_tpred_unann
        annotations_pred_unann[mask_arouse_unann] = arouse_label

        labels_to_process = [
            all_behavior_to_label[behaivor]
            for behaivor in kwargs.get("behaviors_to_process", [])
        ]
        annotations_pred_unann = PostProcessing.compute_window_majority(
            annotations_pred_unann, kwargs.get("majority_window_size", 0)
        )

        annotations_pred_unann = PostProcessing.process_short_cont_intvls(
            annotations_pred_unann,
            labels_to_process,
            kwargs.get("min_short_intvl", 0),
        )
        annotations_pred_unann[annotations_pred_unann == -1] = 0

        if expt_record_unann.has_annotation and verbose:
            annotations_unann = np.load(expt_path_unann / "annotations.npy")
            report_full = get_classification_report(
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
    }
    postprocessing_kwargs = {
        "behaviors_to_process": ["ProboscisPump", "Moving", "Grooming"],
        "majority_window_size": project.fps * 1,
        "min_short_intvl": int(project.fps * 1.5),
    }
    predict_behavioral_bouts(
        project, verbose=(not args.silent), **prediction_kwargs, **postprocessing_kwargs
    )
