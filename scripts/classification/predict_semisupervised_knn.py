import argparse
from collections import defaultdict

import joblib as jl
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

import basty.project.experiment_processing as experiment_processing
import basty.utils.io as io
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
    "--weighting",
    type=str,
    choices=["distance", "sq_distance"],
)
parser.add_argument(
    "--normalize-weights",
    type=str,
    choices=["count", "log_count", "sqrt_count", "proportion"],
)
parser.add_argument(
    "--activation",
    type=str,
    choices=["softmax", "standard"],
)
parser.add_argument(
    "--use-entropy",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--hard-vote",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--save_weights",
    default=False,
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


def predict_behavioral_bouts(project_obj, save_weights=False, **kwargs):
    all_expt_names = list(project_obj.expt_path_dict.keys())
    annotated_expt_names = list(project_obj.annotation_path_dict.keys())
    unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
    if project_obj.evaluation_mode:
        unannotated_expt_names = annotated_expt_names

    assert all_expt_names
    assert annotated_expt_names
    assert unannotated_expt_names
    pairs = misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
    pairs = [(name1, name2) for name1, name2 in pairs if name1 != name2]

    total_weight_dict = defaultdict(dict)
    all_behavior_to_label = {}
    arouse_annotation = None

    results_dir = project_obj.project_path / "results" / "semisupervised_pair_kNN"

    weighting = kwargs.get("weighting", None)
    normalize_weights = kwargs.get("normalize_weights", None)
    num_neigh = kwargs.get("num_neighbors", 10)
    activation = kwargs.get("activation", None)
    use_entropy = kwargs.get("use_entropy", False)

    name_predictions = "predictions"
    name_predictions += f".weighting-{weighting}"
    name_predictions += f".wnorm-{normalize_weights}"
    name_predictions += f".{num_neigh}NN"
    name_predictions += f".activation-{activation}"
    name_predictions += f".entropy-{use_entropy}"

    for expt_name_ann, expt_name_unann in pairs:
        print(f"{expt_name_ann} predicts {expt_name_unann}.")
        expt_path_unann = project_obj.expt_path_dict[expt_name_unann]
        expt_path_ann = project_obj.expt_path_dict[expt_name_ann]

        expt_record_unann = jl.load(expt_path_unann / "expt_record.z")
        expt_record_ann = jl.load(expt_path_ann / "expt_record.z")

        embedding_type = "semisupervised_pair_embedding"

        unann_embedding_name = f"{embedding_type}_{expt_name_ann}_{expt_name_unann}"
        unann_embedding_dir = expt_path_unann / "embeddings"
        X_unann = np.load(unann_embedding_dir / f"{unann_embedding_name}.npy")

        ann_embedding_name = f"{embedding_type}_{expt_name_ann}_{expt_name_unann}"
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

        neigh = KNeighborsClassifier(n_neighbors=num_neigh, weights="distance")
        neigh.fit(X_ann, y_true_ann)

        distances, n_neighbors = neigh.kneighbors(X_unann)
        if weighting == "distance":
            weights = 1 / distances
        elif weighting == "sq_distance":
            weights = 1 / distances ** 2
        else:
            weights = np.ones(distances.shape)

        neighbors_labels = np.take_along_axis(
            y_true_ann[:, np.newaxis], n_neighbors, axis=0
        )
        w_pred_unann = np.zeros((neighbors_labels.shape[0], num_of_labels))
        for i in range(neighbors_labels.shape[0]):
            for j in range(neighbors_labels.shape[1]):
                w_pred_unann[i, neighbors_labels[i, j]] += weights[i, j]

        if normalize_weights is not None:
            uniq_true_ann, counts_true_ann = np.unique(y_true_ann, return_counts=True)
            counts_true_ann += 1
            proportions_true_ann = counts_true_ann / y_true_ann.shape[0]
            for idx, label in enumerate(uniq_true_ann):
                if normalize_weights == "count":
                    denom = counts_true_ann[idx]
                elif normalize_weights == "log_count":
                    denom = np.log2(counts_true_ann[idx])
                elif normalize_weights == "sqrt_count":
                    denom = np.sqrt(counts_true_ann[idx])
                elif normalize_weights == "proportion":
                    denom = proportions_true_ann[idx]
                else:
                    denom = 1
                w_pred_unann[:, label] = w_pred_unann[:, label] / denom

        if activation == "softmax":
            w_pred_unann = softmax(w_pred_unann, axis=1)
        elif activation == "standard":
            w_pred_unann = normalize(w_pred_unann, norm="l1")
        else:
            w_pred_unann = w_pred_unann

        if use_entropy:
            entropy_pred_unann = entropy(w_pred_unann, axis=1, base=2)
            max_entropy = entropy(
                [1 / num_of_labels for _ in range(num_of_labels)], base=2
            )
            w_pred_unann *= max_entropy - entropy_pred_unann.reshape(-1, 1)

        annotations_w_pred_unann = np.zeros((unann_num_of_frames, num_of_labels))
        annotations_w_pred_unann[mask_active_unann, :] = w_pred_unann
        annotations_w_pred_unann[mask_arouse_unann, arouse_label] = np.inf

        total_weight_dict[expt_name_unann][expt_name_ann] = annotations_w_pred_unann

    if save_weights:
        weights_dir = results_dir / name_predictions / "weights"
        io.safe_create_dir(weights_dir)
        for expt_name_unann in total_weight_dict.keys():
            np.savez(
                weights_dir / f"{expt_name_unann}.npz",
                **total_weight_dict[expt_name_unann],
            )

    hard_vote = kwargs.get("hard_vote", False)
    majority_window_size = kwargs.get("majority_window_size", 0)
    min_short_duration = kwargs.get("min_short_duration", 0)
    max_long_duration = kwargs.get("max_long_duration", 0)

    postprocess_short_behaviors = kwargs.get("postprocess_short_behaviors", [])
    postprocess_short_labels = [
        all_behavior_to_label[behaivor] for behaivor in postprocess_short_behaviors
    ]
    postprocess_long_behaviors = kwargs.get("postprocess_long_behaviors", [])
    postprocess_long_labels = [
        all_behavior_to_label[behaivor] for behaivor in postprocess_long_behaviors
    ]

    name_annotations = "annotations"
    name_annotations += f".hard_vote-{hard_vote}"
    name_annotations += f".majority_ws-{majority_window_size}"
    name_annotations += f".min_short-{min_short_duration}"
    name_annotations += f".max_long-{max_long_duration}"

    annotations_dir = results_dir / name_predictions / name_annotations
    io.safe_create_dir(annotations_dir)

    for expt_name_unann in unannotated_expt_names:
        expt_path_unann = project_obj.expt_path_dict[expt_name_unann]
        expt_record_unann = jl.load(expt_path_unann / "expt_record.z")

        num_of_labels = len(all_behavior_to_label.values())
        mask_dormant = expt_record_unann.mask_dormant
        unann_num_of_frames = mask_dormant.shape[0]

        total_weights = list(total_weight_dict[expt_name_unann].values())

        if hard_vote:
            annotations_vt_pred_unann = np.zeros((unann_num_of_frames, num_of_labels))
            for i in range(len(total_weights)):
                ami = np.argmax(total_weights[i], axis=1)
                annotations_vt_pred_unann[np.arange(ami.shape[0]), ami] += 1
            annotations_pred_unann = np.argmax(annotations_vt_pred_unann, axis=1)
        else:
            annotations_wt_pred_unann = np.sum(total_weights, axis=0)
            annotations_pred_unann = np.argmax(annotations_wt_pred_unann, axis=1)

        annotations_pred_unann = PostProcessing.compute_window_majority(
            annotations_pred_unann,
            majority_window_size,
        )
        annotations_pred_unann = PostProcessing.postprocess_wrt_durations(
            annotations_pred_unann,
            postprocess_short_labels,
            min_short_duration,
        )
        annotations_pred_unann = PostProcessing.postprocess_wrt_durations(
            annotations_pred_unann,
            postprocess_long_labels,
            max_long_duration,
        )
        annotations_pred_unann[annotations_pred_unann == -1] = 0

        np.save(annotations_dir / f"{expt_name_unann}.npy", annotations_pred_unann)


if __name__ == "__main__":
    project = experiment_processing.Project(
        args.main_cfg_path,
    )

    prediction_kwargs = {
        "num_neighbors": args.num_neighbors,
        "weighting": args.weighting,
        "normalize_weights": args.normalize_weights,
        "activation": args.activation,
        "use_entropy": args.use_entropy,
        "hard_vote": args.hard_vote,
    }
    postprocessing_kwargs = {
        "majority_window_size": int(project.fps * 3),
        "min_short_duration": int(project.fps * 1.5),
        "postprocess_short_behaviors": [
            "ProboscisPump",
            "Moving",
            "Grooming",
            "HaltereSwitch",
        ],
        "max_long_duration": int(project.fps * 0),
        "postprocess_long_behaviors": [],
    }
    predict_behavioral_bouts(
        project,
        save_weights=args.save_weights,
        **prediction_kwargs,
        **postprocessing_kwargs,
    )
