import argparse

from utils import log_params

from basty.project.behavior_mapping import BehaviorMapping

parser = argparse.ArgumentParser(
    description="Behavior mapping, many options (joint, disparate etc.) are available."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)

# Embedding arguments.
parser.add_argument(
    "--compute-supervised-disparate-embeddings",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--compute-unsupervised-disparate-embeddings",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--compute-unsupervised-joint-embeddings",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--compute-supervised-joint-embeddings",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--compute-semisupervised-pair-embeddings",
    action=argparse.BooleanOptionalAction,
)

# Clustering arguments.
parser.add_argument(
    "--jointly-cluster-supervised-joint",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--jointly-cluster-unsupervised-joint",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--jointly-cluster-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--disparately-cluster-supervised-joint",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--disparately-cluster-unsupervised-joint",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--disparately-cluster-supervised-disparate",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--disparately-cluster-unsupervised-disparate",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--disparately-cluster-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--crosswisely-cluster-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)

# Correspondence arguments.
parser.add_argument(
    "--map-disparate-cluster-supervised-disparate",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--disparately-compute-behavior-score-disparate-cluster-supervised-disparate",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-disparate-cluster-supervised-joint",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-disparate-cluster-unsupervised-disparate",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-disparate-cluster-unsupervised-joint",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-disparate-cluster-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--map-joint-cluster-supervised-joint",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-joint-cluster-unsupervised-joint",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-joint-cluster-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--map-crosswise-cluster-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--crosswisely-compute-behavior-score-crosswise-cluster-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)

args = parser.parse_args()

if __name__ == "__main__":
    UMAP_kwargs = {}
    UMAP_kwargs["embedding_n_neighbors"] = 75
    UMAP_kwargs["embedding_min_dist"] = 0.0
    UMAP_kwargs["embedding_spread"] = 1.0
    UMAP_kwargs["embedding_n_components"] = 2
    UMAP_kwargs["embedding_metric"] = "hellinger"
    UMAP_kwargs["embedding_low_memory"] = False
    use_annotations_to_mask = (True, False)
    embedding_kwargs = {
        **UMAP_kwargs,
        "use_annotations_to_mask": use_annotations_to_mask,
    }

    HDBSCAN_kwargs = {}
    HDBSCAN_kwargs["prediction_data"] = True
    HDBSCAN_kwargs["approx_min_span_tree"] = True
    HDBSCAN_kwargs["cluster_selection_method"] = "eom"
    HDBSCAN_kwargs["cluster_selection_epsilon"] = 0.0
    HDBSCAN_kwargs["min_cluster_size"] = 500
    HDBSCAN_kwargs["min_cluster_samples"] = 5
    clustering_kwargs = {**HDBSCAN_kwargs}

    mapping_postprocessing_kwargs = {}
    behavior_correspondence_kwargs = {}

    log_params(
        args.main_cfg_path,
        "embedding",
        embedding_kwargs,
    )
    log_params(args.main_cfg_path, "clustering", clustering_kwargs)
    log_params(
        args.main_cfg_path, "behavior_correspondence", behavior_correspondence_kwargs
    )
    log_params(
        args.main_cfg_path, "mapping_postprocessing", mapping_postprocessing_kwargs
    )

    behavior_mapper = BehaviorMapping(
        args.main_cfg_path,
        **embedding_kwargs,
        **clustering_kwargs,
        **behavior_correspondence_kwargs,
        **mapping_postprocessing_kwargs
    )

    # Embedding related procedures.
    if args.compute_supervised_disparate_embeddings:
        behavior_mapper.compute_supervised_disparate_embeddings()
    if args.compute_supervised_joint_embeddings:
        behavior_mapper.compute_supervised_joint_embeddings()

    if args.compute_unsupervised_disparate_embeddings:
        behavior_mapper.compute_unsupervised_disparate_embeddings()
    if args.compute_unsupervised_joint_embeddings:
        behavior_mapper.compute_unsupervised_joint_embeddings()

    if args.compute_semisupervised_pair_embeddings:
        behavior_mapper.compute_semisupervised_pair_embeddings()

    # Clustering related procedures.
    if args.jointly_cluster_supervised_joint:
        behavior_mapper.jointly_cluster_supervised_joint()
    if args.jointly_cluster_unsupervised_joint:
        behavior_mapper.jointly_cluster_unsupervised_joint()
    if args.jointly_cluster_semisupervised_pair:
        behavior_mapper.jointly_cluster_semisupervised_pair()

    if args.disparately_cluster_supervised_joint:
        behavior_mapper.disparately_cluster_supervised_joint()
    if args.disparately_cluster_unsupervised_joint:
        behavior_mapper.disparately_cluster_unsupervised_joint()
    if args.disparately_cluster_supervised_disparate:
        behavior_mapper.disparately_cluster_supervised_disparate()
    if args.disparately_cluster_unsupervised_disparate:
        behavior_mapper.disparately_cluster_unsupervised_disparate()
    if args.disparately_cluster_semisupervised_pair:
        behavior_mapper.disparately_cluster_semisupervised_pair()

    if args.crosswisely_cluster_semisupervised_pair:
        behavior_mapper.crosswisely_cluster_semisupervised_pair()

    # Correspondence related procedures.
    if args.map_disparate_cluster_supervised_disparate:
        behavior_mapper.map_disparate_cluster_supervised_disparate()
    if args.disparately_compute_behavior_score_disparate_cluster_supervised_disparate:
        behavior_mapper.disparately_compute_behavior_score_disparate_cluster_supervised_disparate()
    if args.map_disparate_cluster_supervised_joint:
        behavior_mapper.map_disparate_cluster_supervised_joint()
    if args.map_disparate_cluster_unsupervised_disparate:
        behavior_mapper.map_disparate_cluster_unsupervised_disparate()
    if args.map_disparate_cluster_unsupervised_joint:
        behavior_mapper.map_disparate_cluster_unsupervised_joint()
    if args.map_disparate_cluster_semisupervised_pair:
        behavior_mapper.map_disparate_cluster_semisupervised_pair()

    if args.map_joint_cluster_supervised_joint:
        behavior_mapper.map_joint_cluster_supervised_joint()
    if args.map_joint_cluster_unsupervised_joint:
        behavior_mapper.map_joint_cluster_unsupervised_joint()
    if args.map_joint_cluster_semisupervised_pair:
        behavior_mapper.map_joint_cluster_semisupervised_pair()

    if args.map_crosswise_cluster_semisupervised_pair:
        behavior_mapper.map_crosswise_cluster_semisupervised_pair()
    if args.crosswisely_compute_behavior_score_crosswise_cluster_semisupervised_pair:
        behavior_mapper.crosswisely_compute_behavior_score_crosswise_cluster_semisupervised_pair()
