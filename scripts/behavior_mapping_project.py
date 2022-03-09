import argparse

from pathlib import Path

from basty.utils.io import read_config
from basty.project.behavior_mapping import BehaviorMapping

parser = argparse.ArgumentParser(
    description="Initialize project  based on a given main configuration."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
parser.add_argument(
    "--reset-project",
    action=argparse.BooleanOptionalAction,
    help="Option to create a new project.",
)
parser.add_argument(
    "--compute-supervised-embeddings",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--compute-unsupervised-embeddings",
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
    "--disparately-cluster-supervised",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--disparately-cluster-unsupervised",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--disparately-cluster-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--compute-cross-pair-cluster-membership-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--map-disparate-cluster-labels-supervised",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-disparate-cluster-labels-supervised-joint",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-disparate-cluster-labels-unsupervised",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-disparate-cluster-labels-unsupervised-joint",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--map-joint-cluster-labels-supervised-joint",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-joint-cluster-labels-unsupervised-joint",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--map-disparate-cluster-labels-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-joint-cluster-labels-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--map-cross-pair-cluster-labels-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--compute-cross-pair-behavior-membership-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--compute-cross-pair-mean-behavior-membership-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)

args = parser.parse_args()


def backup_old_project(main_cfg_path):
    main_cfg = read_config(main_cfg_path)
    project_path = main_cfg.get("project_path")
    old_proj_dir = Path(project_path)
    if old_proj_dir.exists():
        suffix = ".old"
        while old_proj_dir.with_suffix(suffix).exists():
            suffix = suffix + ".old"
        old_proj_dir.replace(old_proj_dir.with_suffix(suffix))


if __name__ == "__main__":
    """
    Add suffix '.old' to the project created by previous tests.
    """
    if args.reset_project:
        backup_old_project(args.main_cfg_path)

    UMAP_kwargs = {}
    UMAP_kwargs["n_neighbors"] = 90
    UMAP_kwargs["min_dist"] = 0.0
    UMAP_kwargs["spread"] = 1.0
    UMAP_kwargs["n_components"] = 2
    UMAP_kwargs["metric"] = "hellinger"
    UMAP_kwargs["low_memory"] = True

    HDBSCAN_kwargs = {}
    HDBSCAN_kwargs["prediction_data"] = True
    HDBSCAN_kwargs["approx_min_span_tree"] = True
    HDBSCAN_kwargs["cluster_selection_method"] = "eom"
    HDBSCAN_kwargs["cluster_selection_epsilon"] = 0.0
    HDBSCAN_kwargs["min_cluster_size"] = 500
    HDBSCAN_kwargs["min_samples"] = 5

    mapping_post_processing_kwargs = {}

    behavior_mapping = BehaviorMapping(
        args.main_cfg_path,
        **UMAP_kwargs,
        **HDBSCAN_kwargs,
        **mapping_post_processing_kwargs
    )

    if args.compute_supervised_embeddings:
        behavior_mapping.compute_supervised_embeddings()
    if args.compute_unsupervised_embeddings:
        behavior_mapping.compute_unsupervised_embeddings()
    if args.compute_unsupervised_joint_embeddings:
        behavior_mapping.compute_unsupervised_joint_embeddings()
    if args.compute_supervised_joint_embeddings:
        behavior_mapping.compute_supervised_joint_embeddings()
    if args.compute_semisupervised_pair_embeddings:
        behavior_mapping.compute_semisupervised_pair_embeddings()

    if args.jointly_cluster_supervised_joint:
        behavior_mapping.jointly_cluster_supervised_joint()
    if args.jointly_cluster_unsupervised_joint:
        behavior_mapping.jointly_cluster_unsupervised_joint()
    if args.jointly_cluster_semisupervised_pair:
        behavior_mapping.jointly_cluster_semisupervised_pair()

    if args.disparately_cluster_supervised_joint:
        behavior_mapping.disparately_cluster_supervised_joint()
    if args.disparately_cluster_unsupervised_joint:
        behavior_mapping.disparately_cluster_unsupervised_joint()
    if args.disparately_cluster_supervised:
        behavior_mapping.disparately_cluster_supervised()
    if args.disparately_cluster_unsupervised:
        behavior_mapping.disparately_cluster_unsupervised()
    if args.disparately_cluster_semisupervised_pair:
        behavior_mapping.disparately_cluster_semisupervised_pair()

    if args.compute_cross_pair_cluster_membership_semisupervised_pair:
        behavior_mapping.compute_cross_pair_cluster_membership_semisupervised_pair()

    if args.map_disparate_cluster_labels_supervised:
        behavior_mapping.map_disparate_cluster_labels_supervised()
    if args.map_disparate_cluster_labels_supervised_joint:
        behavior_mapping.map_disparate_cluster_labels_supervised_joint()
    if args.map_disparate_cluster_labels_unsupervised:
        behavior_mapping.map_disparate_cluster_labels_unsupervised()
    if args.map_disparate_cluster_labels_unsupervised_joint:
        behavior_mapping.map_disparate_cluster_labels_unsupervised_joint()

    if args.map_joint_cluster_labels_supervised_joint:
        behavior_mapping.map_joint_cluster_labels_supervised_joint()
    if args.map_joint_cluster_labels_unsupervised_joint:
        behavior_mapping.map_joint_cluster_labels_unsupervised_joint()

    if args.map_disparate_cluster_labels_semisupervised_pair:
        behavior_mapping.map_disparate_cluster_labels_semisupervised_pair()
    if args.map_joint_cluster_labels_semisupervised_pair:
        behavior_mapping.map_joint_cluster_labels_semisupervised_pair()
    if args.map_cross_pair_cluster_labels_semisupervised_pair:
        behavior_mapping.map_cross_pair_cluster_labels_semisupervised_pair()

    if args.compute_cross_pair_behavior_membership_semisupervised_pair:
        behavior_mapping.compute_cross_pair_behavior_membership_semisupervised_pair()
    if args.compute_cross_pair_mean_behavior_membership_semisupervised_pair:
        behavior_mapping.compute_cross_pair_mean_behavior_membership_semisupervised_pair()
