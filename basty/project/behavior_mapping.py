import umap
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from hdbscan import HDBSCAN, membership_vector, all_points_membership_vectors

import basty.utils.misc as misc

from basty.project.experiment_processing import Project


EPS = 10 ** (-5)


class BehaviorMixin(Project):
    def __init__(
        self,
        main_cfg_path,
        **kwargs,
    ):
        Project.__init__(self, main_cfg_path, **kwargs)
        self.init_mapping_postprocessing_kwargs(**kwargs)

    def assert_compatible_apporach(
        self, expt_name1, embedding_name1, expt_name2, embedding_name2
    ):
        if expt_name1 in embedding_name2 and expt_name2 in embedding_name1:
            approach1 = embedding_name2.replace(expt_name1, "")
            approach2 = embedding_name1.replace(expt_name2, "")
        else:
            approach1 = embedding_name1
            approach2 = embedding_name2

        is_same_approach = approach1 == approach2

        if not is_same_approach:
            self.logger.direct_error(
                f"Given embedding approaches {approach1} and {approach2}) are not same."
                "Hence they are not compatible."
            )
        return is_same_approach


class BehaviorEmbedding(BehaviorMixin):
    def __init__(
        self,
        main_cfg_path,
        **kwargs,
    ):
        BehaviorMixin.__init__(self, main_cfg_path, **kwargs)
        self.init_behavior_embeddings_kwargs(**kwargs)

    @misc.timeit
    def compute_behavior_embedding(self, unannotated_expt_names, annotated_expt_names):
        all_valid_expt_names = list(self.expt_path_dict.keys())
        is_unannotated_valid = all(
            [expt_name in all_valid_expt_names for expt_name in unannotated_expt_names]
        )
        is_annotated_valid = all(
            [expt_name in all_valid_expt_names for expt_name in annotated_expt_names]
        )
        assert is_unannotated_valid and is_annotated_valid
        assert unannotated_expt_names or annotated_expt_names
        assert not (bool(set(unannotated_expt_names) & set(annotated_expt_names)))

        X_expt_dict = defaultdict()
        y_expt_dict = defaultdict()
        expt_indices_dict = defaultdict(tuple)

        def iterate_expt_for_embedding(expt_name, pbar):
            pbar.set_description(
                f"Loading behavioral reprs. and annotations (if supervised) of {expt_name}"
            )
            expt_path = self.expt_path_dict[expt_name]
            expt_record = self._load_joblib_object(expt_path, "expt_record.z")
            X_expt = self._load_numpy_array(expt_path, "behavioral_reprs.npy")
            return X_expt, expt_record, expt_path

        prev = 0
        pbar = tqdm(unannotated_expt_names)
        for expt_name in pbar:
            X_expt, expt_record, _ = iterate_expt_for_embedding(expt_name, pbar)
            y_expt = np.zeros(X_expt.shape[0], dtype=int) - 1

            mask_dormant = expt_record.mask_dormant
            mask_active = expt_record.mask_active

            X_expt_dict[expt_name] = X_expt[mask_dormant & mask_active]
            y_expt_dict[expt_name] = y_expt[mask_dormant & mask_active]

            expt_indices_dict[expt_name] = prev, prev + y_expt_dict[expt_name].shape[0]
            prev = expt_indices_dict[expt_name][-1]

        pbar = tqdm(annotated_expt_names)
        for expt_name in pbar:
            X_expt, expt_record, expt_path = iterate_expt_for_embedding(expt_name, pbar)

            assert expt_record.has_annotation
            mask_annotated = expt_record.mask_annotated
            mask_dormant = expt_record.mask_dormant
            y_expt = self._load_numpy_array(expt_path, "annotations.npy")

            X_expt_dict[expt_name] = X_expt[mask_dormant & mask_annotated]
            y_expt_dict[expt_name] = y_expt[mask_dormant & mask_annotated]

            expt_indices_dict[expt_name] = (
                prev,
                prev + y_expt_dict[expt_name].shape[0],
            )
            prev = expt_indices_dict[expt_name][-1]

        X = np.concatenate(list(X_expt_dict.values()), axis=0)
        y = np.concatenate(list(y_expt_dict.values()), axis=0)

        umap_transformer = umap.UMAP(**self.UMAP_kwargs)
        if annotated_expt_names:
            embedding = umap_transformer.fit_transform(X, y=y)
        else:
            embedding = umap_transformer.fit_transform(X)

        return embedding, expt_indices_dict

    @misc.timeit
    def compute_semisupervised_pair_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        assert all_expt_names
        assert annotated_expt_names
        assert unannotated_expt_names

        pbar = tqdm(
            misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
        )
        for ann_expt_name, unann_expt_name in pbar:
            pair_name_msg = (
                f"(annotated) {ann_expt_name} and (unannotated) {unann_expt_name}"
            )
            pbar.set_description(
                f"Computing semisupervised embeddding for {pair_name_msg}"
            )
            embedding, expt_indices_dict = self.compute_behavior_embedding(
                [unann_expt_name], [ann_expt_name]
            )

            expt_path = self.expt_path_dict[unann_expt_name]
            start, end = expt_indices_dict[unann_expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                expt_path / "embeddings",
                f"semisupervised_pair_embedding_{ann_expt_name}.npy",
                depth=3,
            )

            expt_path = self.expt_path_dict[ann_expt_name]
            start, end = expt_indices_dict[ann_expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                expt_path / "embeddings",
                f"semisupervised_pair_embedding_{unann_expt_name}.npy",
                depth=3,
            )

    @misc.timeit
    def compute_unsupervised_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        assert all_expt_names

        pbar = tqdm(all_expt_names)
        for expt_name in pbar:
            pbar.set_description(f"Computing unsupervised embeddding for {expt_name}")
            embedding, expt_indices_dict = self.compute_behavior_embedding(
                [expt_name], []
            )
            expt_path = self.expt_path_dict[expt_name]
            start, end = expt_indices_dict[expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                expt_path / "embeddings",
                "unsupervised_embedding.npy",
                depth=3,
            )

    @misc.timeit
    def compute_supervised_embeddings(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        assert annotated_expt_names

        pbar = tqdm(annotated_expt_names)
        for ann_expt_name in pbar:
            pbar.set_description(
                f"Computing unsupervised embeddding for {ann_expt_name}"
            )
            embedding, expt_indices_dict = self.compute_behavior_embedding(
                [], [ann_expt_name]
            )
            expt_path = self.expt_path_dict[ann_expt_name]
            start, end = expt_indices_dict[ann_expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                expt_path / "embeddings",
                "supervised_embedding.npy",
                depth=3,
            )

    @misc.timeit
    def compute_unsupervised_joint_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        assert all_expt_names
        embedding, expt_indices_dict = self.compute_behavior_embedding(
            all_expt_names, []
        )

        pbar = tqdm(all_expt_names)
        for expt_name in all_expt_names:
            pbar.set_description(
                "Computing joint unsupervised embeddding for all experiments"
            )
            expt_path = self.expt_path_dict[expt_name]
            start, end = expt_indices_dict[expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                expt_path / "embeddings",
                "unsupervised_joint_embedding.npy",
                depth=3,
            )

    @misc.timeit
    def compute_supervised_joint_embeddings(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        assert annotated_expt_names
        embedding, expt_indices_dict = self.compute_behavior_embedding(
            [], annotated_expt_names
        )

        pbar = tqdm(annotated_expt_names)
        for ann_expt_name in pbar:
            pbar.set_description(
                "Computing joint unsupervised embeddding for annotated experiments"
            )
            expt_path = self.expt_path_dict[ann_expt_name]
            start, end = expt_indices_dict[ann_expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                expt_path / "embeddings",
                "supervised_joint_embedding.npy",
                depth=3,
            )


class BehaviorClustering(BehaviorMixin):
    def __init__(
        self,
        main_cfg_path,
        **kwargs,
    ):
        BehaviorMixin.__init__(self, main_cfg_path, **kwargs)
        self.init_behavior_clustering_kwargs(**kwargs)

    @misc.timeit
    def jointly_cluster(self, expt_names, embedding_names):
        embedding_expt_dict = defaultdict()
        expt_indices_dict = defaultdict(tuple)

        prev = 0
        pbar = tqdm(expt_names)
        for i, expt_name in enumerate(pbar):
            embedding_name = embedding_names[i]
            embedding_name_msg = " ".join(embedding_name.split("_"))
            self.logger.direct_info(
                f"Loading {embedding_name_msg} of {expt_name} for joint clustering"
            )
            expt_path = self.expt_path_dict[expt_name]
            embedding_expt = self._load_numpy_array(
                expt_path / "embeddings", f"{embedding_name}.npy"
            )

            embedding_expt_dict[expt_name] = embedding_expt
            expt_indices_dict[expt_name] = prev, prev + embedding_expt.shape[0]
            prev = expt_indices_dict[expt_name][-1]

        embedding = np.concatenate(list(embedding_expt_dict.values()), axis=0)
        clusterer = HDBSCAN(**self.HDBSCAN_kwargs)
        cluster_labels = (clusterer.fit_predict(embedding) + 1).astype(int)

        pbar = tqdm(expt_names)
        for i, expt_name in enumerate(pbar):
            embedding_name = embedding_names[i]
            expt_path = self.expt_path_dict[expt_name]
            start, end = expt_indices_dict[expt_name]
            expt_indices_dict[expt_name] = prev, prev + embedding_expt.shape[0]
            cluster_labels_expt = cluster_labels[start:end]
            self._save_numpy_array(
                cluster_labels_expt,
                expt_path / "clusterings",
                f"labels_joint_cluster_{embedding_name}.npy",
                depth=3,
            )
            cluster_membership = all_points_membership_vectors(clusterer)
            cluster_membership = np.hstack(
                (
                    1 - np.sum(cluster_membership[:, 1:], axis=1, keepdims=True),
                    cluster_membership,
                )
            )
            self._save_numpy_array(
                cluster_membership,
                expt_path / "clusterings",
                f"membership_joint_cluster_{embedding_name}.npy",
                depth=3,
            )

    @misc.timeit
    def jointly_cluster_supervised_joint(self):
        ann_expt_names = list(self.annotation_path_dict.keys())
        embedding_names = ["supervised_joint_embedding" for _ in ann_expt_names]
        self.jointly_cluster(ann_expt_names, embedding_names)

    @misc.timeit
    def jointly_cluster_unsupervised_joint(self):
        all_expt_names = list(self.expt_path_dict.keys())
        embedding_names = ["unsupervised_joint_embedding" for _ in all_expt_names]
        self.jointly_cluster(all_expt_names, embedding_names)

    @misc.timeit
    def jointly_cluster_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        for ann_expt_name, unann_expt_name in misc.list_cartesian_product(
            annotated_expt_names, unannotated_expt_names
        ):
            embedding_names = [
                f"semisupervised_pair_embedding_{ann_expt_name}",
                f"semisupervised_pair_embedding_{unann_expt_name}",
            ]
            self.jointly_cluster([unann_expt_name, ann_expt_name], embedding_names)

    @misc.timeit
    def disparately_cluster(self, expt_names, embedding_names):
        pbar = tqdm(expt_names)
        for i, expt_name in enumerate(pbar):
            embedding_name = embedding_names[i]
            embedding_name_msg = " ".join(embedding_name.split("_"))
            pbar.set_description(
                f"Disparately clustering {embedding_name_msg} of {expt_name}"
            )
            expt_path = self.expt_path_dict[expt_name]
            embedding_expt = self._load_numpy_array(
                expt_path / "embeddings", f"{embedding_name}.npy"
            )
            clusterer = HDBSCAN(**self.HDBSCAN_kwargs)
            cluster_labels = (clusterer.fit_predict(embedding_expt) + 1).astype(int)
            self._save_numpy_array(
                cluster_labels,
                expt_path / "clusterings",
                f"labels_disparate_cluster_{embedding_name}.npy",
                depth=3,
            )
            cluster_membership = all_points_membership_vectors(clusterer)
            cluster_membership = np.hstack(
                (
                    1 - np.sum(cluster_membership[:, 1:], axis=1, keepdims=True),
                    cluster_membership,
                )
            )
            self._save_numpy_array(
                cluster_membership,
                expt_path / "clusterings",
                f"membership_disparate_cluster_{embedding_name}.npy",
                depth=3,
            )

    @misc.timeit
    def disparately_cluster_supervised_joint(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        embedding_name = ["supervised_joint_embedding" for _ in annotated_expt_names]
        self.disparately_cluster(annotated_expt_names, embedding_name)

    @misc.timeit
    def disparately_cluster_unsupervised_joint(self):
        all_expt_names = list(self.expt_path_dict.keys())
        embedding_name = ["unsupervised_joint_embedding" for _ in all_expt_names]
        self.disparately_cluster(all_expt_names, embedding_name)

    @misc.timeit
    def disparately_cluster_supervised(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        embedding_name = ["supervised_embedding" for _ in annotated_expt_names]
        self.disparately_cluster(annotated_expt_names, embedding_name)

    @misc.timeit
    def disparately_cluster_unsupervised(self):
        all_expt_names = list(self.expt_path_dict.keys())
        embedding_name = ["unsupervised_embedding" for _ in all_expt_names]
        self.disparately_cluster(all_expt_names, embedding_name)

    @misc.timeit
    def disparately_cluster_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        for ann_expt_name, unann_expt_name in misc.list_cartesian_product(
            annotated_expt_names, unannotated_expt_names
        ):
            embedding_names = [f"semisupervised_pair_embedding_{unann_expt_name}"]
            self.disparately_cluster([ann_expt_name], embedding_names)
            # Not really needed.
            # embedding_names = [f"semisupervised_pair_embedding_{ann_expt_name}"]
            # self.disparately_cluster([unann_expt_name], embedding_names)

    @misc.timeit
    def compute_cross_pair_cluster_membership(
        self, expt_name1, expt_name2, embedding_name1, embedding_name2
    ):
        expt_path1 = self.expt_path_dict[expt_name1]
        expt_path2 = self.expt_path_dict[expt_name2]

        assert self.assert_compatible_apporach(
            expt_name1, embedding_name1, expt_name2, embedding_name2
        )

        embedding_expt1 = self._load_numpy_array(
            expt_path1 / "embeddings", f"{embedding_name1}.npy"
        )
        embedding_expt2 = self._load_numpy_array(
            expt_path2 / "embeddings", f"{embedding_name2}.npy"
        )

        clusterer = HDBSCAN(**self.HDBSCAN_kwargs)
        cluster_labels = clusterer.fit_predict(embedding_expt1) + 1
        cluster_membership = membership_vector(clusterer, embedding_expt2)
        cluster_membership = np.hstack(
            (
                1 - np.sum(cluster_membership[:, 1:], axis=1, keepdims=True),
                cluster_membership,
            )
        )

        self._save_numpy_array(
            cluster_membership,
            expt_path2 / "clusterings",
            f"membership_cross_pair_cluster_{embedding_name2}_{embedding_name1}.npy",
            depth=3,
        )
        self._save_numpy_array(
            cluster_labels,
            expt_path1 / "clusterings",
            f"labels_cross_pair_cluster_{embedding_name1}_{embedding_name2}.npy",
            depth=3,
        )

    @misc.timeit
    def compute_cross_pair_cluster_membership_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))

        pbar = tqdm(
            misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
        )
        for ann_expt_name, unann_expt_name in pbar:
            ann_embedding_name = f"semisupervised_pair_embedding_{unann_expt_name}"
            unann_embedding_name = f"semisupervised_pair_embedding_{ann_expt_name}"
            pair_name_msg = f"{unann_embedding_name} with {ann_embedding_name}"
            pbar.set_description(
                f"Computing cross pair cluster membership {pair_name_msg}"
            )
            self.compute_cross_pair_cluster_membership(
                ann_expt_name, unann_expt_name, ann_embedding_name, unann_embedding_name
            )
            # Not really needed.
            # self.compute_cross_pair_cluster_membership(
            #     unann_expt_name, ann_expt_name, unann_embedding_name, ann_embedding_name
            # )
            # pair_name_msg = f"{ann_embedding_name} with {unann_embedding_name}"
            # pbar.set_description(
            #     f"Computing cross pair cluster membership {pair_name_msg}"
            # )


class BehaviorMapping(BehaviorEmbedding, BehaviorClustering):
    def __init__(
        self,
        main_cfg_path,
        **kwargs,
    ):
        BehaviorEmbedding.__init__(self, main_cfg_path, **kwargs)
        BehaviorClustering.__init__(self, main_cfg_path, **kwargs)

    @misc.timeit
    def map_cluster_labels_to_behavior_labels(self, expt_name, clustering_name):
        expt_path = self.expt_path_dict[expt_name]
        expt_record = self._load_joblib_object(expt_path, "expt_record.z")

        assert expt_record.has_annotation
        y_ann = self._load_numpy_array(expt_path, "annotations.npy")

        unsupervised_embedding_names = [
            "unsupervised_embedding",
            "unsupervised_joint_embedding",
        ]
        if any([name in clustering_name for name in unsupervised_embedding_names]):
            y_ann = y_ann[expt_record.mask_dormant & expt_record.mask_active]
        else:
            y_ann = y_ann[expt_record.mask_dormant & expt_record.mask_annotated]

        y_cluster = self._load_numpy_array(
            expt_path / "clusterings", f"labels_{clustering_name}.npy"
        )

        mapping_dictionary = defaultdict(dict)
        y_cluster_uniq = np.unique(y_cluster)
        for cluster_lbl in y_cluster_uniq:
            y_ann_masked = y_ann[y_cluster == cluster_lbl]
            y_ann_uniq_c, ann_uniq_c_counts = np.unique(
                y_ann_masked, return_counts=True
            )

            mapping_dictionary[int(cluster_lbl)] = {
                key: 0 for key in expt_record.label_to_behavior.keys()
            }
            # mapping_dictionary[int(cluster_lbl)] = {
            #     key: 0 for key in expt_record.behavior_to_label.keys()
            # }

            for idx, ann_lbl in enumerate(y_ann_uniq_c):
                mapping_dictionary[int(cluster_lbl)][int(ann_lbl)] = float(
                    ann_uniq_c_counts[idx] / y_ann_masked.shape[0]
                )
                # ann_behavior = expt_record.label_to_behavior[ann_lbl]
                # mapping_dictionary[int(cluster_lbl)][ann_behavior] = float(
                #     ann_uniq_c_counts[idx] / y_ann_masked.shape[0]
                # )

            assert abs(sum(mapping_dictionary[int(cluster_lbl)].values()) - 1) < EPS

        self._save_yaml_dictionary(
            dict(mapping_dictionary),
            expt_path / "mappings",
            f"mapping_{clustering_name}.yaml",
            depth=3,
        )

    @misc.timeit
    def map_disparate_cluster_labels_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        pbar = tqdm(
            misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
        )
        for ann_expt_name, unann_expt_name in pbar:
            embedding_name = f"semisupervised_pair_embedding_{unann_expt_name}"
            clustering_name = f"disparate_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_disparate_cluster_labels_supervised(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        pbar = tqdm(annotated_expt_names)
        for ann_expt_name in pbar:
            embedding_name = "supervised_embedding"
            clustering_name = f"disparate_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_disparate_cluster_labels_supervised_joint(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        pbar = tqdm(annotated_expt_names)
        for ann_expt_name in pbar:
            embedding_name = "supervised_joint_embedding"
            clustering_name = f"disparate_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_disparate_cluster_labels_unsupervised(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        pbar = tqdm(annotated_expt_names)
        for ann_expt_name in pbar:
            embedding_name = "unsupervised_embedding"
            clustering_name = f"disparate_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_disparate_cluster_labels_unsupervised_joint(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        pbar = tqdm(annotated_expt_names)
        for ann_expt_name in pbar:
            embedding_name = "unsupervised_joint_embedding"
            clustering_name = f"disparate_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_joint_cluster_labels_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        pbar = tqdm(
            misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
        )
        for ann_expt_name, unann_expt_name in pbar:
            embedding_name = f"semisupervised_pair_embedding_{unann_expt_name}"
            clustering_name = f"joint_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_joint_cluster_labels_supervised_joint(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        pbar = tqdm(annotated_expt_names)
        for ann_expt_name in pbar:
            embedding_name = "supervised_joint_embedding"
            clustering_name = f"joint_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_joint_cluster_labels_unsupervised_joint(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        pbar = tqdm(annotated_expt_names)
        for ann_expt_name in pbar:
            embedding_name = "unsupervised_joint_embedding"
            clustering_name = f"joint_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_cross_pair_cluster_labels_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        pbar = tqdm(
            misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
        )
        for ann_expt_name, unann_expt_name in pbar:
            embedding_name1 = f"semisupervised_pair_embedding_{unann_expt_name}"
            embedding_name2 = f"semisupervised_pair_embedding_{ann_expt_name}"
            clustering_name = f"cross_pair_cluster_{embedding_name1}_{embedding_name2}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def compute_behavior_membership(self, expt_name, clustering_name):
        expt_path = self.expt_path_dict[expt_name]

        expt_record = self._load_joblib_object(expt_path, "expt_record.z")

        assert expt_record.has_annotation

        cluster_membership = self._load_numpy_array(
            expt_path / "clusterings",
            f"membership_{clustering_name}.npy",
        )
        mapping = self._load_yaml_dictionary(
            expt_path / "mappings",
            f"mapping_{clustering_name}.yaml",
        )

        behavior_membership = np.zeros(
            (cluster_membership.shape[0], len(expt_record.label_to_behavior))
        )
        for cluster_lbl, behavior_weights in mapping.items():
            for behavior_lbl, weight in behavior_weights.items():
                behavior_membership[:, behavior_lbl] = (
                    behavior_membership[:, behavior_lbl]
                    + cluster_membership[:, cluster_lbl] * weight
                )
        self._save_numpy_array(
            behavior_membership,
            expt_path / "mappings",
            f"membership_{clustering_name.replace('cluster', 'behavior')}.npy",
            depth=3,
        )

    @misc.timeit
    def compute_cross_pair_behavior_membership(
        self, expt_name1, expt_name2, embedding_name1, embedding_name2
    ):
        expt_path1 = self.expt_path_dict[expt_name1]
        expt_path2 = self.expt_path_dict[expt_name2]

        expt_record1 = self._load_joblib_object(expt_path1, "expt_record.z")

        assert expt_record1.has_annotation
        assert self.assert_compatible_apporach(
            expt_name1, embedding_name1, expt_name2, embedding_name2
        )

        cluster_membership = self._load_numpy_array(
            expt_path2 / "clusterings",
            f"membership_cross_pair_cluster_{embedding_name2}_{embedding_name1}.npy",
        )
        # cluster_labels = self._load_numpy_array(
        #     expt_path1 / "clusterings",
        #     f"cross_pair_cluster_labels_{embedding_name1}_{embedding_name2}.npy",
        # )
        mapping = self._load_yaml_dictionary(
            expt_path1 / "mappings",
            f"mapping_cross_pair_cluster_{embedding_name1}_{embedding_name2}.yaml",
        )

        behavior_membership = np.zeros(
            (cluster_membership.shape[0], len(expt_record1.label_to_behavior))
        )
        for cluster_lbl, behavior_weights in mapping.items():
            for behavior_lbl, weight in behavior_weights.items():
                behavior_membership[:, behavior_lbl] = (
                    behavior_membership[:, behavior_lbl]
                    + cluster_membership[:, cluster_lbl] * weight
                )
        self._save_numpy_array(
            behavior_membership,
            expt_path2 / "mappings",
            f"membership_cross_pair_behavior_{embedding_name2}_{embedding_name1}.npy",
            depth=3,
        )

    @misc.timeit
    def compute_cross_pair_behavior_membership_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))

        pbar = tqdm(
            misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
        )
        for ann_expt_name, unann_expt_name in pbar:
            ann_embedding_name = f"semisupervised_pair_embedding_{unann_expt_name}"
            unann_embedding_name = f"semisupervised_pair_embedding_{ann_expt_name}"
            pair_name_msg = f"{unann_embedding_name} with {ann_embedding_name}"
            pbar.set_description(
                f"Computing cross pair behavior membership {pair_name_msg}"
            )
            self.compute_cross_pair_behavior_membership(
                ann_expt_name, unann_expt_name, ann_embedding_name, unann_embedding_name
            )

    @misc.timeit
    def compute_cross_pair_mean_behavior_membership_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))

        pbar = tqdm(
            misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
        )
        behavior_membership_dict = defaultdict(list)
        for ann_expt_name, unann_expt_name in pbar:
            unann_expt_path = self.expt_path_dict[unann_expt_name]
            ann_embedding_name = f"semisupervised_pair_embedding_{unann_expt_name}"
            unann_embedding_name = f"semisupervised_pair_embedding_{ann_expt_name}"
            behavior_membership = self._load_numpy_array(
                unann_expt_path / "mappings",
                f"membership_cross_pair_behavior_{unann_embedding_name}_{ann_embedding_name}.npy",
            )
            behavior_membership_dict[unann_expt_name].append(behavior_membership)

        for (unann_expt_name, behavior_memberships) in behavior_membership_dict.items():
            unann_expt_path = self.expt_path_dict[unann_expt_name]
            mean_behavior_membership = np.mean(behavior_memberships, axis=0)
            embedding_name_msg = f"semisupervised_pair_embedding_{unann_expt_name}"
            pbar.set_description(
                f"Computing mean behavior membership of {embedding_name_msg}"
            )
            self._save_numpy_array(
                mean_behavior_membership,
                unann_expt_path / "mappings",
                "avg_membership_cross_pair_behavior_semisupervised_pair_embedding.npy",
                depth=3,
            )
