from collections import defaultdict

import numpy as np
import umap
from hdbscan import HDBSCAN  # approximate_predict,
from hdbscan import all_points_membership_vectors, membership_vector
from sklearn.preprocessing import normalize, scale
from tqdm import tqdm

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
        self.init_behavior_mapping_postprocessing_kwargs(**kwargs)

    def is_compatible_approach(self, expt_name1, name1, expt_name2, name2):
        if expt_name1 in name2 and expt_name2 in name1:
            approach1 = name2.replace(expt_name1, "").replace(expt_name2, "")
            approach2 = name1.replace(expt_name2, "").replace(expt_name1, "")
        else:
            approach1 = name1
            approach2 = name2

        compatible = approach1 == approach2

        if not compatible:
            self.logger.direct_error(
                f"Given approaches {approach1} and {approach2}) are not same."
                "Hence they are not compatible."
            )
        return compatible


class BehaviorEmbedding(BehaviorMixin):
    def __init__(
        self,
        main_cfg_path,
        **kwargs,
    ):
        BehaviorMixin.__init__(self, main_cfg_path, **kwargs)
        self.init_behavior_embeddings_kwargs(**kwargs)

    def _update_expt_record(self, expt_path, embedding_name):
        expt_record = self._load_joblib_object(expt_path, "expt_record.z")
        if self.use_annotations_to_mask and expt_record.has_annotation:
            expt_record.use_annotations_to_mask[embedding_name] = True
        else:
            expt_record.use_annotations_to_mask[embedding_name] = False
        self._save_joblib_object(expt_record, expt_path, "expt_record.z")

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

        def iterate_expt_for_embedding(expt_name):
            expt_path = self.expt_path_dict[expt_name]
            expt_record = self._load_joblib_object(expt_path, "expt_record.z")
            X_expt = self._load_numpy_array(expt_path, "behavioral_reprs.npy")
            return X_expt, expt_record, expt_path

        prev = 0
        for expt_name in unannotated_expt_names:
            X_expt, expt_record, expt_path = iterate_expt_for_embedding(expt_name)
            y_expt = np.zeros(X_expt.shape[0], dtype=int) - 1

            if self.use_annotations_to_mask and expt_record.has_annotation:
                mask_active = expt_record.mask_annotated
            else:
                mask_active = expt_record.mask_active
            mask_dormant = expt_record.mask_dormant

            X_expt_dict[expt_name] = X_expt[mask_dormant & mask_active]
            y_expt_dict[expt_name] = y_expt[mask_dormant & mask_active]

            expt_indices_dict[expt_name] = prev, prev + y_expt_dict[expt_name].shape[0]
            prev = expt_indices_dict[expt_name][-1]

        for expt_name in annotated_expt_names:
            X_expt, expt_record, expt_path = iterate_expt_for_embedding(expt_name)

            assert expt_record.has_annotation
            y_expt = self._load_numpy_array(expt_path, "annotations.npy")

            if self.use_annotations_to_mask and expt_record.has_annotation:
                mask_active = expt_record.mask_annotated
            else:
                mask_active = expt_record.mask_active
            mask_dormant = expt_record.mask_dormant

            X_expt_dict[expt_name] = X_expt[mask_dormant & mask_active]
            y_expt_dict[expt_name] = y_expt[mask_dormant & mask_active]

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
        if self.evaluation_mode:
            unannotated_expt_names = annotated_expt_names

        assert all_expt_names
        assert annotated_expt_names
        assert unannotated_expt_names

        pairs = misc.list_cartesian_product(
            annotated_expt_names, unannotated_expt_names
        )
        pairs = [(name1, name2) for name1, name2 in pairs if name1 != name2]

        pbar = tqdm(pairs)
        for ann_expt_name, unann_expt_name in pbar:
            pair_name_msg = (
                f"(annotated) {ann_expt_name} and (unannotated) {unann_expt_name}"
            )
            pbar.set_description(
                f"Computing semisupervised embeddding for {pair_name_msg}"
            )

            embedding_type = "semisupervised_pair_embedding"

            unann_expt_path = self.expt_path_dict[unann_expt_name]
            unann_embedding_name = f"{embedding_type}_{ann_expt_name}_{unann_expt_name}"
            self._update_expt_record(unann_expt_path, unann_embedding_name)

            ann_expt_path = self.expt_path_dict[ann_expt_name]
            ann_embedding_name = f"{embedding_type}_{ann_expt_name}_{unann_expt_name}"
            self._update_expt_record(ann_expt_path, ann_embedding_name)

            embedding, expt_indices_dict = self.compute_behavior_embedding(
                [unann_expt_name], [ann_expt_name]
            )
            start, end = expt_indices_dict[unann_expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                unann_expt_path / "embeddings",
                f"{unann_embedding_name}.npy",
                depth=3,
            )
            start, end = expt_indices_dict[ann_expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                ann_expt_path / "embeddings",
                f"{ann_embedding_name}.npy",
                depth=3,
            )

    @misc.timeit
    def compute_unsupervised_disparate_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        assert all_expt_names

        pbar = tqdm(all_expt_names)
        for expt_name in pbar:
            pbar.set_description(
                f"Computing unsupervised disparate embeddding for {expt_name}"
            )

            expt_path = self.expt_path_dict[expt_name]
            embedding_name = "unsupervised_disparate_embedding"
            self._update_expt_record(expt_path, embedding_name)

            embedding, expt_indices_dict = self.compute_behavior_embedding(
                [expt_name], []
            )
            start, end = expt_indices_dict[expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                expt_path / "embeddings",
                f"{embedding_name}.npy",
                depth=3,
            )

    @misc.timeit
    def compute_supervised_disparate_embeddings(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        assert annotated_expt_names

        pbar = tqdm(annotated_expt_names)
        for ann_expt_name in pbar:
            pbar.set_description(
                f"Computing supervised disparate embeddding for {ann_expt_name}"
            )

            ann_expt_path = self.expt_path_dict[ann_expt_name]
            ann_embedding_name = "supervised_disparate_embedding"
            self._update_expt_record(ann_expt_path, ann_embedding_name)

            embedding, expt_indices_dict = self.compute_behavior_embedding(
                [], [ann_expt_name]
            )
            start, end = expt_indices_dict[ann_expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                ann_expt_path / "embeddings",
                f"{ann_embedding_name}.npy",
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
            embedding_name = "unsupervised_joint_embedding"
            self._update_expt_record(expt_path, embedding_name)

            start, end = expt_indices_dict[expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                expt_path / "embeddings",
                f"{embedding_name}.npy",
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

            ann_expt_path = self.expt_path_dict[ann_expt_name]
            ann_embedding_name = "supervised_joint_embedding"
            self._update_expt_record(ann_expt_path, ann_embedding_name)

            start, end = expt_indices_dict[ann_expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                ann_expt_path / "embeddings",
                f"{ann_embedding_name}.npy",
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
                f"Loading {embedding_name_msg} of {expt_name} for joint clustering."
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
            cluster_membership = all_points_membership_vectors(clusterer)[start:end]
            cluster_membership = np.hstack(
                (
                    1 - np.sum(cluster_membership[:, :], axis=1, keepdims=True),
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
                    1 - np.sum(cluster_membership[:, :], axis=1, keepdims=True),
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
    def disparately_cluster_supervised_disparate(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        embedding_name = [
            "supervised_disparate_embedding" for _ in annotated_expt_names
        ]
        self.disparately_cluster(annotated_expt_names, embedding_name)

    @misc.timeit
    def disparately_cluster_unsupervised_disparate(self):
        all_expt_names = list(self.expt_path_dict.keys())
        embedding_name = ["unsupervised_disparate_embedding" for _ in all_expt_names]
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
            embedding_names = [f"semisupervised_pair_embedding_{ann_expt_name}"]
            self.disparately_cluster([ann_expt_name], embedding_names)
            self.disparately_cluster([unann_expt_name], embedding_names)

    @misc.timeit
    def crosswisely_cluster(
        self, expt_names1, expt_names2, embedding_names1, embedding_names2
    ):
        embedding_expt_dict = defaultdict()
        expt_indices_dict = defaultdict(tuple)

        for idx, expt_name in enumerate(expt_names1):
            embedding_name1 = embedding_names1[idx]
            for idx1, expt_name1 in enumerate(expt_names1[(idx + 1) :]):
                embedding_name11 = embedding_names1[idx1]
                assert self.is_compatible_approach(
                    expt_name, embedding_name1, expt_name1, embedding_name11
                )
            for idx2, expt_name2 in enumerate(expt_names2):
                embedding_name2 = embedding_names2[idx2]
                assert self.is_compatible_approach(
                    expt_name, embedding_name1, expt_name2, embedding_name2
                )

        prev = 0
        pbar = tqdm(expt_names1)
        for i, expt_name in enumerate(pbar):
            embedding_name = embedding_names1[i]
            embedding_name_msg = " ".join(embedding_name.split("_"))
            self.logger.direct_info(
                f"Loading {embedding_name_msg} of {expt_name} for crosswise clustering."
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
        cluster_membership = all_points_membership_vectors(clusterer)
        clustered_expt_names = "_".join(expt_names1)

        pbar = tqdm(expt_names1)
        for i, expt_name in enumerate(pbar):
            embedding_name = embedding_names1[i]
            expt_path = self.expt_path_dict[expt_name]
            start, end = expt_indices_dict[expt_name]
            expt_indices_dict[expt_name] = prev, prev + embedding_expt.shape[0]

            cluster_membership_expt = cluster_membership[start:end]
            cluster_membership_expt = np.hstack(
                (
                    1 - np.sum(cluster_membership_expt[:, :], axis=1, keepdims=True),
                    cluster_membership_expt,
                )
            )
            self._save_numpy_array(
                cluster_membership_expt,
                expt_path / "clusterings",
                f"membership_crosswise_cluster_{clustered_expt_names}_{embedding_name}.npy",
                depth=3,
            )
            cluster_labels_expt = cluster_labels[start:end]
            self._save_numpy_array(
                cluster_labels_expt,
                expt_path / "clusterings",
                f"labels_crosswise_cluster_{clustered_expt_names}_{embedding_name}.npy",
                depth=3,
            )

        pbar = tqdm(expt_names2)
        for i, expt_name in enumerate(pbar):
            embedding_name = embedding_names2[i]
            embedding_name_msg = " ".join(embedding_name.split("_"))
            self.logger.direct_info(
                f"Crosswisely clustering {embedding_name_msg} of {expt_name}"
            )
            expt_path = self.expt_path_dict[expt_name]
            embedding_expt = self._load_numpy_array(
                expt_path / "embeddings", f"{embedding_name}.npy"
            )

            cluster_membership_expt = membership_vector(clusterer, embedding_expt)
            cluster_membership_expt = np.hstack(
                (
                    1 - np.sum(cluster_membership_expt[:, 1:], axis=1, keepdims=True),
                    cluster_membership_expt,
                )
            )
            self._save_numpy_array(
                cluster_membership_expt,
                expt_path / "clusterings",
                f"membership_crosswise_cluster_{clustered_expt_names}_{embedding_name}.npy",
                depth=3,
            )
            # cluster_labels_expt = (
            #     approximate_predict(clusterer, embedding_expt) + 1
            # ).astype(int)
            cluster_labels_expt = np.argmax(cluster_membership_expt, axis=1)
            self._save_numpy_array(
                cluster_labels_expt,
                expt_path / "clusterings",
                f"labels_crosswise_cluster_{clustered_expt_names}_{embedding_name}.npy",
                depth=3,
            )

    @misc.timeit
    def crosswisely_cluster_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))

        for ann_expt_name, unann_expt_name in misc.list_cartesian_product(
            annotated_expt_names, unannotated_expt_names
        ):
            embedding_type = "semisupervised_pair_embedding"
            ann_embedding_name = f"{embedding_type}_{ann_expt_name}_{unann_expt_name}"
            unann_embedding_name = f"{embedding_type}_{ann_expt_name}_{unann_expt_name}"
            self.crosswisely_cluster(
                [ann_expt_name],
                [unann_expt_name],
                [ann_embedding_name],
                [unann_embedding_name],
            )


class BehaviorCorrespondence(BehaviorMixin):
    def __init__(
        self,
        main_cfg_path,
        **kwargs,
    ):
        BehaviorMixin.__init__(self, main_cfg_path, **kwargs)
        self.init_behavior_correspondence_kwargs(**kwargs)

    @misc.timeit
    def map_cluster_labels_to_behavior_labels(self, expt_name, clustering_name):
        expt_path = self.expt_path_dict[expt_name]
        expt_record = self._load_joblib_object(expt_path, "expt_record.z")

        assert expt_record.has_annotation
        y_ann = self._load_numpy_array(expt_path, "annotations.npy")

        using_annotations_to_mask = [
            key for key, val in expt_record.use_annotations_to_mask.items() if val
        ]
        if any([name in clustering_name for name in using_annotations_to_mask]):
            y_ann = y_ann[expt_record.mask_dormant & expt_record.mask_annotated]
        else:
            y_ann = y_ann[expt_record.mask_dormant & expt_record.mask_active]

        y_cluster = self._load_numpy_array(
            expt_path / "clusterings", f"labels_{clustering_name}.npy"
        )

        mapping_dictionary = defaultdict(dict)
        y_cluster_uniq, cluster_uniq_counts = np.unique(y_cluster, return_counts=True)
        y_ann_uniq, ann_uniq_counts = np.unique(y_ann, return_counts=True)
        ann_counts_ref = {
            y_ann_uniq[i]: ann_uniq_counts[i] for i in range(y_ann_uniq.shape[0])
        }
        for idx1, cluster_lbl in enumerate(y_cluster_uniq):
            y_ann_masked = y_ann[y_cluster == cluster_lbl]
            y_ann_uniq_cluster, ann_uniq_cluster_counts = np.unique(
                y_ann_masked, return_counts=True
            )

            mapping_dictionary[int(cluster_lbl)] = {
                key: 0 for key in expt_record.label_to_behavior.keys()
            }

            for idx2, ann_lbl in enumerate(y_ann_uniq_cluster):
                ann_cluster_count = ann_uniq_cluster_counts[idx2]

                tf = ann_cluster_count / (cluster_uniq_counts[idx1] + 1)
                # tf = 0.5 + 0.5 * (ann_cluster_count / max(ann_uniq_cluster_counts))
                # tf = np.log2(ann_cluster_count / (cluster_uniq_counts[idx1] + 1))
                # tf = np.log2(1 + ann_cluster_count / (cluster_uniq_counts[idx1] + 1))
                # tf = np.log2(1 + ann_cluster_count)

                denom = cluster_uniq_counts[idx1] / ann_counts_ref[ann_lbl]
                # y_cluster_masked = y_cluster[y_ann == ann_lbl]
                # na_max = max(np.unique(y_cluster_masked, return_counts=True)[1])
                # denom = na_max / ann_counts_ref[ann_lbl]
                # nc = len(np.unique(y_cluster[y_ann == ann_lbl]))
                # max_count_ann_lbl = y_ann_uniq_cluster[np.argmax(ann_uniq_cluster_counts)]
                # nc_max = len(np.unique(y_cluster[y_ann == max_count_ann_lbl]))
                # idf = np.log2(nc_max / (nc + 1)) + 1
                # idf = np.log2(len(y_cluster_uniq) / (1 + nc)) + 1
                # denom = idf

                mapping_dictionary[cluster_lbl][ann_lbl] = float(tf * denom)

            # L1- normalization of mapping weights.
            sum_weights = sum(list(mapping_dictionary[int(cluster_lbl)].values()))
            for ann_lbl in y_ann_uniq_cluster:
                mapping_dictionary[cluster_lbl][ann_lbl] = (
                    mapping_dictionary[cluster_lbl][ann_lbl] / sum_weights
                )
            assert abs(sum(mapping_dictionary[int(cluster_lbl)].values()) - 1) < EPS

        self._save_yaml_dictionary(
            dict(mapping_dictionary),
            expt_path / "correspondences",
            f"mapping_{clustering_name}.yaml",
            depth=3,
        )

    @misc.timeit
    def map_disparate_cluster_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        pbar = tqdm(
            misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
        )
        for ann_expt_name, unann_expt_name in pbar:
            embedding_name = (
                f"semisupervised_pair_embedding_{ann_expt_name}_{unann_expt_name}"
            )
            clustering_name = f"disparate_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_disparate_cluster_supervised_disparate(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        pbar = tqdm(annotated_expt_names)
        for ann_expt_name in pbar:
            embedding_name = "supervised_disparate_embedding"
            clustering_name = f"disparate_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_disparate_cluster_supervised_joint(self):
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
    def map_disparate_cluster_unsupervised_disparate(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        pbar = tqdm(annotated_expt_names)
        for ann_expt_name in pbar:
            embedding_name = "unsupervised_disparate_embedding"
            clustering_name = f"disparate_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_disparate_cluster_unsupervised_joint(self):
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
    def map_joint_cluster_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        pbar = tqdm(
            misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
        )
        for ann_expt_name, unann_expt_name in pbar:
            embedding_name = (
                f"semisupervised_pair_embedding_{ann_expt_name}_{unann_expt_name}"
            )
            clustering_name = f"joint_cluster_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def map_joint_cluster_supervised_joint(self):
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
    def map_joint_cluster_unsupervised_joint(self):
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
    def map_crosswise_cluster_semisupervised_pair(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        pbar = tqdm(
            misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
        )
        for ann_expt_name, unann_expt_name in pbar:
            embedding_name = (
                f"semisupervised_pair_embedding_{ann_expt_name}_{unann_expt_name}"
            )
            clustering_name = f"crosswise_cluster_{ann_expt_name}_{embedding_name}"
            pbar.set_description(
                f"Mapping cluster labels of {clustering_name} to behavior labels"
            )
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    @misc.timeit
    def disparately_compute_behavior_score(self, expt_names, clustering_names):
        pbar = tqdm(expt_names)
        for i, expt_name in enumerate(pbar):
            expt_path = self.expt_path_dict[expt_name]
            clustering_name = clustering_names[i]

            expt_record = self._load_joblib_object(expt_path, "expt_record.z")
            assert expt_record.has_annotation
            label_to_behavior = expt_record.label_to_behavior
            behavior_to_label = expt_record.behavior_to_label
            inactive_annotation = expt_record.inactive_annotation

            num_behavior = len(label_to_behavior)
            assert num_behavior > 1

            mapping = self._load_yaml_dictionary(
                expt_path / "correspondences",
                f"mapping_{clustering_name}.yaml",
            )

            cluster_membership = self._load_numpy_array(
                expt_path / "clusterings",
                f"membership_{clustering_name}.npy",
            )
            behavior_score = np.zeros((cluster_membership.shape[0], num_behavior))
            for cluster_lbl, behavior_weights in mapping.items():
                for behavior_lbl, weight in behavior_weights.items():
                    if behavior_lbl == behavior_to_label[inactive_annotation]:
                        weight += (
                            np.sqrt(1 / (num_behavior - 1))
                            if not cluster_lbl
                            else 1 / (num_behavior - 1)
                        )
                    behavior_score[:, behavior_lbl] = (
                        behavior_score[:, behavior_lbl]
                        + cluster_membership[:, cluster_lbl]
                        * weight
                        * cluster_membership.shape[1]
                    )
            behavior_score = normalize(behavior_score, norm="l1")
            # behavior_score = scale(behavior_score, axis=0)

            self._save_numpy_array(
                behavior_score,
                expt_path / "correspondences",
                f"score_{clustering_name.replace('cluster', 'behavior')}.npy",
                depth=3,
            )

    @misc.timeit
    def crosswisely_compute_behavior_score(
        self, expt_names1, expt_names2, clustering_names1, clustering_names2
    ):
        total_mapping = defaultdict(dict)
        label_to_behavior = defaultdict()
        behavior_to_label = defaultdict()
        inactive_annotation = str()

        for idx1, expt_name1 in enumerate(expt_names1):
            clustering_name1 = clustering_names1[idx1]
            for idx2, expt_name2 in enumerate(expt_names2):
                clustering_name2 = clustering_names2[idx2]
                assert self.is_compatible_approach(
                    expt_name1, clustering_name1, expt_name2, clustering_name2
                )
        assert all(list(map(lambda x: "disparate_cluster" not in x, clustering_names1)))
        assert all(list(map(lambda x: "disparate_cluster" not in x, clustering_names2)))

        for idx, expt_name in enumerate(expt_names1):
            expt_path = self.expt_path_dict[expt_name]
            clustering_name = clustering_names1[idx]

            expt_record = self._load_joblib_object(expt_path, "expt_record.z")
            assert idx == 0 or (label_to_behavior == expt_record.label_to_behavior)
            assert idx == 0 or (behavior_to_label == expt_record.behavior_to_label)
            assert idx == 0 or (inactive_annotation == expt_record.inactive_annotation)
            assert expt_record.has_annotation
            label_to_behavior = expt_record.label_to_behavior
            behavior_to_label = expt_record.behavior_to_label
            inactive_annotation = expt_record.inactive_annotation

            mapping = self._load_yaml_dictionary(
                expt_path / "correspondences",
                f"mapping_{clustering_name}.yaml",
            )
            for cluster_lbl, behavior_weights in mapping.items():
                for behavior_lbl, weight in behavior_weights.items():
                    weight_n = weight / len(expt_names1)
                    total_mapping[cluster_lbl][behavior_lbl] = (
                        total_mapping[cluster_lbl].get(behavior_lbl, 0) + weight_n
                    )

        num_behavior = len(label_to_behavior)
        assert num_behavior > 1

        expt_names = expt_names1 + expt_names2
        clustering_names = clustering_names1 + clustering_names2

        for idx, expt_name in enumerate(expt_names):
            expt_path = self.expt_path_dict[expt_name]
            clustering_name = clustering_names[idx]

            cluster_membership = self._load_numpy_array(
                expt_path / "clusterings",
                f"membership_{clustering_name}.npy",
            )
            behavior_score = np.zeros((cluster_membership.shape[0], num_behavior))
            for cluster_lbl, behavior_weights in total_mapping.items():
                for behavior_lbl, weight in behavior_weights.items():
                    if behavior_lbl == behavior_to_label[inactive_annotation]:
                        weight += (
                            np.sqrt(1 / (num_behavior - 1))
                            if not cluster_lbl
                            else 1 / (num_behavior - 1)
                        )
                    cluster_score = (
                        cluster_membership[:, cluster_lbl]
                        * weight
                        * cluster_membership.shape[1]
                    )
                    behavior_score[:, behavior_lbl] = (
                        behavior_score[:, behavior_lbl] + cluster_score
                    )
            behavior_score = normalize(behavior_score, norm="l1")
            # behavior_score = scale(behavior_score, axis=0)

            self._save_numpy_array(
                behavior_score,
                expt_path / "correspondences",
                f"score_{clustering_name.replace('cluster', 'behavior')}.npy",
                depth=3,
            )

    @misc.timeit
    def disparately_compute_behavior_score_disparate_cluster_supervised_disparate(
        self,
    ):
        annotated_expt_names = list(self.annotation_path_dict.keys())

        for ann_expt_name in annotated_expt_names:
            ann_embedding_name = "supervised_disparate_embedding"
            ann_clustering_name = f"disparate_cluster_{ann_embedding_name}"
            self.disparately_compute_behavior_score(
                [ann_expt_name],
                [ann_clustering_name],
            )

    @misc.timeit
    def crosswisely_compute_behavior_score_crosswise_cluster_semisupervised_pair(
        self,
    ):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))

        for ann_expt_name, unann_expt_name in misc.list_cartesian_product(
            annotated_expt_names, unannotated_expt_names
        ):
            ann_embedding_name = (
                f"semisupervised_pair_embedding_{ann_expt_name}_{unann_expt_name}"
            )
            unann_embedding_name = (
                f"semisupervised_pair_embedding_{ann_expt_name}_{ann_expt_name}"
            )
            ann_clustering_name = (
                f"crosswise_cluster_{ann_expt_name}_{ann_embedding_name}"
            )
            unann_clustering_name = (
                f"crosswise_cluster_{ann_expt_name}_{unann_embedding_name}"
            )
            self.crosswisely_compute_behavior_score(
                [ann_expt_name],
                [unann_expt_name],
                [ann_clustering_name],
                [unann_clustering_name],
            )


class BehaviorMapping(BehaviorEmbedding, BehaviorClustering, BehaviorCorrespondence):
    def __init__(
        self,
        main_cfg_path,
        **kwargs,
    ):
        BehaviorEmbedding.__init__(self, main_cfg_path, **kwargs)
        BehaviorClustering.__init__(self, main_cfg_path, **kwargs)
        BehaviorCorrespondence.__init__(self, main_cfg_path, **kwargs)
