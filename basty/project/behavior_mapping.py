import umap
import hdbscan
import numpy as np

from tqdm import tqdm
from collections import defaultdict

import basty.utils.misc as misc

from basty.project.experiment_processing import Project


class BehaviorMapping(Project):
    def __init__(
        self,
        main_cfg_path,
        **kwargs,
    ):
        Project.__init__(self, main_cfg_path, **kwargs)
        self.init_behavior_embeddings_kwargs(**kwargs)
        self.init_behavior_clustering_kwargs(**kwargs)
        self.init_mapping_postprocessing_kwargs(**kwargs)

    @misc.timeit
    def compute_behavior_space(self, unannotated_expt_names, annotated_expt_names):
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
                f"Loading behavioral reprs. and annotations (if exists) of {expt_name}"
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
            y_expt = self._load_numpy_array(expt_path, "annotations.npy")

            X_expt_dict[expt_name] = X_expt[mask_annotated]
            y_expt_dict[expt_name] = y_expt[mask_annotated]

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
        unannotated_expt_names = [
            expt_name
            for expt_name in all_expt_names
            if expt_name not in annotated_expt_names
        ]
        assert all_expt_names
        assert annotated_expt_names
        assert unannotated_expt_names

        for unann_expt_name in unannotated_expt_names:
            for ann_expt_name in annotated_expt_names:
                embedding, expt_indices_dict = self.compute_behavior_space(
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
        for expt_name in all_expt_names:
            embedding, expt_indices_dict = self.compute_behavior_space([expt_name], [])
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
        for ann_expt_name in annotated_expt_names:
            embedding, expt_indices_dict = self.compute_behavior_space(
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
        embedding, expt_indices_dict = self.compute_behavior_space(all_expt_names, [])
        for expt_name in all_expt_names:
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
        embedding, expt_indices_dict = self.compute_behavior_space(
            [], annotated_expt_names
        )
        for ann_expt_name in annotated_expt_names:
            expt_path = self.expt_path_dict[ann_expt_name]
            start, end = expt_indices_dict[ann_expt_name]
            embedding_expt = embedding[start:end]
            self._save_numpy_array(
                embedding_expt,
                expt_path / "embeddings",
                "supervised_joint_embedding.npy",
                depth=3,
            )

    def jointly_cluster_embeddings(self, expt_names, embedding_name):
        embedding_expt_dict = defaultdict()
        expt_indices_dict = defaultdict(tuple)
        embedding_name_msg = " ".join(embedding_name.split("_"))

        prev = 0
        pbar = tqdm(expt_names)
        for expt_name in pbar:
            expt_path = self.expt_path_dict[expt_name]
            embedding_expt = self._load_numpy_array(
                expt_path / "embeddings", f"{embedding_name}.npy"
            )

            embedding_expt_dict[expt_name] = embedding_expt
            expt_indices_dict[expt_name] = prev, prev + embedding_expt.shape[0]
            prev = expt_indices_dict[expt_name][-1]

        self.logger.direct_info(
            f"Joint clustering {embedding_name_msg} of all experiments."
        )
        embedding = np.concatenate(list(embedding_expt_dict.values()), axis=0)
        clusterer = hdbscan.HDBSCAN(**self.HDBSCAN_kwargs)
        cluster_labels = (clusterer.fit_predict(embedding) + 1).astype(int)

        pbar = tqdm(expt_names)
        for expt_name in pbar:
            expt_path = self.expt_path_dict[expt_name]
            start, end = expt_indices_dict[expt_name]
            expt_indices_dict[expt_name] = prev, prev + embedding_expt.shape[0]
            cluster_labels_expt = cluster_labels[start:end]
            self._save_numpy_array(
                cluster_labels_expt,
                expt_path / "cluster_labels",
                f"joint_cluster_labels_{embedding_name}.npy",
                depth=3,
            )

    def jointly_cluster_supervised_joint_embeddings(self):
        ann_expt_names = list(self.annotation_path_dict.keys())
        embedding_name = "supervised_joint_embedding"
        self.jointly_cluster_embeddings(ann_expt_names, embedding_name)

    def jointly_cluster_unsupervised_joint_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        embedding_name = "unsupervised_joint_embedding"
        self.jointly_cluster_embeddings(all_expt_names, embedding_name)

    def disparately_cluster_embeddings(self, expt_names, embedding_name):
        embedding_name_msg = " ".join(embedding_name.split("_"))
        pbar = tqdm(expt_names)
        for expt_name in pbar:
            pbar.set_description(
                f"Individual clustering {embedding_name_msg} of {expt_name}"
            )
            expt_path = self.expt_path_dict[expt_name]
            embedding_expt = self._load_numpy_array(
                expt_path / "embeddings", f"{embedding_name}.npy"
            )
            clusterer = hdbscan.HDBSCAN(**self.HDBSCAN_kwargs)
            cluster_labels = (clusterer.fit_predict(embedding_expt) + 1).astype(int)
            self._save_numpy_array(
                cluster_labels,
                expt_path / "cluster_labels",
                f"disparate_cluster_labels_{embedding_name}.npy",
                depth=3,
            )

    def disparately_cluster_supervised_joint_embeddings(self):
        embedding_name = "supervised_joint_embedding"
        annotated_expt_names = list(self.annotation_path_dict.keys())
        self.disparately_cluster_embeddings(annotated_expt_names, embedding_name)

    def disparately_cluster_unsupervised_joint_embeddings(self):
        embedding_name = "unsupervised_joint_embedding"
        all_expt_names = list(self.expt_path_dict.keys())
        self.disparately_cluster_embeddings(all_expt_names, embedding_name)

    def disparately_cluster_supervised_embeddings(self):
        embedding_name = "supervised_embedding"
        annotated_expt_names = list(self.annotation_path_dict.keys())
        self.disparately_cluster_embeddings(annotated_expt_names, embedding_name)

    def disparately_cluster_unsupervised_embeddings(self):
        embedding_name = "unsupervised_embedding"
        all_expt_names = list(self.expt_path_dict.keys())
        self.disparately_cluster_embeddings(all_expt_names, embedding_name)
