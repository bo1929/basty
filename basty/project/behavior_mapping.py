import umap
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from hdbscan import HDBSCAN, membership_vector

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
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        assert all_expt_names
        assert annotated_expt_names
        assert unannotated_expt_names

        for ann_expt_name, unann_expt_name in misc.list_cartesian_product(
            annotated_expt_names, unannotated_expt_names
        ):
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

    def jointly_cluster_embeddings(self, expt_names, embedding_names):
        embedding_expt_dict = defaultdict()
        expt_indices_dict = defaultdict(tuple)

        prev = 0
        pbar = tqdm(expt_names)
        for i, expt_name in enumerate(pbar):
            embedding_name = embedding_names[i]
            embedding_name_msg = " ".join(embedding_name.split("_"))
            self.logger.direct_info(
                f"Loading {embedding_name_msg} for joint clustering of all experiments."
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
                expt_path / "cluster_labels",
                f"joint_cluster_labels_{embedding_name}.npy",
                depth=3,
            )

    def jointly_cluster_supervised_joint_embeddings(self):
        ann_expt_names = list(self.annotation_path_dict.keys())
        embedding_names = ["supervised_joint_embedding" for _ in ann_expt_names]
        self.jointly_cluster_embeddings(ann_expt_names, embedding_names)

    def jointly_cluster_unsupervised_joint_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        embedding_names = ["unsupervised_joint_embedding" for _ in all_expt_names]
        self.jointly_cluster_embeddings(all_expt_names, embedding_names)

    def jointly_cluster_semisupervised_pair_embeddings(self):
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
            self.jointly_cluster_embeddings(
                [unann_expt_name, ann_expt_name], embedding_names
            )

    def disparately_cluster_embeddings(self, expt_names, embedding_names):
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
                expt_path / "cluster_labels",
                f"disparate_cluster_labels_{embedding_name}.npy",
                depth=3,
            )

    def disparately_cluster_supervised_joint_embeddings(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        embedding_name = ["supervised_joint_embedding" for _ in annotated_expt_names]
        self.disparately_cluster_embeddings(annotated_expt_names, embedding_name)

    def disparately_cluster_unsupervised_joint_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        embedding_name = ["unsupervised_joint_embedding" for _ in all_expt_names]
        self.disparately_cluster_embeddings(all_expt_names, embedding_name)

    def disparately_cluster_supervised_embeddings(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        embedding_name = ["supervised_embedding" for _ in annotated_expt_names]
        self.disparately_cluster_embeddings(annotated_expt_names, embedding_name)

    def disparately_cluster_unsupervised_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        embedding_name = ["unsupervised_embedding" for _ in all_expt_names]
        self.disparately_cluster_embeddings(all_expt_names, embedding_name)

    def disparately_cluster_semisupervised_pair_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        for ann_expt_name, unann_expt_name in misc.list_cartesian_product(
            annotated_expt_names, unannotated_expt_names
        ):
            embedding_names = [f"semisupervised_pair_embedding_{unann_expt_name}"]
            self.disparately_cluster_embeddings([ann_expt_name], embedding_names)
            # Not really needed.
            # embedding_names = [f"semisupervised_pair_embedding_{ann_expt_name}"]
            # self.disparately_cluster_embeddings([unann_expt_name], embedding_names)

    def compute_cross_pair_cluster_membership(
        self, expt_name1, expt_name2, embedding_name1, embedding_name2
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
            raise ValueError

        expt_path1 = self.expt_path_dict[expt_name1]
        expt_path2 = self.expt_path_dict[expt_name2]

        embedding_expt1 = self._load_numpy_array(
            expt_path1 / "embeddings", f"{embedding_name1}.npy"
        )
        embedding_expt2 = self._load_numpy_array(
            expt_path2 / "embeddings", f"{embedding_name2}.npy"
        )

        clusterer = HDBSCAN(**self.HDBSCAN_kwargs)
        clusterer.fit(embedding_expt1)
        cluster_membership = membership_vector(clusterer, embedding_expt2)

        self._save_numpy_array(
            cluster_membership,
            expt_path2 / "cluster_labels",
            f"cross_pair_cluster_membership_{embedding_name2}_{embedding_name1}.npy",
            depth=3,
        )

    def cross_pair_cluster_membership_semisupervised_pair_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))

        pbar = tqdm(
            misc.list_cartesian_product(annotated_expt_names, unannotated_expt_names)
        )
        for ann_expt_name, unann_expt_name in pbar:
            ann_embedding_name = f"semisupervised_pair_embedding_{unann_expt_name}"
            unann_embedding_name = f"semisupervised_pair_embedding_{ann_expt_name}"
            self.compute_cross_pair_cluster_membership(
                ann_expt_name, unann_expt_name, ann_embedding_name, unann_embedding_name
            )
            # Not really needed.
            # self.compute_cross_pair_cluster_membership(
            #     unann_expt_name, ann_expt_name, unann_embedding_name, ann_embedding_name
            # )

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
            y_ann = y_ann[expt_record.mask_annotated]

        y_cluster = self._load_numpy_array(
            expt_path / "cluster_labels", f"{clustering_name}.npy"
        )

        mapping_dictionary = defaultdict(dict)
        y_cluster_uniq = np.unique(y_cluster)
        for cluster_lbl in y_cluster_uniq:
            y_ann_masked = y_ann[y_cluster == cluster_lbl]
            y_ann_uniq_c, ann_uniq_c_counts = np.unique(
                y_ann_masked, return_counts=True
            )
            for idx, ann_lbl in enumerate(y_ann_uniq_c):
                # mapping_dictionary[int(cluster_lbl)][int(ann_lbl)] = float(
                #     ann_uniq_c_counts[idx] / y_ann_masked.shape[0]
                # )
                ann_behavior = expt_record.label_to_behavior[ann_lbl]
                mapping_dictionary[int(cluster_lbl)][ann_behavior] = float(
                    ann_uniq_c_counts[idx] / y_ann_masked.shape[0]
                )

        self._save_yaml_dictionary(
            dict(mapping_dictionary),
            expt_path / "behavior_labels",
            f"mapping_{clustering_name}.yaml",
        )

    def map_disparate_cluster_labels_semisupervised_pair_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        for ann_expt_name, unann_expt_name in misc.list_cartesian_product(
            annotated_expt_names, unannotated_expt_names
        ):
            embedding_name = f"semisupervised_pair_embedding_{unann_expt_name}"
            clustering_name = f"disparate_cluster_labels_{embedding_name}"
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    def map_disparate_cluster_labels_supervised_embeddings(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        for ann_expt_name in annotated_expt_names:
            embedding_name = "supervised_embedding"
            clustering_name = f"disparate_cluster_labels_{embedding_name}"
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    def map_disparate_cluster_labels_supervised_joint_embeddings(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        for ann_expt_name in annotated_expt_names:
            embedding_name = "supervised_joint_embedding"
            clustering_name = f"disparate_cluster_labels_{embedding_name}"
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    def map_disparate_cluster_labels_unsupervised_embeddings(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        for ann_expt_name in annotated_expt_names:
            embedding_name = "unsupervised_embedding"
            clustering_name = f"disparate_cluster_labels_{embedding_name}"
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    def map_disparate_cluster_labels_unsupervised_joint_embeddings(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        for ann_expt_name in annotated_expt_names:
            embedding_name = "unsupervised_joint_embedding"
            clustering_name = f"disparate_cluster_labels_{embedding_name}"
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    def map_joint_cluster_labels_semisupervised_pair_embeddings(self):
        all_expt_names = list(self.expt_path_dict.keys())
        annotated_expt_names = list(self.annotation_path_dict.keys())
        unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
        for ann_expt_name, unann_expt_name in misc.list_cartesian_product(
            annotated_expt_names, unannotated_expt_names
        ):
            embedding_name = f"semisupervised_pair_embedding_{unann_expt_name}"
            clustering_name = f"joint_cluster_labels_{embedding_name}"
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    def map_joint_cluster_labels_supervised_joint_embeddings(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        for ann_expt_name in annotated_expt_names:
            embedding_name = "supervised_joint_embedding"
            clustering_name = f"joint_cluster_labels_{embedding_name}"
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)

    def map_joint_cluster_labels_unsupervised_joint_embeddings(self):
        annotated_expt_names = list(self.annotation_path_dict.keys())
        for ann_expt_name in annotated_expt_names:
            embedding_name = "unsupervised_joint_embedding"
            clustering_name = f"joint_cluster_labels_{embedding_name}"
            self.map_cluster_labels_to_behavior_labels(ann_expt_name, clustering_name)
