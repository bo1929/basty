import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters

from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier

import basty.utils.misc as misc

from basty.behavior_mapping.behavioral_windows import BehavioralWindows


class OutlineMixin:
    @staticmethod
    def get_cont_intvsl_dict(labels):
        intvls = misc.cont_intvls(labels)

        cont_intvls_dict = defaultdict(list)

        for i in range(1, intvls.shape[0]):
            lbl = labels[intvls[i - 1]]
            intvl_start, intvl_end = intvls[i - 1], intvls[i]
            cont_intvls_dict[lbl].append((intvl_start, intvl_end))

        return cont_intvls_dict

    @staticmethod
    def postprocess_outlines(labels, winsize, wintype="boxcar"):
        if winsize > 1:
            win_positive_counts = BehavioralWindows.get_behavior_counts(
                labels,
                wintype=wintype,
                winsize=winsize,
                stepsize=1,
            )
            labels_postprocessed = np.array(
                [np.argmax(counts) for counts in (win_positive_counts)]
            )
        else:
            labels_postprocessed = labels
        return labels_postprocessed

    @staticmethod
    def get_datums_values(df_values, datums=[], winsize=3):
        assert isinstance(df_values, pd.DataFrame) and df_values.ndim == 2

        X = np.zeros(df_values.shape[0])
        for d in datums:
            if d in df_values.columns:
                X = X + df_values[d].to_numpy()
            else:
                raise ValueError(f"Given datum name {d} is not defined.")

        if not datums:
            X = df_values.sum(axis=1, skipna=True).to_numpy()

        X[np.isnan(X)] = np.inf

        if winsize is not None:
            X = filters.median_filter(X, winsize)

        return X


class OutlineRandomForestClassifier(OutlineMixin):
    def __init__(self, **kwargs):
        self.clf = RandomForestClassifier(**kwargs)
        self.is_fitted = False

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X):
        return self.clf.predict(X)

    def construct_outline_decision_tree(self, X_train_list, y_train_list, **kwargs):
        assert isinstance(X_train_list, list) and isinstance(y_train_list, list)
        assert len(X_train_list) == len(y_train_list)
        assert all(
            isinstance(X_expt, np.ndarray) and X_expt.ndim == 2
            for X_expt in X_train_list
        )
        assert all(
            all(np.issubdtype(type(lbl), np.int) for lbl in y_train)
            for y_train in y_train_list
        )

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        self.__init__(**kwargs)
        self.clf.fit(X_train, y_train)
        return self.clf


class OutlineThresholdGMM(OutlineMixin):
    @staticmethod
    def threshold_detection(X, **kwargs):
        assert isinstance(X, np.ndarray) and X.ndim == 2
        assert X.shape[1] == 1
        gmm = GaussianMixture(**kwargs)
        clusters = gmm.fit_predict(X)

        xc = np.stack((X[:, 0], clusters))
        xc_s = xc[:, xc[0, :].argsort()]
        X_s, clusters_s = xc_s[0, :], xc_s[1, :]

        cluster_boundaries = X_s[misc.change_points(clusters_s) + 1]
        sorted_means = sorted(gmm.means_)

        # Choose appropriate one based on the distribution.
        return cluster_boundaries, list(map(lambda x: x[0], sorted_means))

    @classmethod
    def get_threshold(cls, X, threshold_key, num_gmm_comp, threshold_idx, **kwargs):
        assert threshold_idx < num_gmm_comp
        kwargs["n_components"] = num_gmm_comp
        diff_points, cluster_means = cls.threshold_detection(X, **kwargs)
        if threshold_key == "local_min":
            threshold = diff_points[threshold_idx]
        elif threshold_key == "local_max":
            threshold = cluster_means[threshold_idx]
        else:
            raise ValueError(
                f"Given threshold key {threshold_key} is not defined. "
                "Use (local_min) local minimum or (loca_max) local max."
            )
        return threshold


class SummaryCoefsCWT:
    @staticmethod
    def coef_logsum_scales(cwt_coefs):
        return np.log2(np.sum(cwt_coefs[:, :, :], axis=1) + 1)

    @staticmethod
    def coef_logmax_scales(cwt_coefs):
        return np.log2(np.amax(cwt_coefs[:, :, :], axis=1) + 1)

    @staticmethod
    def coef_sum_scales(cwt_coefs):
        return np.sum(cwt_coefs[:, :, :], axis=1)

    @staticmethod
    def coef_max_scales(cwt_coefs):
        return np.max(cwt_coefs[:, :, :], axis=1)

    @classmethod
    def get_df_coefs_summary(cls, cwt_coefs, colum_names, log=True, method="sum"):
        assert len(colum_names) == cwt_coefs.shape[-1]
        known_methods = ["sum", "max"]

        if log and method == "sum":
            df = pd.DataFrame(
                cls.coef_logsum_scales(cwt_coefs),
                columns=colum_names.keys(),
            )
        elif log and method == "max":
            df = pd.DataFrame(
                cls.coef_logmax_scales(cwt_coefs),
                columns=colum_names.keys(),
            )
        elif not log and method == "max":
            df = pd.DataFrame(
                cls.coef_max_scales(cwt_coefs),
                columns=colum_names.keys(),
            )
        elif not log and method == "sum":
            df = pd.DataFrame(
                cls.coef_sum_scales(cwt_coefs),
                columns=colum_names.keys(),
            )
        else:
            raise ValueError(f"Unkown 'method', must be one of {str(known_methods)}.")
        return df


class ActiveBouts(OutlineThresholdGMM, OutlineRandomForestClassifier, SummaryCoefsCWT):
    @classmethod
    def compute_active_bouts(cls, X, thresholds, winsize=30, wintype="boxcar"):
        assert isinstance(X, np.ndarray) and X.ndim == 2
        assert len(thresholds) == X.shape[1]

        mask_active = np.full(X.shape[0], False, dtype=np.bool_)
        active_mask_per_datums = []

        for i in range(X.shape[1]):
            values_per_datums = X[:, i]
            mask_active_tmp = values_per_datums > thresholds[i]
            active_mask_per_datums.append(mask_active_tmp)
            mask_active = np.logical_or(mask_active, mask_active_tmp)

        mask_active = cls.postprocess_outlines(
            mask_active.astype(int), winsize=winsize, wintype=wintype
        ).astype(bool)
        return mask_active, active_mask_per_datums

    def construct_active_bouts_decision_tree(
        self, X_train_list, y_train_list, **kwargs
    ):
        self.decision_tree = self.construct_outline_decision_tree(
            X_train_list, y_train_list, **kwargs
        )

    def predict_active_bouts(self, X, winsize=30, wintype="boxcar"):
        final_labels = np.array(self.decision_tree.predict(X))
        mask_active = final_labels == 1
        mask_active = self.postprocess_outlines(
            mask_active.astype(int), winsize=winsize, wintype=wintype
        ).astype(bool)
        return mask_active


class DormantEpochs(OutlineThresholdGMM, OutlineRandomForestClassifier):
    label_to_name = {0: "Dormant", 1: "Arouse", -1: "Betwixt"}

    @staticmethod
    def process_short_cont_intvls(labels, marker_label, min_intvl):
        intvls = misc.cont_intvls(labels.astype(int))
        processed_labels = np.empty_like(labels)

        for i in range(1, intvls.shape[0]):
            lbl = labels[intvls[i - 1]]
            intvl_start, intvl_end = intvls[i - 1], intvls[i]
            if intvl_end - intvl_start < min_intvl and lbl == marker_label:
                processed_labels[intvl_start:intvl_end] = -1
            else:
                processed_labels[intvl_start:intvl_end] = lbl
        return processed_labels

    @classmethod
    def compute_dormant_epochs(
        cls,
        X,
        threshold,
        min_dormant,
        tol_duration,
        tol_percent=0.4,
        winsize=30,
    ):
        assert isinstance(X, np.ndarray) and X.ndim == 2
        assert X.shape[1] == 1

        initial_labels = np.where(X[:, 0] < threshold, 0, 1).astype(int)
        cont_intvls_dict = cls.get_cont_intvsl_dict(initial_labels)
        label_win = misc.sliding_window(initial_labels, winsize=winsize, stepsize=1)
        intermediate_labels = np.empty_like(initial_labels)

        for i, (lbl, item) in enumerate(cont_intvls_dict.items()):
            if lbl == 0:
                for intvl_start, intvl_end in item:
                    intermediate_labels[intvl_start:intvl_end] = 0  # dormant
            else:
                for intvl_start, intvl_end in item:
                    dur = intvl_end - intvl_start
                    short_arouse = dur < tol_duration
                    epoch_mid = intvl_start + dur // 2
                    mostly_dormant = (
                        np.sum(label_win[epoch_mid]) / winsize
                    ) < tol_percent
                    if short_arouse and mostly_dormant:
                        intermediate_labels[intvl_start:intvl_end] = -1  # betwixt
                    else:
                        intermediate_labels[intvl_start:intvl_end] = 1  # arouse
        intermediate_labels[intermediate_labels == -1] = 0
        final_labels = cls.process_short_cont_intvls(
            intermediate_labels, 0, min_dormant
        )
        mask_dormant = final_labels == 0
        return mask_dormant, final_labels

    def construct_dormant_epochs_decision_tree(
        self, X_train_list, y_train_list, **kwargs
    ):
        self.decision_tree = self.construct_outline_decision_tree(
            X_train_list, y_train_list, **kwargs
        )

    def predict_dormant_epochs(self, X, min_dormant=900, winsize=90, wintype="boxcar"):
        initial_labels = np.array(self.decision_tree.predict(X))
        intermediate_labels = self.postprocess_outlines(
            initial_labels, winsize=winsize, wintype=wintype
        )
        final_labels = self.__class__.process_short_cont_intvls(
            intermediate_labels, 0, min_dormant
        )
        mask_dormant = final_labels == 0
        return mask_dormant
