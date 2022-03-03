import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters

from collections import defaultdict
from sklearn.mixture import GaussianMixture

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
    def postprocess_outlines(mask_postive, winsize, wintype="boxcar"):
        win_positive_counts = BehavioralWindows.get_behavior_counts(
            mask_postive.astype(int),
            wintype=wintype,
            winsize=winsize,
            stepsize=1,
        )
        mask_postive = np.array(
            [np.argmax(counts) for counts in (win_positive_counts)]
        ).astype(bool)
        return mask_postive

    @staticmethod
    def get_datums_values(df_values, datums=[], winsize=3):
        assert isinstance(df_values, pd.DataFrame) and df_values.ndim == 2

        values = np.zeros(df_values.shape[0])
        for d in datums:
            if d in df_values.columns:
                values = values + df_values[d].to_numpy()
            else:
                raise ValueError(f"Given datum name {d} is not defined.")

        if not datums:
            values = df_values.sum(axis=1, skipna=True).to_numpy()

        values[np.isnan(values)] = np.inf

        if winsize is not None:
            values = filters.median_filter(values, winsize)

        return values


class SupervisedOutline(OutlineMixin):
    pass


class UnsupervisedOutline(OutlineMixin):
    @staticmethod
    def threshold_detection(values, log=True, **kwargs):
        assert isinstance(values, np.ndarray) and values.ndim == 1
        if log:
            X = np.log2(values + 1)
        else:
            X = values

        gmm = GaussianMixture(**kwargs)
        clusters = gmm.fit_predict(X.reshape((-1, 1)))

        xc = np.stack((X, clusters))
        xc_s = xc[:, xc[0, :].argsort()]
        X_s, clusters_s = xc_s[0, :], xc_s[1, :]

        cluster_boundaries = X_s[misc.change_points(clusters_s) + 1]
        sorted_means = sorted(gmm.means_)

        # Choose appropriate one based on the distribution.
        return cluster_boundaries, list(map(lambda x: x[0], sorted_means))

    @classmethod
    def get_threshold(
        cls, values, threshold_log, threshold_key, num_gmm_comp, threshold_idx
    ):
        assert threshold_idx < num_gmm_comp

        diff_points, cluster_means = cls.threshold_detection(
            values, log=threshold_log, n_components=num_gmm_comp
        )
        if threshold_key == "local_min":
            threshold = diff_points[threshold_idx]
        elif threshold_key == "local_max":
            threshold = cluster_means[threshold_idx]
        else:
            raise ValueError(
                f"Given threshold key {threshold_key} is not defined. "
                "Use (local_min) local minimum or (loca_max) local maximum."
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
    def get_df_summary_coefs(cls, cwt_coefs, colum_names, log=True, method="sum"):
        assert len(colum_names) == cwt_coefs.shape[-1]
        known_methods = ["sum", "maximum"]

        if log and method == "sum":
            df = pd.DataFrame(
                cls.coef_logsum_scales(cwt_coefs),
                columns=colum_names.keys(),
            )
        elif log and method == "maximum":
            df = pd.DataFrame(
                cls.coef_logmax_scales(cwt_coefs),
                columns=colum_names.keys(),
            )
        elif not log and method == "maximum":
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


class ActiveBouts(UnsupervisedOutline, SupervisedOutline, SummaryCoefsCWT):
    @classmethod
    def compute_active_bouts(cls, values, thresholds, winsize=30, wintype="boxcar"):
        assert isinstance(values, np.ndarray) and values.ndim == 2
        assert len(thresholds) == values.shape[1]

        mask_active = np.full(values.shape[0], False, dtype=np.bool_)
        active_mask_per_datums = []

        for i in range(values.shape[1]):
            values_per_datums = values[:, i]
            mask_active_tmp = values_per_datums > thresholds[i]
            active_mask_per_datums.append(mask_active_tmp)
            mask_active = np.logical_or(mask_active, mask_active_tmp)

        mask_active = cls.postprocess_outlines(
            mask_active, winsize=winsize, wintype="boxcar"
        )

        return mask_active, active_mask_per_datums


class DormantEpochs(UnsupervisedOutline, SupervisedOutline):
    label_to_name = {0: "Dormant", 1: "Arouse", -1: "Betwixt"}

    @classmethod
    def compute_dormant_epochs(
        cls,
        values,
        threshold,
        min_dormant,
        tol_duration,
        tol_percent=0.4,
        winsize=30,
    ):
        assert isinstance(values, np.ndarray) and values.ndim == 1
        initial_labels = np.where(values < threshold, 0, 1).astype(int)

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

        intvls = misc.cont_intvls(intermediate_labels)
        final_labels = np.empty_like(intermediate_labels)

        for i in range(1, intvls.shape[0]):
            lbl = intermediate_labels[intvls[i - 1]]
            intvl_start, intvl_end = intvls[i - 1], intvls[i]
            if intvl_end - intvl_start < min_dormant and lbl == 0:
                final_labels[intvl_start:intvl_end] = -1
            else:
                final_labels[intvl_start:intvl_end] = lbl

        mask_dormant = final_labels != 1

        return mask_dormant, final_labels
