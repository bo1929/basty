import numpy as np
import sklearn.metrics as metrics

from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

import basty.utils.misc as misc


def label_coverage(y_ann, y_pred, normalize=False):
    assert y_ann.shape[0] == y_pred.shape[0]

    pred_coverage = {}
    for y_hat in np.unique(y_pred):
        uniq_y_ann, counts = np.unique(y_ann[y_pred == y_hat], return_counts=True)
        coverage = (
            counts / (y_pred[y_pred == y_hat].shape[0] + 1) if normalize else counts
        )
        pred_coverage[y_hat] = dict(zip(uniq_y_ann, coverage))

    ann_coverage = {}
    for y_hat in np.unique(y_ann):
        uniq_y_pred, counts = np.unique(y_pred[y_ann == y_hat], return_counts=True)
        coverage = (
            counts / (y_ann[y_ann == y_hat].shape[0] + 1) if normalize else counts
        )
        ann_coverage[y_hat] = dict(zip(uniq_y_pred, coverage))

    return misc.sort_dict(ann_coverage), misc.sort_dict(pred_coverage)


def total_occupation(y):
    uniq_y, counts = np.unique(y, return_counts=True)
    return dict(zip(uniq_y.astype(int), counts))


def absolute_freq(y):
    occupation = total_occupation(y)
    return {key: val / y.shape[0] for key, val in occupation.items()}


def evolution_rate(y, normalize=False):
    uniq_y = np.unique(y)
    if normalize:
        denom = total_occupation(y)
    else:
        denom = [1 for _ in uniq_y]
    rate = {
        int(y_hat): np.cumsum(np.where(y == y_hat, 1, 0)) / denom[y_hat]
        for y_hat in uniq_y
    }
    return rate


class BoutDetails:
    def __init__(self, fps, y):
        self.fps = fps
        self.y = y
        self.intervals = misc.cont_intvls(y)

    def get_bout_feature_details(self, values, offset=1):
        intervals = self.intervals
        y = self.y
        bout_details_dict = defaultdict(list)

        for i in range(1, intervals.shape[0]):
            start = max(intervals[i - 1] - offset * self.fps, 0)
            end = intervals[i] + offset * self.fps
            bout_details_dict[y[intervals[i - 1]]].append(
                ((start, end), values[start:end, :])
            )
        bout_details_dict = misc.sort_dict(bout_details_dict)

        return bout_details_dict

    def get_bout_temporal_details(self):
        intervals = self.intervals
        y = self.y
        bout_details_dict = defaultdict(list)

        for i in range(1, intervals.shape[0]):
            start, end = intervals[i - 1], intervals[i]
            details = {"duration": end - start, "start": start, "end": end}
            details["label"] = y[intervals[i - 1]]
            bout_details_dict[y[intervals[i - 1]]].append(details)
        bout_details_dict = misc.sort_dict(bout_details_dict)

        return bout_details_dict


class kNN:
    def __init__(self, k):
        self.k = k
        self.kNeigh = NearestNeighbors(n_neighbors=self.k)

    def fit(self, X_embedded):
        self.kNeigh.fit(X_embedded)
        self.neighbours = self.kNeigh.kneighbors(X_embedded)

    def knn_accuracy(self, y, ignore_consecutive=0):
        assert ignore_consecutive >= 0
        neig_idx = self.neighbours[1]

        neig_avgdist = [
            sum([(i - j) for j in neig_idx[i]]) / self.k for i in range(len(neig_idx))
        ]
        majority_pred = [
            misc.most_common(
                [y[j] for j in neig_idx[i] if abs(j - i) > ignore_consecutive]
            )
            for i in range(len(neig_idx))
        ]

        accuracy = metrics.accuracy_score(y, majority_pred)
        print(f"Majority {self.k}-NN prediction accuracy is {accuracy}")

        return accuracy, neig_avgdist


class MetricMixin:
    def __init__(self, y_ann, y_pred, ignore_ann=None, ignore_pred=None):
        assert isinstance(y_ann, np.ndarray)
        assert isinstance(y_pred, np.ndarray)

        self.ignore_ann = [] if ignore_ann is None else ignore_ann
        self.ignore_pred = [] if ignore_pred is None else ignore_pred

        y_pred, y_ann = self.__class__._delete_y(y_pred, y_ann, ignore_pred)
        self.y_ann, self.y_pred = self.__class__._delete_y(y_ann, y_pred, ignore_ann)

    @staticmethod
    def _delete_y(y0, y1, ignore0):
        for y_hat in ignore0:
            y1 = y1[y0 != y_hat]
            y0 = y0[y0 != y_hat]
        return y0, y1


class ClassificationMetrics(MetricMixin):
    def __init__(self, y_ann, y_pred, ignore_ann=None, ignore_pred=None):
        super().__init__(y_ann, y_pred, ignore_ann=ignore_ann, ignore_pred=ignore_pred)
        self.ann_coverage, self.pred_coverage = label_coverage(
            self.y_ann, self.y_pred, normalize=True
        )
        pred_majority_dict = {
            y_hat: max(counts, key=(lambda key: counts[key]))
            for y_hat, counts in self.pred_coverage.items()
        }
        self.y_majority = np.array([pred_majority_dict[y_i] for y_i in self.y_pred])

        from sklearn import preprocessing

        lb = preprocessing.LabelBinarizer()
        self.y_ann_binary = lb.fit_transform(self.y_ann)

        self.y_pred_coverage_scores = np.zeros(self.y_ann_binary.shape)
        for i in range(self.y_pred_coverage_scores.shape[0]):
            for y_hat, val in enumerate(self.pred_coverage[self.y_pred[i]].items()):
                self.y_pred_coverage_scores[i, y_hat] = val

    def f1_score(self, **kwargs):
        return metrics.f1_score(self.y_ann, self.y_majority, **kwargs)

    def accuracy_score(self, **kwargs):
        return metrics.accuracy_score(self.y_ann, self.y_majority, **kwargs)

    def balanced_accuracy_score(self, **kwargs):
        return metrics.balanced_accuracy_score(self.y_ann, self.y_majority, **kwargs)

    def brier_score_loss(self, **kwargs):
        return metrics.brier_score_loss(self.y_ann, self.y_majority, **kwargs)

    def jaccard_score(self, **kwargs):
        return metrics.jaccard_score(self.y_ann, self.y_majority, **kwargs)

    def recall_score(self, **kwargs):
        return metrics.recall_score(self.y_ann, self.y_majority, **kwargs)

    def precision_score(self, **kwargs):
        return metrics.precision_score(self.y_ann, self.y_majority, **kwargs)

    def coverage_error(self, y_scores=None, **kwargs):
        y_scores = self.y_pred_coverage_scores if y_scores is None else y_scores
        return metrics.coverage_error(self.y_ann_binary, y_scores, **kwargs)

    def average_precision_score(self, y_scores=None, **kwargs):
        y_scores = self.y_pred_coverage_scores if y_scores is None else y_scores
        return metrics.average_precision_score(self.y_ann_binary, y_scores, **kwargs)

    def label_ranking_avg_precision_score(self, y_scores=None, **kwargs):
        y_scores = self.y_pred_coverage_scores if y_scores is None else y_scores
        return metrics.label_ranking_average_precision_score(
            self.y_ann_binary, y_scores, **kwargs
        )

    def label_ranking_loss(self, y_scores=None, **kwargs):
        y_scores = self.y_pred_coverage_scores if y_scores is None else y_scores
        return metrics.label_ranking_loss(self.y_ann_binary, y_scores, **kwargs)


class ClusteringMetrics(MetricMixin):
    def __init__(self, y_ann, y_pred, ignore_ann=None, ignore_pred=None):
        super().__init__(y_ann, y_pred, ignore_ann=ignore_ann, ignore_pred=ignore_pred)

    def adjusted_rand_score(self):
        (tn, fp), (fn, tp) = metrics.cluster.pair_confusion_matrix(
            self.y_ann, self.y_pred
        )
        # Special cases, namely empty data or full agreement.
        if fn == 0 and fp == 0:
            return 1.0
        return (
            2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        )

    def adjusted_mutual_info_score(self, **kwargs):
        return metrics.cluster.adjusted_mutual_info_score(
            self.y_ann, self.y_pred, **kwargs
        )

    def normalized_mutual_info_score(self, **kwargs):
        return metrics.cluster.normalized_mutual_info_score(
            self.y_ann, self.y_pred, **kwargs
        )

    def v_measure_score(self, **kwargs):
        return metrics.cluster.v_measure_score(self.y_ann, self.y_pred, **kwargs)

    def completeness_score(self):
        return metrics.cluster.completeness_score(self.y_ann, self.y_pred)

    def homogeneity_score(self):
        return metrics.cluster.homogeneity_score(self.y_ann, self.y_pred)


def compute_hausdorff_distance(X, y):
    # '0' stands for 'inactive' or 'unannotated' frames.
    assert 0 in y
    uniq_y = np.unique(y)
    # How far two subsets of a metric space are from each other.
    hausdorff_dist = np.zeros((uniq_y.shape[0], uniq_y.shape[0]))
    # hausdorff_dist[np.diag_indices(hausdorff_dist.shape[0])] = 0

    from scipy.spatial.distance import directed_hausdorff

    for i in range(1, hausdorff_dist.shape[0]):
        for j in range(i + 1, hausdorff_dist.shape[1]):
            dir1 = directed_hausdorff(X[y == uniq_y[i]], X[y == uniq_y[j]])
            dir2 = directed_hausdorff(X[y == uniq_y[i]], X[y == uniq_y[j]])
            hausdorff_dist[i, j] = (dir1[0] + dir2[0]) / 2
    hausdorff_dist = hausdorff_dist + np.triu(hausdorff_dist).T
    return hausdorff_dist


def compute_distn_jensenshannon(X, y):
    # '0' stands for 'inactive' or 'unannotated' frames.
    assert 0 in y
    uniq_y = np.unique(y)
    # Jensen-Shannon distance (metric) between two probability arrays of same size.
    js_dist = np.zeros((uniq_y.shape[0], uniq_y.shape[0]))

    from scipy.spatial.distance import jensenshannon

    for i in range(1, js_dist.shape[0]):
        for j in range(i, js_dist.shape[1]):
            js_dist[i, j] = jensenshannon(X[y == uniq_y[i]], X[y == uniq_y[j]])
    js_dist = js_dist + np.triu(js_dist).T
    return js_dist


def compute_silhouette_score(X, y, metric="euclidean", sample_size=None, **kwargs):
    rng = np.random.default_rng()

    if sample_size is not None:
        indices = rng.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            X, y = X[indices].T[indices].T, y[indices]
        else:
            X, y = X[indices], y[indices]

    return np.mean(metrics.silhouette_samples(X, y, metric=metric, **kwargs))


def compute_lagged_bivariate_prob(y_binary, min_lag, max_lag):
    num_uniq_y = y_binary.shape[1]
    lagged_bivar_prob = np.zeros((max_lag - min_lag + 1, num_uniq_y, num_uniq_y))
    for lag in range(min_lag, max_lag + 1):
        for y_i in range(num_uniq_y):
            for y_j in range(num_uniq_y):
                lagged_bivar_prob[lag - min_lag, y_i, y_j] = np.mean(
                    y_binary[:-lag, y_i] * y_binary[lag:, y_j]
                )
    return lagged_bivar_prob


def cramers_serial_dependency(y_binary, min_lag, max_lag):
    assert y_binary.shape > 1
    hat_pi = np.mean(y_binary, axis=0)
    hat_ij_pi = hat_pi[:, np.newaxis] @ hat_pi[np.newaxis, :]
    cramers = np.zeros(max_lag + 1)
    lagged_bivar_prob = compute_lagged_bivariate_prob(y_binary, min_lag, max_lag)
    for lag in range(min_lag, max_lag + 1):
        cramers[lag] = np.sqrt(
            np.sum((lagged_bivar_prob[lag - min_lag] - hat_ij_pi) ** 2 / hat_ij_pi)
            / (y_binary.shape[1] - 1)
        )
    return cramers


def cohens_serial_dependency(y_binary, min_lag, max_lag):
    hat_pi = np.mean(y_binary, axis=0)
    denom = 1 - np.sum(hat_pi ** 2)
    cohens = np.zeros(max_lag + 1)
    lagged_bivar_prob = compute_lagged_bivariate_prob(y_binary, min_lag, max_lag)
    for lag in range(min_lag, max_lag + 1):
        cohens[lag] = (
            np.sum(np.diag(lagged_bivar_prob[lag - min_lag]) - hat_pi ** 2) / denom
        )
    return cohens
