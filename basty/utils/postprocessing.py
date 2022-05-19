from collections import defaultdict

import numpy as np
import scipy.ndimage as ndimage

import basty.utils.misc as misc
from basty.behavior_mapping.behavioral_windows import BehavioralWindows


class PostProcessing:
    @staticmethod
    def process_short_cont_intvls(labels, marker_labels, min_intvl):
        if min_intvl > 1:
            intvls = misc.cont_intvls(labels.astype(int))
            processed_labels = np.empty_like(labels)

            for i in range(1, intvls.shape[0]):
                lbl = labels[intvls[i - 1]]
                intvl_start, intvl_end = intvls[i - 1], intvls[i]
                if intvl_end - intvl_start < min_intvl and lbl in marker_labels:
                    processed_labels[intvl_start:intvl_end] = -1
                else:
                    processed_labels[intvl_start:intvl_end] = lbl
        else:
            processed_labels = labels
        return processed_labels

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
    def compute_window_majority(labels, winsize, wintype="boxcar"):
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
    def median_filter(X, winsize, **kwargs):
        if X.ndim == 2:
            for i in range(X.shape[1]):
                X[:, i] = ndimage.median_filter(X[:, i], size=winsize, **kwargs)
        elif X.ndim == 1:
            X = ndimage.median_filter(X, size=winsize, **kwargs)
        else:
            raise ValueError("Number of dimensions have to be one or two.")
        return X

    @staticmethod
    def running_mean(X, winsize, **kwargs):
        if X.ndim == 2:
            for i in range(X.shape[1]):
                X[:, i] = ndimage.uniform_filter1d(X[:, i], size=winsize, **kwargs)
        elif X.ndim == 1:
            X = ndimage.uniform_filter1d(X, size=winsize, **kwargs)
        else:
            raise ValueError("Number of dimensions have to be one or two.")
        return X
