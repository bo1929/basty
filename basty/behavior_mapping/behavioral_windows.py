import logging

import numpy as np
from scipy.signal.windows import get_window
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

import basty.utils.misc as misc


class BehavioralWindows:
    def __init__(self, fps, win_cfg={}, use_stride_tricks=False):
        self.win_cfg = win_cfg
        self.logger = logging.getLogger("main")

        self.wintype = self.win_cfg.get("wintype", "boxcar")
        self.winsize = int(self.win_cfg.get("winsize", 10) * fps)
        self.stepsize = int(self.win_cfg.get("stepsize", 5) * fps)
        self.method = self.win_cfg.get("method", "tf-idf")
        self.norm = self.win_cfg.get("norm", None)
        self.use_stride_tricks = use_stride_tricks

        assert self.winsize > 1

        if self.stepsize > self.winsize:
            self.logger.warning("Step-size should be smaller than window-size.")

        if self.method == "tf-idf" and self.wintype != "boxcar":
            raise ValueError(
                "The tf-idf statistics can only be used with boxcar window type."
            )
        if self.norm is None and self.method == "tf-idf":
            self.logger.warning(
                "A normalization method should be specifed for tf-idf representations."
                + "Default is 'L1'."
            )
            self.norm = self.win_cfg.get("norm", "l1")

        if self.norm is not None and self.method == "tf-idf":
            self.logger.warning(
                f"Normalization method is ignored for {self.method} representations."
            )
            self.norm = None

    @staticmethod
    def align_behavioral_windows(win_repr, uniq_labels, num_total_uniq):
        assert win_repr.dim > 1
        win_repr_aligned = np.zeros((win_repr.shape[0], num_total_uniq))
        for idx, lbl in sorted(uniq_labels):
            win_repr_aligned[:, lbl] = win_repr[:, idx]
        return win_repr_aligned

    def get_window_representation(self, bhv_seq):
        if self.method == "tf-idf":
            win_repr = self.get_tfidf_vectors(
                bhv_seq,
                winsize=self.winsize,
                stepsize=self.stepsize,
                norm=self.norm,
                use_stride_tricks=self.use_stride_tricks,
            )
        elif self.method == "binary":
            win_repr = self.get_binary_vectors(
                bhv_seq,
                winsize=self.winsize,
                stepsize=self.stepsize,
                use_stride_tricks=self.use_stride_tricks,
            )
        elif self.method == "count" and self.norm is None:
            win_repr = self.get_behavior_counts(
                bhv_seq,
                wintype=self.wintype,
                winsize=self.winsize,
                stepsize=self.stepsize,
                use_stride_tricks=self.use_stride_tricks,
            )
        elif self.method == "count" and self.norm in ["l1", "l2"]:
            win_repr = self.get_count_vectors(
                bhv_seq,
                wintype=self.wintype,
                winsize=self.winsize,
                stepsize=self.stepsize,
                norm=self.norm,
                use_stride_tricks=self.use_stride_tricks,
            )
        elif self.method == "majority":
            win_repr = self.get_majority_labels(
                bhv_seq,
                wintype=self.wintype,
                winsize=self.winsize,
                stepsize=self.stepsize,
                use_stride_tricks=self.use_stride_tricks,
            )
        else:
            raise ValueError(f"Incompatible pair ({self.method},{self.norm}) is given.")
        return win_repr

    @staticmethod
    def _get_bhv_windows(bhv_seq, winsize, stepsize, use_stride_tricks=False):
        assert winsize > 1
        if use_stride_tricks and stepsize == 1:
            from numpy.lib.stride_tricks import sliding_window_view

            bhv_win = sliding_window_view(bhv_seq, winsize)
        else:
            bhv_win = [
                w
                for w in misc.sliding_window(
                    bhv_seq, winsize=winsize, stepsize=stepsize
                )
            ]
        return bhv_win

    @classmethod
    def get_behavior_counts(
        cls, bhv_seq, wintype="boxcar", winsize=60, stepsize=1, use_stride_tricks=False
    ):
        num_bhv = np.max(bhv_seq) + 1

        bhv_win = cls._get_bhv_windows(
            bhv_seq, winsize, stepsize, use_stride_tricks=use_stride_tricks
        )
        window_arr = get_window(wintype, winsize)

        bhv_count = np.zeros((len(bhv_win), num_bhv), dtype=np.float)

        for i, bhv in enumerate(bhv_win):
            for j, bhv_j in enumerate(bhv):
                bhv_count[i, bhv_j] += window_arr[j]
        return bhv_count

    @classmethod
    def get_tfidf_vectors(
        cls, bhv_seq, winsize=60, stepsize=1, norm="l1", use_stride_tricks=False
    ):
        assert norm in ["l1", "l2"]
        bhv_count = cls.get_behavior_counts(
            bhv_seq,
            winsize=winsize,
            stepsize=stepsize,
            use_stride_tricks=use_stride_tricks,
        )
        vectorizer = TfidfTransformer(norm=norm)
        bhv_tfidf = vectorizer.fit_transform(bhv_count).toarray()
        return bhv_tfidf

    @classmethod
    def get_binary_vectors(
        cls, bhv_seq, winsize=60, stepsize=1, use_stride_tricks=False
    ):
        bhv_win = cls._get_bhv_windows(
            bhv_seq, winsize, stepsize, use_stride_tricks=use_stride_tricks
        )
        mlb_vectorizer = MultiLabelBinarizer()
        bhv_binary = mlb_vectorizer.fit_transform(bhv_win)
        return bhv_binary

    @classmethod
    def get_count_vectors(
        cls,
        bhv_seq,
        wintype="boxcar",
        winsize=60,
        stepsize=1,
        norm="l1",
        use_stride_tricks=False,
    ):
        assert norm in ["l1", "l2", "max"]
        bhv_count = cls.get_behavior_counts(
            bhv_seq,
            wintype=wintype,
            winsize=winsize,
            stepsize=stepsize,
            use_stride_tricks=use_stride_tricks,
        )
        bhv_count_n = normalize(bhv_count, norm=norm)
        return bhv_count_n

    @classmethod
    def get_majority_labels(
        cls,
        bhv_seq,
        wintype="boxcar",
        winsize=60,
        stepsize=1,
        ignore_idx=[],
        use_stride_tricks=False,
    ):
        bhv_win = cls._get_bhv_windows(
            bhv_seq, winsize, stepsize, use_stride_tricks=use_stride_tricks
        )

        if ignore_idx:
            bhv_majority = np.array(
                [misc.most_common(bhv_win[i]) for i in range(len(bhv_win))]
            )
        else:
            bhv_count = cls.get_behavior_counts(
                bhv_seq,
                wintype=wintype,
                winsize=winsize,
                stepsize=stepsize,
                use_stride_tricks=use_stride_tricks,
            )
            bhv_count[:, ignore_idx] = -np.inf
            bhv_majority = [np.argmax(bhv_count[i, :]) for i in range(len(bhv_count))]
        return bhv_majority
