import math
import itertools
import logging
import numpy as np
import pandas as pd

from scipy.ndimage.filters import uniform_filter1d
from copy import deepcopy

import basty.utils.misc as misc


np.seterr(all="ignore")


class SpatioTemporal:
    def __init__(self, fps, stft_cfg={}):
        self.stft_cfg = deepcopy(stft_cfg)
        self.logger = logging.getLogger("main")

        assert fps > 0
        self.get_delta = lambda x, scale: self.calc_delta(x, scale, fps)
        self.get_moving_mean = lambda x, winsize: self.calc_moving_mean(x, winsize, fps)
        self.get_moving_std = lambda x, winsize: self.calc_moving_std(x, winsize, fps)

        delta_scales_ = [100, 300, 500]
        window_sizes_ = [300, 500]

        if "delta_scales" not in stft_cfg.keys():
            self.logger.info(
                "Scale valuess can not be found in configuration for delta features."
                + f"Default values are {str(delta_scales_)[1:-1]}."
            )

        if "window_sizes" not in stft_cfg.keys():
            self.logger.info(
                "Window sizes can not be found in configuration for window features."
                + f"Default values are {str(window_sizes_)[1:-1]}."
            )

        self.stft_cfg["delta_scales"] = stft_cfg.get("delta_scales", delta_scales_)
        self.stft_cfg["window_sizes"] = stft_cfg.get("window_sizes", window_sizes_)

        self.stft_set = ["pose", "distance", "angle"]
        for ft_set in self.stft_set:
            ft_set_dt = ft_set + "_delta"
            self.stft_cfg[ft_set] = stft_cfg.get(ft_set, [])
            self.stft_cfg[ft_set_dt] = stft_cfg.get(ft_set_dt, [])

        self.angle_between = self.angle_between_atan

    @staticmethod
    def angle_between_arccos(v1, v2):
        """
        Returns the abs(angle) in radians between vectors 'v1' and 'v2'.
        angle_between((1, 0, 0), (0, 1, 0)) --> 1.5707963267948966
        angle_between((1, 0, 0), (1, 0, 0)) --> 0.0
        angle_between((1, 0, 0), (-1, 0, 0)) --> 3.141592653589793
        """
        assert isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray)

        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)

        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    @staticmethod
    def angle_between_atan(v1, v2):
        """
        Returns the abs(angle) in radians between vectors 'v1' and 'v2'.
        """
        assert isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray)

        angle = np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))

        return np.abs(angle)

    def get_group_value(self, stft_group, opt):
        if opt == "avg":
            group_value = np.nanmean(stft_group, axis=1)
        elif opt == "min":
            group_value = np.nanamin(stft_group, axis=1)
        elif opt == "max":
            group_value = np.nanmax(stft_group, axis=1)
        else:
            raise ValueError(f"Unkown option {opt} is given for feature group.")
        return group_value

    @staticmethod
    def calc_delta(x, scale, fps):
        # In terms of millisecond.
        delta_values = []
        scale_frame = math.ceil(fps * (1000 / scale))

        y = uniform_filter1d(x, size=scale_frame, axis=0)
        delta_y = np.abs(np.gradient(y, 1 / fps * 1000, axis=0, edge_order=2))
        delta_values.append(delta_y)

        return delta_values

    @staticmethod
    def calc_moving_mean(x, winsize, fps):
        mean_values = []
        w_frame = math.ceil(fps * (winsize / 1000))

        mean_values.append(x.rolling(w_frame, min_periods=1, center=True).mean())

        return mean_values

    @staticmethod
    def calc_moving_std(x, winsize, fps):
        std_values = []
        w_frame = math.ceil(fps * (winsize / 1000))

        std_values.append(x.rolling(w_frame, min_periods=1, center=True).std())

        return std_values

    def extract(self, ft_set, df_pose, ft_cfg_set):
        extraction_functions = {
            "pose": self._extract_pose,
            "angle": self._extract_angle,
            "distance": self._extract_distance,
        }
        val = extraction_functions[ft_set](df_pose, ft_cfg_set)
        return val

    def get_column_names(self, ft_set):
        stft_cfg = self.stft_cfg
        name_col = []

        def get_stft_name(defn):
            if isinstance(defn, dict):
                name = (
                    list(defn.keys())[0]
                    + "("
                    + ",".join(["-".join(item) for item in list(defn.values())[0]])
                    + ")"
                )
            elif isinstance(defn, list):
                name = "-".join(defn)
            else:
                raise ValueError(
                    f"Given feature definition {defn} has incorrect formatting."
                )
            return name

        if not stft_cfg.get(ft_set, False):
            raise ValueError(f"Unkown value {ft_set} is given for feature set.")

        if "pose" in ft_set:
            ft_names = list(
                itertools.chain.from_iterable(
                    ([item + "_x"], [item + "_y"]) for item in stft_cfg[ft_set]
                )
            )
        else:
            ft_names = stft_cfg[ft_set]

        if "delta" not in ft_set:
            name_col = [ft_set + "." + get_stft_name(item) for item in ft_names]

        else:
            scales = stft_cfg["delta_scales"]
            name_col = misc.flatten(
                [
                    [
                        ft_set + "." + get_stft_name(item) + ".s" + str(t)
                        for item in ft_names
                    ]
                    for t in scales
                ]
            )

        return name_col

    @staticmethod
    def _get_coord(df_pose, name, axis):
        # Axis name x or y.
        name_c = name + "_" + axis

        if name_c in df_pose.columns:
            coord = df_pose[name_c]
        elif name == "origin":
            coord = np.zeros(df_pose.shape[0])
        else:
            raise ValueError(f"No coordinate values can be found for {name}.")

        return coord

    def _extract_pose(self, df_pose, body_parts):
        xy_pose_values = np.ndarray((df_pose.shape[0], len(body_parts) * 2))

        if not isinstance(body_parts, list):
            raise ValueError(
                f"Given argument has type {type(body_parts)}."
                + "Pose features should be defined by a list of body-parts."
            )

        for i, bp in enumerate(body_parts):
            if not isinstance(bp, str):
                raise ValueError(
                    f"Given feature definition contains {bp}, which is not a body-part."
                )
            xy_pose_values[:, i * 2] = self.__class__._get_coord(df_pose, bp, "x")
            xy_pose_values[:, i * 2 + 1] = self.__class__._get_coord(df_pose, bp, "y")

        return xy_pose_values

    def _extract_angle(self, df_pose, triplets):
        angle_values = np.ndarray((df_pose.shape[0], len(triplets)))

        def f_angle(x):
            return self.angle_between(x[:2] - x[2:4], x[4:] - x[2:4])

        def angle_along_axis(xy_values, angle_values):
            for j in range(xy_values.shape[0]):
                v1 = xy_values[j, :2] - xy_values[j, 2:4]
                v2 = xy_values[j, 4:] - xy_values[j, 2:4]
                angle_values[j, i] = self.angle_between(v1, v2)
            return angle_values

        for i, triplet in enumerate(triplets):
            if isinstance(triplet, dict):
                opt = list(triplet.keys())[0]
                group = list(triplet.values())[0]
                if len(group) > 0 and opt in ["avg", "min", "max"]:
                    angle_group = self._extract_angle(df_pose, group)
                else:
                    raise ValueError(f"Given feature definition {triplet} is unknown.")
                angle_values[:, i] = self.get_group_value(angle_group, opt)
            else:
                xy_values, _ = self._extract_pose(df_pose, triplet)
                # angle_values[:, i] = np.apply_along_axis(f_angle, 1, xy_values)
                # This is somehow faster.
                angle_values[:, i] = angle_along_axis(xy_values, angle_values)

        return angle_values

    def _extract_distance(self, df_pose, pairs):
        distance_values = np.ndarray((df_pose.shape[0], len(pairs)))

        for i, pair in enumerate(pairs):
            if isinstance(pair, dict):
                opt = list(pair.keys())[0]
                group = list(pair.values())[0]
                if len(group) > 0 and opt in ["avg", "min", "max"]:
                    distance_group = self._extract_distance(df_pose, group)
                else:
                    raise ValueError(f"Given feature definition {pair} is unkwon.")
                distance_values[:, i] = self.get_group_value(distance_group, opt)
            else:
                xy_values = self._extract_pose(df_pose, pair)
                diff_xy = xy_values[:, 2:4] - xy_values[:, :2]
                distance_values[:, i] = np.sqrt(diff_xy[:, 0] ** 2 + diff_xy[:, 1] ** 2)

        return distance_values

    def _extract_moving_stat(self, df_stft, stft_names_dict, stat, winsizes):
        if stat == "mean":
            get_moving_stat = self.get_moving_mean
        elif stat == "std":
            get_moving_stat = self.get_moving_std
        else:
            raise ValueError(f"Unkown value {stat} is given for moving statistics.")

        name_col = df_stft.columns

        mv_stat = pd.concat(
            itertools.chain(*map(lambda w: get_moving_stat(df_stft, w), winsizes)),
            axis=1,
        )
        df_stat = pd.DataFrame(data=mv_stat)

        stat_columns = misc.flatten(
            [
                [
                    stat + "." + stft_names_dict[name] + ".w" + str(w)
                    for name in name_col
                ]
                for w in winsizes
            ]
        )
        name_dict = {i: stat_columns[i] for i in range(len(stat_columns))}
        df_stat.columns = list(name_dict.keys())

        return df_stat, name_dict

    def extract_snap_stft(self, df_pose):
        stft_cfg = self.stft_cfg

        df_snap_list = []

        for ft_set in self.stft_set:
            if stft_cfg.get(ft_set, False):
                temp_df = pd.DataFrame(self.extract(ft_set, df_pose, stft_cfg[ft_set]))
                temp_df.columns = self.get_column_names(ft_set)
                df_snap_list.append(temp_df)

        if len(df_snap_list) <= 0:
            raise ValueError(
                "At least one snap feature must given in the feature configuration."
            )

        df_snap = pd.concat(df_snap_list, axis=1)

        name_col = df_snap.columns
        name_dict = {i: name_col[i] for i in range(len(name_col))}
        df_snap.columns = list(name_dict.keys())
        self.ftname_to_snapft = name_dict

        return df_snap, name_dict

    def extract_delta_stft(self, df_pose):
        stft_cfg = self.stft_cfg
        delta_scales = stft_cfg["delta_scales"]

        df_delta_list = []

        for ft_set in self.stft_set:
            ft_set_dt = ft_set + "_delta"

            if stft_cfg.get(ft_set_dt, False):
                temp_snap = self.extract(ft_set, df_pose, stft_cfg[ft_set_dt])
                temp_delta = itertools.chain(
                    *map(
                        lambda s: self.get_delta(temp_snap, s),
                        delta_scales,
                    )
                )

                temp_df = pd.DataFrame(
                    np.concatenate(
                        tuple(temp_delta),
                        axis=1,
                    ),
                    columns=self.get_column_names(ft_set_dt),
                )
                df_delta_list.append(temp_df)

        if len(df_delta_list) <= 0:
            raise ValueError(
                "At least one delta feature must given in the feature configuration."
            )

        df_delta = pd.concat(df_delta_list, axis=1)

        name_col = df_delta.columns
        name_dict = {i: name_col[i] for i in range(len(name_col))}
        df_delta.columns = list(name_dict.keys())
        self.ftname_to_deltaft = name_dict

        return df_delta, name_dict

    def extract_window_snap_stft(self, df_stft, opt):
        window_sizes = self.stft_cfg["window_sizes"]
        if opt == "mean":
            df_window, name_dict = self._extract_moving_stat(
                df_stft, self.ftname_to_snapft, "mean", window_sizes
            )
            self.ftname_to_snapft_mean = name_dict
        elif opt == "std":
            df_window, name_dict = self._extract_moving_stat(
                df_stft, self.ftname_to_snapft, "std", window_sizes
            )
            self.ftname_to_snapft_std = name_dict
        return df_window, name_dict

    def extract_window_delta_stft(self, df_stft, opt):
        window_sizes = self.stft_cfg["window_sizes"]
        if opt == "mean":
            df_window, name_dict = self._extract_moving_stat(
                df_stft, self.ftname_to_deltaft, "mean", window_sizes
            )
            self.ftname_to_deltaft_mean = name_dict
        elif opt == "std":
            df_window, name_dict = self._extract_moving_stat(
                df_stft, self.ftname_to_deltaft, "std", window_sizes
            )
            self.ftname_to_deltaft_std = name_dict
        return df_window, name_dict
