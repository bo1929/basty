import logging
import random

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.filters import uniform_filter1d

import basty.utils.misc as misc


class BodyPose:
    def __init__(self, pose_cfg={}):
        self.pose_cfg = pose_cfg.copy()
        self.logger = logging.getLogger("main")

        self.counterparts = self.pose_cfg.get("counterparts", {})
        self.singles = self.pose_cfg.get("singles", [])
        self.connected_parts = self.pose_cfg.get("connected_parts", [])

        self.groups = self.pose_cfg.get("groups", {})

        self.defined_points = self.pose_cfg.get("defined_points", {})
        self.centerline = self.pose_cfg.get("centerline", [])

        if len(self.centerline) != 2 and len(self.centerline) != 0:
            raise ValueError(
                "A centerline must be defined by two body-parts"
                + f"Given value {self.centerline} is not suitable for definition."
            )

    @staticmethod
    def get_sub_df_pose(df_pose, ind):
        if ind not in ["x", "y"]:
            raise ValueError(f"Given sub-dataframe indicator {ind} is not defined.")
        sub_df = df_pose[df_pose.columns[df_pose.columns.str.endswith(ind)]]
        sub_df.columns = [name[:-2] for name in sub_df.columns]
        return sub_df

    @staticmethod
    def get_sub_df_coord(df_coord, ind):
        if ind not in ["x", "y", "likelihood"]:
            raise ValueError(f"Given sub-dataframe indicator {ind} is not defined.")
        try:
            sub_df = df_coord[df_coord.columns[df_coord.loc[1] == ind]]
            sub_df.columns = sub_df.iloc[0]
            sub_df = sub_df.drop([0, 1]).fillna(0).reset_index(drop=True)
            assert sub_df.shape[1] > 0
        except AssertionError:
            sub_df = df_coord[df_coord.columns[df_coord.loc[2] == ind]]
            sub_df.columns = sub_df.iloc[1]
            sub_df = sub_df.drop([0, 1, 2]).fillna(0).reset_index(drop=True)
        return sub_df

    def split_sub_df_coord(self, df_coord):
        llh = self.get_sub_df_coord(df_coord, "likelihood").astype(float)
        x = self.get_sub_df_coord(df_coord, "x").astype(float)
        y = self.get_sub_df_coord(df_coord, "y").astype(float)
        return llh, x, y

    def make_frames_egocentric(self, df_pose, per_frame=False):
        spine = self.centerline
        eps = np.finfo(np.float32).eps
        if len(spine) == 0:
            raise ValueError(
                "A centerline must be given to construct egencentric frames."
            )

        dfPose_x = self.get_sub_df_pose(df_pose, "x")
        dfPose_y = self.get_sub_df_pose(df_pose, "y")

        dfPose_new_dict = {}

        if per_frame:
            raise NotImplementedError(
                "Constructing egocentric frames per frame is not implemented yet."
            )
        else:
            s1_x = dfPose_x[spine[0]].to_numpy()
            s1_y = dfPose_y[spine[0]].to_numpy()

            s2_x = dfPose_x[spine[1]].to_numpy()
            s2_y = dfPose_y[spine[1]].to_numpy()

            r2_shifted = np.stack((s2_x - s1_x, s2_y - s1_y), axis=1)

            r2_x = np.linalg.norm(r2_shifted, axis=1)
            r2_y = np.zeros(s2_y.shape[0])
            r2 = np.stack((r2_x, r2_y), axis=1)

            def get_rotation_matrix(v1, v2):
                r1, r2 = (v1 / np.linalg.norm(v1)).reshape(2), (
                    v2 / np.linalg.norm(v2)
                ).reshape(2)
                rc = np.cross(r1, r2)
                rd = np.dot(r1, r2)
                rc_norm = np.linalg.norm(rc)

                kmat = np.array([[0, -rc], [rc, 0]])

                rotation_matrix = (
                    np.eye(2)
                    + kmat
                    + kmat.dot(kmat) * ((1 - rd) / (rc_norm + eps) ** 2)
                )

                return rotation_matrix

            egocentric_transform = np.ndarray((r2.shape[0], r2.shape[1], r2.shape[1]))
            for i in range(egocentric_transform.shape[0]):
                egocentric_transform[i] = get_rotation_matrix(
                    r2_shifted[i, :], r2[i, :]
                )

            # rs_pinv = np.linalg.pinv(r2_shifted[:,:,np.newaxis])
            # egocentric_transform = np.matmul(r2[:, :, np.newaxis], rs_pinv)

            for name in dfPose_x.columns:
                r = np.stack(
                    (
                        dfPose_x[name].to_numpy() - s1_x,
                        dfPose_y[name].to_numpy() - s1_y,
                    ),
                    axis=1,
                )
                r_new = np.matmul(egocentric_transform, r[:, :, np.newaxis])

                dfPose_new_dict[name + "_x"] = r_new[:, 0, 0]
                dfPose_new_dict[name + "_y"] = r_new[:, 1, 0]

            # It's very likely for df_pose to contain NaNs at this point.
            # Interpolating NaN's might be a good idea.
            dfEgocentric = pd.DataFrame.from_dict(dfPose_new_dict)
            return dfEgocentric

    def _finalize_orientation(self, orientations, left_llh, right_llh, winsize=3):
        left_llh = uniform_filter1d(left_llh, size=winsize)
        right_llh = uniform_filter1d(right_llh, size=winsize)

        left_idx = np.asarray(left_llh > right_llh).nonzero()[0]
        right_idx = np.asarray(right_llh > left_llh).nonzero()[0]

        orientations["left"].update(left_idx)
        orientations["right"].update(right_idx)
        orientations["idx"].difference_update(left_idx)
        orientations["idx"].difference_update(right_idx)

        for idx in orientations["idx"]:
            lc = np.abs(np.min(np.array(list(orientations["left"])) - idx))
            rc = np.abs(np.min(np.array(list(orientations["right"])) - idx))
            if lc < rc:
                orientations["left"].add(idx)
            elif rc < lc:
                orientations["right"].add(idx)
            else:
                rand_orient = random.choice(["right", "left"])
                orientations[rand_orient].add(idx)

        orientations["idx"].difference_update(orientations["idx"])

        return orientations

    def _window_count_orientation(self, orientations, left_llh, right_llh, winsize):
        likely_left = left_llh > right_llh

        for idx, window in enumerate(sliding_window_view(likely_left, 2 * winsize + 1)):
            if idx in orientations["idx"]:
                left_count = np.count_nonzero(window)
                right_count = winsize - left_count
                if left_count > winsize:
                    orientations["left"].add(idx)
                    orientations["idx"].remove(idx)
                elif right_count > winsize:
                    orientations["right"].add(idx)
                    orientations["idx"].remove(idx)
                else:
                    pass

        return orientations

    def _nearest_time_point_orientation(self, orientations, left_llh, right_llh):
        for idx in list(orientations["idx"]):
            left_closest = (
                min(orientations["left"], key=lambda x: abs(x - idx))
                if len(orientations["left"]) > 0
                else -1
            )
            right_closest = (
                min(orientations["right"], key=lambda x: abs(x - idx))
                if len(orientations["right"]) > 0
                else -1
            )

            if (left_llh.iloc[idx] > right_llh.iloc[idx]) and (
                abs(idx - left_closest) > abs(idx - right_closest)
            ):
                orientations["left"].add(idx)
                orientations["idx"].remove(idx)
            elif (left_llh.iloc[idx] > right_llh.iloc[idx]) and (
                abs(idx - left_closest) > abs(idx - right_closest)
            ):
                orientations["right"].add(idx)
                orientations["idx"].remove(idx)
            else:
                pass

        return orientations

    def _threshold_orientation(
        self, orientations, threshold, left_llh, right_llh, winsize=10
    ):
        left_llh = uniform_filter1d(left_llh, size=winsize)
        right_llh = uniform_filter1d(right_llh, size=winsize)

        left_idx = np.asarray((left_llh - right_llh) > threshold).nonzero()[0]
        right_idx = np.asarray((right_llh - left_llh) > threshold).nonzero()[0]

        orientations["left"].update(left_idx)
        orientations["right"].update(right_idx)
        orientations["idx"].difference_update(left_idx)
        orientations["idx"].difference_update(right_idx)

        return orientations

    def get_pose(self, df_coord):
        singles = self.singles

        llh, x, y = self.split_sub_df_coord(df_coord)

        if len(singles) == 0:
            self.logger.warning(
                "Empty singles body-part list in the pose configuration."
            )
            self.logger.info("Constructing pose by using all possible body-parts.")
            singles = llh.columns

        pose_dict = {}
        llh_dict = {}

        for name in singles:
            pose_dict[name + "_x"] = x[name]
            pose_dict[name + "_y"] = y[name]
            llh_dict[name] = llh[name]

        for name, components in self.defined_points.items():
            if (set(components).issubset(x.columns)) and (
                set(components).issubset(y.columns)
            ):
                def_xval = x[components].mean(axis=1)
                def_yval = y[components].mean(axis=1)
                def_llhval = llh[components].mean(axis=1)

                pose_dict[name + "_x"] = def_xval
                pose_dict[name + "_y"] = def_yval
                llh_dict[name] = def_llhval

        df_pose = pd.DataFrame.from_dict(pose_dict)
        df_llh = pd.DataFrame.from_dict(llh_dict)

        return df_pose, df_llh

    def get_oriented_pose(self, df_coord, threshold=0.5, verbose=False):
        counterparts = self.counterparts
        self.logger = logging.getLogger("main")

        def log_ckp(method, body_part, percent):
            self.logger.info(
                f"After applying {method} for {body_part}, "
                + f"orientations of {round(percent, 2)} are determined."
            )

        if len(counterparts) == 0:
            raise ValueError("Empty counterparts in the pose configuration.")

        llh, x, y = self.split_sub_df_coord(df_coord)

        pose_dict = {}
        llh_dict = {}

        for name, [left_part, right_part] in counterparts.items():
            connected_left = misc.in_nested_list(self.connected_parts, left_part)
            connected_right = misc.in_nested_list(self.connected_parts, right_part)

            left_llh = (
                llh[connected_left].mean(axis=1)
                if len(connected_left) > 0
                else llh[left_part]
            )
            right_llh = (
                llh[connected_right].mean(axis=1)
                if len(connected_right) > 0
                else llh[right_part]
            )

            # Determine orientation for each counterpart, based on likelihood values.
            orientations = {
                "left": set([]),
                "right": set([]),
                "idx": set(range(0, left_llh.shape[0])),
            }

            # (1) Applying hard threshold.
            orientations = self._threshold_orientation(
                orientations, threshold, left_llh, right_llh, winsize=6
            )
            if verbose:
                log_ckp(
                    "hard threshold comparison",
                    name,
                    1 - (len(orientations["idx"]) / left_llh.shape[0]),
                )

            # (2) In order to determine retained frames.
            orientations = self._window_count_orientation(
                orientations, left_llh, right_llh, winsize=15
            )
            if verbose:
                log_ckp(
                    "window count based comparison",
                    name,
                    1 - (len(orientations["idx"]) / left_llh.shape[0]),
                )

            # (3) If still could not be determined, compare left-right counterparts.
            orientations = self._finalize_orientation(
                orientations, left_llh, right_llh, winsize=10
            )

            if len(orientations["idx"]) > 0:
                raise AssertionError(
                    "Orientation could not be determined for some frames."
                )

            # Take coordinate values based on orientations.
            x_oriented = np.ndarray((x.shape[0]))
            y_oriented = np.ndarray((y.shape[0]))
            llh_oriented = np.ndarray((y.shape[0]))

            x_oriented[list(orientations["left"])] = x[left_part].iloc[
                list(orientations["left"])
            ]
            y_oriented[list(orientations["left"])] = y[left_part].iloc[
                list(orientations["left"])
            ]
            llh_oriented[list(orientations["left"])] = llh[left_part].iloc[
                list(orientations["left"])
            ]

            x_oriented[list(orientations["right"])] = x[right_part].iloc[
                list(orientations["right"])
            ]
            y_oriented[list(orientations["right"])] = y[right_part].iloc[
                list(orientations["right"])
            ]
            llh_oriented[list(orientations["right"])] = llh[right_part].iloc[
                list(orientations["right"])
            ]

            pose_dict[name + "_x"] = x_oriented
            pose_dict[name + "_y"] = y_oriented
            llh_dict[name] = llh_oriented

        for name, components in self.defined_points.items():
            def_xval = []
            def_yval = []
            def_llhval = []

            for item in components:
                if (item in x.columns) and (item in y.columns):
                    def_xval.append(x[item].to_numpy())
                    def_yval.append(y[item].to_numpy())
                    def_llhval.append(llh[item].to_numpy())
                if (item + "_x" in pose_dict.keys()) and (
                    item + "_y" in pose_dict.keys()
                ):
                    def_xval.append(pose_dict[item + "_x"])
                    def_yval.append(pose_dict[item + "_y"])
                    def_llhval.append(llh_dict[item])

            pose_dict[name + "_x"] = np.mean(np.array(def_xval), axis=0)
            pose_dict[name + "_y"] = np.mean(np.array(def_yval), axis=0)
            llh_dict[name] = np.mean(np.array(def_llhval), axis=0)

        df_pose, df_llh = self.get_pose(df_coord)

        df_pose_oriented = pd.DataFrame.from_dict(pose_dict)
        df_llh_oriented = pd.DataFrame.from_dict(llh_dict)

        return (
            pd.concat([df_pose, df_pose_oriented], axis=1),
            pd.concat([df_llh, df_llh_oriented], axis=1),
        )
