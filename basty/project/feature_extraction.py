import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import Normalizer

import basty.utils.preprocessing as prep
import basty.utils.misc as misc

from basty.project.experiment_processing import Project
from basty.feature_extraction.body_pose import BodyPose
from basty.feature_extraction.spatiotemporal_features import SpatioTemporal
from basty.feature_extraction.wavelet_transformation import WaveletTransformation


class FeatureExtraction(Project):
    def __init__(self, main_cfg_path, **kwargs):
        Project.__init__(self, main_cfg_path, **kwargs)
        self.init_preprocessing_kwargs(**kwargs)
        self.init_behavioral_reprs_kwargs(**kwargs)

    @misc.timeit
    def compute_pose_values(self):
        bp = BodyPose(self.pose_cfg)

        pbar = tqdm(self.data_path_dict.items())
        for name, data_path in pbar:
            pbar.set_description(f"Computing pose values for experiment {name}")

            expt_path = self.expt_path_dict[name]
            expt_record = self._load_joblib_object(expt_path, "expt_record.z")

            expt_idx_key = []
            df_pose_list = []
            df_llh_list = []

            expt_pose_path = expt_path / "pose.pkl"
            if expt_pose_path.exists():
                self.logger.file_already_exists("pose.pkl", expt_pose_path)
            else:
                file_path_list = list(data_path.glob(name + "-" + "[0-9]" * 4 + ".csv"))
                # At least one .csv file  with appropriate naming must be found
                # in the given experiment data directory.
                assert len(file_path_list) != 0

                for file_path in tqdm(sorted(file_path_list), leave=False, desc=name):
                    expt_idx = file_path.stem.split("-")[-1]
                    expt_idx_key.append(int(expt_idx))

                    df_coord = pd.read_csv(file_path, low_memory=False)

                    if self.compute_oriented_pose:
                        pose_tmp, llh_tmp = bp.get_oriented_pose(df_coord)
                    else:
                        pose_tmp, llh_tmp = bp.get_pose(df_coord)

                    df_llh_list.append(llh_tmp)

                    if self.compute_egocentric_frames:
                        df_pose_list.append(bp.make_frames_egocentric(pose_tmp))
                    else:
                        df_pose_list.append(pose_tmp)

                expt_idx_key = np.array(expt_idx_key)
                assert expt_idx_key.shape[0] == 1 or (np.diff(expt_idx_key) == 1).all()

                df_llh = pd.concat(
                    df_llh_list,
                    keys=expt_idx_key,
                    names=["video-part", "frame-index"],
                )

                if self.save_likelihood:
                    self._save_pandas_dataframe(df_llh, expt_path, "llh.pkl")

                def pose_interpolate(df_pose):
                    df_pose.interpolate(
                        method=self.interpolation_method,
                        **self.interpolation_kwargs,
                        axis=0,
                        inplace=True,
                    )
                    df_pose = prep.fill_interpolate(df_pose)
                    return df_pose

                df_pose = pd.concat(
                    df_pose_list, keys=expt_idx_key, names=["video-part", "frame-index"]
                )
                df_pose.reset_index(inplace=True)
                vcols = [
                    col
                    for col in df_pose.columns
                    if col not in ["video-part", "frame-index"]
                ]

                if df_pose[vcols].isnull().values.any():
                    df_pose = pose_interpolate(df_pose[vcols])

                if self.local_outlier_winsize > 0:
                    df_pose[vcols] = prep.remove_local_outliers(
                        df_pose[vcols],
                        winsize=self.local_outlier_winsize,
                        threshold=self.local_outlier_threshold,
                    )

                if self.decreasing_llh_winsize > 0:
                    df_pose[vcols], _ = prep.remove_decrasing_llh_frames(
                        df_pose[vcols],
                        df_llh,
                        winsize=self.decreasing_llh_winsize,
                        lower_z=self.decreasing_llh_lower_z,
                    )

                df_pose[vcols], _ = prep.remove_low_llh_frames(
                    df_pose[vcols], df_llh, threshold=self.low_llh_threshold
                )
                df_pose.replace([np.inf, -np.inf], np.nan, inplace=True)

                df_pose[vcols] = pose_interpolate(df_pose[vcols])

                if self.median_filter_winsize > 0:
                    df_pose[vcols] = prep.median_filter(
                        df_pose[vcols], self.median_filter_winsize
                    )

                # Interpolate jumps based on the gradient quantiles.
                if self.jump_quantile > 0:
                    df_pose[vcols] = prep.remove_jumps(
                        df_pose[vcols], q=self.jump_quantile
                    )
                    df_pose[vcols] = pose_interpolate(df_pose[vcols])

                if self.boxcar_filter_winsize > 0:
                    df_pose[vcols] = prep.boxcar_center_filter(
                        df_pose[vcols], self.boxcar_filter_winsize
                    )

                if self.kalman_filter_kwargs:
                    df_pose = prep.rts_smoother_one_dimensional(
                        df_pose, self.kalman_filter_kwargs, dim_x=2, dim_z=1
                    )

                # Fix and name multi-index.
                df_pose.set_index(["video-part", "frame-index"], inplace=True)
                for i, lvl in enumerate(df_pose.index.levels):
                    df_pose.index = df_pose.index.set_levels(lvl.astype(int), level=i)

                # Check and count NaN and Inf values.
                count = np.isinf(df_pose).values.sum()
                count += np.isnan(df_pose).values.sum()
                if count > 0:
                    self.logger.invalid_value_warning("pose values", count)

                self._save_pandas_dataframe(df_pose, expt_path, "pose.pkl")

                group_indices = df_pose.groupby(level=0, as_index=True).indices
                part_to_frame_count = {
                    key: max(val) + 1 for key, val in group_indices.items()
                }

                expt_record.part_to_frame_count = part_to_frame_count
                self._save_joblib_object(
                    expt_record,
                    expt_path,
                    "expt_record.z",
                    "with the frame count information",
                )

    @misc.timeit
    def compute_spatiotemporal_features(self, delta=True, snap=True):
        stft_extractor = SpatioTemporal(self.fps, self.feature_cfg)

        pbar = tqdm(self.expt_path_dict.items())
        for name, expt_path in pbar:
            pbar.set_description(f"Computing spatiotempral features of {name}")

            df_pose = self._load_pandas_dataframe(expt_path, "pose.pkl")

            if delta:
                delta_path = expt_path / "delta_stft.pkl"
                if delta_path.exists():
                    self.logger.file_already_exists(
                        "delta spatio-temporal features", delta_path
                    )
                else:
                    df_delta, ftname_to_deltaft = stft_extractor.extract_delta_stft(
                        df_pose
                    )
                    self._save_yaml_dictionary(
                        ftname_to_deltaft, expt_path, "ftname_to_deltaft.yaml"
                    )
                    self._save_pandas_dataframe(df_delta, expt_path, "delta_stft.pkl")

            if snap:
                snap_path = expt_path / "snap_stft.pkl"
                if snap_path.exists():
                    self.logger.file_already_exists(
                        "snap spatio-temporal features", snap_path
                    )
                else:
                    df_snap, ftname_to_snapft = stft_extractor.extract_snap_stft(
                        df_pose
                    )
                    self._save_yaml_dictionary(
                        ftname_to_snapft, expt_path, "ftname_to_snapft.yaml"
                    )
                    self._save_pandas_dataframe(df_snap, expt_path, "snap_stft.pkl")

    @misc.timeit
    def compute_postural_dynamics(self):
        wt_detached_cfg = self.temporal_cfg.get("wt_detached", {})
        wt_cfg = self.temporal_cfg.get("wt", {})

        pbar = tqdm(self.expt_path_dict.items())
        for name, expt_path in pbar:
            pbar.set_description(f"Computing postural dynamics of {name}")

            ws_path = expt_path / "wsft.npy"
            if ws_path.exists():
                self.logger.file_already_exists("wavelet spectrum values", ws_path)
            else:
                snap_stft = self._load_pandas_dataframe(expt_path, "snap_stft.pkl")
                snap_stft = snap_stft.to_numpy()
                ftname_to_snapft = self._load_yaml_dictionary(
                    expt_path, "ftname_to_snapft.yaml"
                )
                wsft_list = []

                for key in ftname_to_snapft.keys():
                    idx = key
                    snap_wv_cfg = wt_detached_cfg.get(key, wt_cfg)

                    if not snap_wv_cfg:
                        self.missing_cfg_key_error(
                            self, "wavelet transformation configuration"
                        )

                    wvt = WaveletTransformation(self.fps, snap_wv_cfg)
                    ftws_r = wvt.cwtransform(snap_stft[:, [idx]])
                    wsft_list.append(wvt.normalize_channels(ftws_r))

                wsft = np.concatenate(wsft_list, axis=-1)
                self._save_numpy_array(wsft, expt_path, "wsft.npy")

    @staticmethod
    def compute_cartesian_blent(wsft, ftname_to_snapft):
        component_dict = defaultdict(list)
        new_wsft_list = []

        for i, val in enumerate(ftname_to_snapft.items()):
            name_split = val.split(".")
            if name_split[0] == "pose":
                bp_name = name_split[-1].split("_")[0]
                component_dict[bp_name].append(i)
            else:
                new_wsft_list.append(wsft[:, :, i])

        for _, val in component_dict.items():
            new_wsft_list.append(np.sum(wsft[:, :, val], axis=2) / len(val))

        return np.stack(new_wsft_list, axis=-1)

    @misc.timeit
    def compute_behavioral_representations(self):
        pbar = tqdm(self.expt_path_dict.items())
        for name, expt_path in pbar:
            pbar.set_description(f"Computing behavioral representations of {name}")

            expt_path = self.expt_path_dict[name]

            wsft = self._load_numpy_array(expt_path, "wsft.npy")
            if self.use_cartesian_blent:
                ftname_to_snapft = self._load_yaml_dictionary(
                    expt_path, "ftname_to_snapft.yaml"
                )
                wsft = self.__class__.compute_cartesian_blent(wsft, ftname_to_snapft)

            flat_dim = wsft.shape[1] * wsft.shape[2]
            ftws_flat = np.reshape(wsft, (wsft.shape[0], flat_dim))

            l1_normalizer = Normalizer(norm=self.norm)
            X_expt = l1_normalizer.transform(ftws_flat)

            self._save_numpy_array(X_expt, expt_path, "behavioral_reprs.npy")
