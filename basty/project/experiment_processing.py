import warnings
import numpy as np

from tqdm import tqdm
from pathlib import Path

import basty.utils.io as io
import basty.utils.misc as misc

from basty.utils.annotations import HumanAnnotations as HumAnn
from basty.project.helper import SavingHelper, LoadingHelper, ParameterHandler, Logger
from basty.experiment_processing.experiment_info import ExptRecord
from basty.experiment_processing.experiment_outline import (
    DormantEpochs,
    ActiveBouts,
)


warnings.filterwarnings("ignore")


class Project(ParameterHandler, LoadingHelper, SavingHelper):
    def __init__(self, main_cfg_path, **kwargs):
        self.logger = Logger(filepath=None)
        ParameterHandler.__init__(self)
        LoadingHelper.__init__(self, self.logger)
        SavingHelper.__init__(self, self.logger)

        self.main_cfg = io.read_config(main_cfg_path)

        def try_to_get_value(cfg, key, name):
            try:
                value = cfg[key]
            except KeyError as err:
                self.logger.missing_cfg_key_error(name)
                self.logger.direct_error(err)
                raise
            return value

        def try_to_get_path(cfg, key, name):
            value = Path(try_to_get_value(cfg, key, name))
            return value

        def try_to_read_cfg(path, name):
            try:
                cfg = io.read_config(path)
            except FileNotFoundError as err:
                self.logger.file_missing(name, path, depth=2)
                self.logger.direct_error(err)
                raise
            return cfg

        # Read main project path and create a directories at that path.
        self.project_path = try_to_get_path(
            self.main_cfg, "project_path", "project path"
        )
        io.safe_create_dir(self.project_path)

        # Read each configuration file paths given in the main configuration.
        self.cfg_path_dict = try_to_get_value(
            self.main_cfg, "configuration_paths", "configuration paths"
        )
        self.pose_cfg_path = try_to_get_path(
            self.cfg_path_dict, "pose_cfg", "pose configuration path"
        )
        self.feature_cfg_path = try_to_get_path(
            self.cfg_path_dict, "feature_cfg", "feature configuration path"
        )
        self.temporal_cfg_path = try_to_get_path(
            self.cfg_path_dict, "temporal_cfg", "temporal configuration path"
        )

        # Read and load each each configuration file.
        self.pose_cfg = try_to_read_cfg(self.pose_cfg_path, "pose configuration")
        self.feature_cfg = try_to_read_cfg(
            self.feature_cfg_path, "feature configuration"
        )
        self.temporal_cfg = try_to_read_cfg(
            self.temporal_cfg_path, "temporal configuration"
        )

        # Read FPS value from the temporal configuration.
        self.fps = try_to_get_value(self.temporal_cfg, "fps", "fps value")

        # Read experiment data paths from the main configuration.
        expt_data_paths = try_to_get_value(
            self.main_cfg, "experiment_data_paths", "experiment data paths"
        )
        self.data_path_dict = {key: Path(val) for key, val in expt_data_paths.items()}

        # Duplicate and update configuration files to use later.
        # Save them into the project directory at the given project path.
        self.main_cfg["configuration_paths"]["feature_cfg"] = str(
            self.project_path / "feature_cfg.yaml"
        )
        self.main_cfg["configuration_paths"]["pose_cfg"] = str(
            self.project_path / "pose_cfg.yaml"
        )
        self.main_cfg["configuration_paths"]["temporal_cfg"] = str(
            self.project_path / "temporal_cfg.yaml"
        )

        cfg_path_dict = self.main_cfg["configuration_paths"]
        io.dump_yaml(self.pose_cfg, cfg_path_dict["pose_cfg"])
        io.dump_yaml(self.feature_cfg, cfg_path_dict["feature_cfg"])
        io.dump_yaml(self.temporal_cfg, cfg_path_dict["temporal_cfg"])
        io.dump_yaml(self.main_cfg, self.project_path / "main_cfg.yaml")

        # Create sub-directories in the project for each experiment.
        self.expt_names = []
        self.expt_path_dict = {}

        pbar = tqdm(self.data_path_dict.items())
        for name, _ in pbar:
            pbar.set_description(f"Creating empty experiment record objects for {name}")
            self.expt_names.append(name)

            expt_path = self.project_path / name
            self.expt_path_dict[name] = expt_path
            io.safe_create_dir(expt_path)
            io.safe_create_dir(expt_path / "embeddings")
            io.safe_create_dir(expt_path / "cluster_labels")
            io.safe_create_dir(expt_path / "behavior_labels")

            if not (expt_path / "expt_record.z").exists():
                expt_record = ExptRecord(
                    name, self.data_path_dict[name], expt_path, fps=self.fps
                )
                self._save_joblib_object(expt_record, expt_path, "expt_record.z")
            # else:
            #     self.logger.file_already_exists(
            #         "expt_record.z", expt_path / "expt_record.z", depth=2
            #     )

        # Process, save and visualize given human annotations.
        self.init_annotation_kwargs(**kwargs)
        self.annotation_path_dict = self.main_cfg.get("annotation_paths", {})

        pbar = tqdm(self.annotation_path_dict.items())
        for name, ann_path in pbar:
            expt_path = self.expt_path_dict[name]

            if not (expt_path / "annotations.npy").exists():
                pbar.set_description(f"Processing human annotations of {name}")

                annotator = HumAnn(ann_path)
                y_ann_list = annotator.get_annotations()
                y_ann = annotator.label_converter(
                    y_ann_list, priority_order=self.annotation_priority
                )
                self._save_numpy_array(y_ann, expt_path, "annotations.npy")

                expt_record = self._load_joblib_object(expt_path, "expt_record.z")
                expt_record.has_annotation = True
                expt_record._label_to_behavior = annotator.label_to_behavior
                expt_record._behavior_to_label = annotator.behavior_to_label
                expt_record.mask_annotated = (
                    annotator.behavior_to_label[annotator.inactive_behavior] != y_ann
                )
                self._save_joblib_object(
                    expt_record,
                    expt_path,
                    "expt_record.z",
                    "with annotation information",
                )
            # else:
            #     self.logger.file_already_exists(
            #         "annotations.npy", expt_path / "annotations.npy", depth=2
            #     )


class ExptDormantEpochs(Project):
    def __init__(self, main_cfg_path, use_supervised_learning=False, **kwargs):
        Project.__init__(self, main_cfg_path, **kwargs)
        self.use_supervised_learning = use_supervised_learning
        self.init_dormant_epochs_kwargs(self.fps, use_supervised_learning, **kwargs)

    @misc.timeit
    def outline_dormant_epochs(self):
        pbar = tqdm(self.expt_path_dict.items())
        for name, expt_path in pbar:
            pbar.set_description(f"Outlining dormant epochs of {name}")

            expt_record = self._load_joblib_object(expt_path, "expt_record.z")

            delta_stft = self._load_pandas_dataframe(expt_path, "delta_stft.pkl")
            ftname_to_deltaft = self._load_yaml_dictionary(
                expt_path, "ftname_to_deltaft.yaml"
            )

            if not self.datums:
                datums = list(ftname_to_deltaft.keys())
            else:
                datums = self.datums

            if not self.use_supervised_learning:
                value = DormantEpochs.get_datums_values(
                    delta_stft, datums=datums, winsize=self.datums_winsize
                )

                if self.threshold_log:
                    value = np.log2(value + 1)

                threshold = DormantEpochs.get_threshold(
                    value,
                    self.threshold_log,
                    self.threshold_key,
                    self.num_gmm_comp,
                    self.threshold_idx,
                )

                mask_dormant, final_labels = DormantEpochs.compute_dormant_epochs(
                    value,
                    threshold,
                    self.min_dormant,
                    self.tol_duration,
                    self.tol_percent,
                    self.epoch_winsize,
                )
            else:
                raise NotImplementedError

            expt_record.mask_dormant = mask_dormant

            self._save_joblib_object(
                expt_record,
                expt_path,
                "expt_record.z",
                "with a mask for dormant epochs",
            )


class ExptActiveBouts(Project):
    def __init__(self, main_cfg_path, use_supervised_learning=False, **kwargs):
        Project.__init__(self, main_cfg_path, **kwargs)
        self.use_supervised_learning = use_supervised_learning
        self.init_active_bouts_kwargs(self.fps, use_supervised_learning, **kwargs)

    @misc.timeit
    def outline_active_bouts(self):
        pbar = tqdm(self.expt_path_dict.items())
        for name, expt_path in pbar:
            pbar.set_description(f"Outlining active bouts of {name}")

            expt_record = self._load_joblib_object(expt_path, "expt_record.z")
            mask_dormant = expt_record.mask_dormant

            wsft = self._load_numpy_array(expt_path, "wsft.npy")
            ftname_to_snapft = self._load_yaml_dictionary(
                expt_path, "ftname_to_snapft.yaml"
            )

            if not self.use_supervised_learning:
                df_coefs = ActiveBouts.get_df_summary_coefs(
                    wsft, ftname_to_snapft, log=self.threshold_log, method="sum"
                )

                del wsft
                thresholds = []
                values = []

                if not any(self.datums_list):
                    datums_list = [[]]  # [[ftname] for ftname in df_coefs.columns]
                else:
                    datums_list = self.datums_list

                for datums in datums_list:
                    values.append(
                        ActiveBouts.get_datums_values(
                            df_coefs, datums=datums, winsize=self.datums_winsize
                        )
                    )
                    thresholds.append(
                        ActiveBouts.get_threshold(
                            values[-1][mask_dormant],
                            False,
                            self.threshold_key,
                            self.num_gmm_comp,
                            self.threshold_idx,
                        )
                    )
                del df_coefs

                mask_active, active_mask_per_datums = ActiveBouts.compute_active_bouts(
                    np.stack(values, axis=1),
                    thresholds,
                    winsize=self.post_processing_winsize,
                    wintype=self.post_processing_wintype,
                )
            else:
                raise NotImplementedError

            expt_record.mask_active = mask_active

            dormant_and_active = np.logical_and(
                expt_record.mask_active, expt_record.mask_dormant
            )

            dormant_and_active_percent = np.round(
                np.count_nonzero(dormant_and_active)
                * 100
                / dormant_and_active.shape[0],
                1,
            )
            self.logger.direct_info(
                f"Dormant and active frames are {dormant_and_active_percent}%."
            )

            mean_bout = np.round(
                np.mean(np.diff(misc.cont_intvls(dormant_and_active))) / self.fps, 2
            )
            self.logger.direct_info(
                f"Mean length of the dormant and active frames are {mean_bout} sec."
            )

            self._save_joblib_object(
                expt_record,
                expt_path,
                "expt_record.z",
                "with a mask for active bouts",
            )
