import warnings
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

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

        for name, _ in self.data_path_dict.items():
            self.expt_names.append(name)

            expt_path = self.project_path / name
            self.expt_path_dict[name] = expt_path
            io.safe_create_dir(expt_path)
            io.safe_create_dir(expt_path / "embeddings")
            io.safe_create_dir(expt_path / "clusterings")
            io.safe_create_dir(expt_path / "mappings")
            # io.safe_create_dir(expt_path / "figures")

            if not (expt_path / "expt_record.z").exists():
                self.logger.direct_info(
                    f"Creating empty experiment record objects for {name}"
                )
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

        for idx, (name, ann_path) in enumerate(self.annotation_path_dict.items()):
            expt_path = self.expt_path_dict[name]

            if not (expt_path / "annotations.npy").exists():
                self.logger.direct_info(f"Processing human annotations of {name}")

                annotator = HumAnn(
                    ann_path,
                    inactive_annotation=self.inactive_annotation,
                    noise_annotation=self.noise_annotation,
                    arouse_annotation=self.arouse_annotation,
                )

                if idx > 0:
                    assert annotator.behavior_to_label == expt_record.behavior_to_label
                    assert annotator.label_to_behavior == expt_record.label_to_behavior

                y_ann_list = annotator.get_annotations()
                y_ann = annotator.label_converter(
                    y_ann_list, priority_order=self.annotation_priority
                )
                self._save_numpy_array(y_ann, expt_path, "annotations.npy")

                expt_record = self._load_joblib_object(expt_path, "expt_record.z")
                expt_record.has_annotation = True
                expt_record.inactive_annotation = annotator.inactive_annotation
                expt_record.noise_annotation = annotator.noise_annotation
                expt_record.arouse_annotation = annotator.arouse_annotation
                expt_record.label_to_behavior = annotator.label_to_behavior
                expt_record.behavior_to_label = annotator.behavior_to_label
                expt_record.mask_noise = (
                    annotator.behavior_to_label[annotator.noise_annotation] == y_ann
                )
                mask_annotated = (
                    annotator.behavior_to_label[annotator.inactive_annotation] != y_ann
                )
                expt_record.mask_annotated = np.logical_and(
                    mask_annotated, np.logical_not(expt_record.mask_noise)
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
        X_expt_dict = dict()
        if not self.use_supervised_learning:
            threshold_expt_dict = dict()

        pbar = tqdm(self.expt_path_dict.items())
        for name, expt_path in pbar:
            pbar.set_description(
                f"Computing feature values of {name} for outlining dormant epochs"
            )

            expt_record = self._load_joblib_object(expt_path, "expt_record.z")
            delta_stft = self._load_pandas_dataframe(expt_path, "delta_stft.pkl")
            ftname_to_deltaft = self._load_yaml_dictionary(
                expt_path, "ftname_to_deltaft.yaml"
            )

            if not self.datums:
                datums = list(ftname_to_deltaft.keys())
            else:
                datums = self.datums

            X = DormantEpochs.get_datums_values(
                delta_stft, datums=datums, winsize=self.datums_winsize
            ).reshape((-1, 1))
            if self.log_scale:
                X = np.log2(X + 1)
            X_expt_dict[name] = X

            if not self.use_supervised_learning:
                threshold_expt_dict[name] = DormantEpochs.get_threshold(
                    X,
                    self.threshold_key,
                    self.num_gmm_comp,
                    self.threshold_idx,
                )

        if self.use_supervised_learning:
            X_train_list = []
            y_train_list = []

            for ann_expt_name, _ in self.annotation_path_dict.items():
                expt_path = self.expt_path_dict[ann_expt_name]
                expt_record = self._load_joblib_object(expt_path, "expt_record.z")
                y_ann = self._load_numpy_array(expt_path, "annotations.npy")
                lbl1 = expt_record.behavior_to_label[expt_record.arouse_annotation]
                lbl2 = expt_record.behavior_to_label[expt_record.noise_annotation]

                y_train = np.zeros(y_ann.shape, dtype=int)
                y_train[y_ann == lbl1] = 1
                y_train[y_ann == lbl2] = 2

                X_train_list.append(X_expt_dict[ann_expt_name])
                y_train_list.append(y_train)

            self.logger.direct_info("Training the decision tree for dormant epochs.")
            dormant_epochs = DormantEpochs()
            dormant_epochs.construct_dormant_epochs_decision_tree(
                X_train_list, y_train_list, **self.decision_tree_kwargs
            )

        pbar = tqdm(self.expt_path_dict.items())
        for name, expt_path in pbar:
            pbar.set_description(f"Outlining dormant epochs of {name}")

            X = X_expt_dict[name]
            expt_record = self._load_joblib_object(expt_path, "expt_record.z")

            if self.use_supervised_learning:
                mask_dormant = dormant_epochs.predict_dormant_epochs(
                    X,
                    min_dormant=self.min_dormant,
                    winsize=self.post_processing_winsize,
                    wintype=self.post_processing_wintype,
                )
            else:
                threshold = threshold_expt_dict[name]
                mask_dormant, final_labels = DormantEpochs.compute_dormant_epochs(
                    X,
                    threshold,
                    self.min_dormant,
                    self.tol_duration,
                    self.tol_percent,
                    self.epoch_winsize,
                )

            expt_record.mask_dormant = mask_dormant
            assert mask_dormant.any()

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
        X_expt_dict = dict()
        if not self.use_supervised_learning:
            thresholds_expt_dict = defaultdict(list)

        pbar = tqdm(self.expt_path_dict.items())
        for name, expt_path in pbar:
            pbar.set_description(
                f"Computing feature values of {name} for outlining active bouts"
            )

            expt_record = self._load_joblib_object(expt_path, "expt_record.z")
            wsft = self._load_numpy_array(expt_path, "wsft.npy")
            ftname_to_snapft = self._load_yaml_dictionary(
                expt_path, "ftname_to_snapft.yaml"
            )

            mask_dormant = expt_record.mask_dormant

            df_coefs = ActiveBouts.get_df_summary_coefs(
                wsft, ftname_to_snapft, log=self.log_scale, method="sum"
            )
            del wsft

            if not any(self.datums_list):
                datums_list = [[datum] for datum in df_coefs.columns]  # [[]]
            else:
                datums_list = self.datums_list

            values = []
            for datums in datums_list:
                values.append(
                    ActiveBouts.get_datums_values(
                        df_coefs, datums=datums, winsize=self.datums_winsize
                    )
                )
            del df_coefs

            X = np.stack(values, axis=1)
            X_expt_dict[name] = X
            del values

            if not self.use_supervised_learning:
                for idx, datums in enumerate(datums_list):
                    thresholds_expt_dict[name].append(
                        ActiveBouts.get_threshold(
                            X[mask_dormant, idx, np.newaxis],
                            self.threshold_key,
                            self.num_gmm_comp,
                            self.threshold_idx,
                        )
                    )

        if self.use_supervised_learning:
            X_train_list = []
            y_train_list = []

            for ann_expt_name, _ in self.annotation_path_dict.items():
                expt_path = self.expt_path_dict[ann_expt_name]
                expt_record = self._load_joblib_object(expt_path, "expt_record.z")
                y_ann = self._load_numpy_array(expt_path, "annotations.npy")
                behavior_to_label = expt_record.behavior_to_label
                lbl1 = behavior_to_label[expt_record.inactive_annotation]
                lbl2 = behavior_to_label[expt_record.noise_annotation]

                y_train = np.zeros(y_ann.shape, dtype=int)
                y_train[y_ann != lbl1] = 1
                y_train[y_ann == lbl2] = 2

                X_train_list.append(X_expt_dict[ann_expt_name])
                y_train_list.append(y_train)

            self.logger.direct_info("Training the decision tree for active bouts.")
            active_bouts = ActiveBouts()
            active_bouts.construct_active_bouts_decision_tree(
                X_train_list, y_train_list, **self.decision_tree_kwargs
            )

        pbar = tqdm(self.expt_path_dict.items())
        for name, expt_path in pbar:
            pbar.set_description(f"Outlining active bouts of {name}")

            X = X_expt_dict[name]
            expt_record = self._load_joblib_object(expt_path, "expt_record.z")

            if self.use_supervised_learning:
                mask_active = active_bouts.predict_active_bouts(
                    X,
                    winsize=self.post_processing_winsize,
                    wintype=self.post_processing_wintype,
                )
            else:
                thresholds = thresholds_expt_dict[name]
                mask_active, active_mask_per_datums = ActiveBouts.compute_active_bouts(
                    X,
                    thresholds,
                    winsize=self.post_processing_winsize,
                    wintype=self.post_processing_wintype,
                )

            expt_record.mask_active = mask_active
            assert mask_active.any()
            dormant_and_active = np.logical_and(
                expt_record.mask_active, expt_record.mask_dormant
            )
            assert dormant_and_active.any()
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
            self.logger.direct_info(f"Mean length of the bouts is {mean_bout} sec.")

            self._save_joblib_object(
                expt_record,
                expt_path,
                "expt_record.z",
                "with a mask for active bouts",
            )
