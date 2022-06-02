import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import robust_scale
from tqdm import tqdm

import basty.utils.io as io
import basty.utils.misc as misc
from basty.experiment_processing.experiment_info import ExptRecord
from basty.experiment_processing.experiment_outline import ActiveBouts, DormantEpochs
from basty.project.helper import LoadingHelper, Logger, ParameterHandler, SavingHelper
from basty.utils.annotations import HumanAnnotations as HumAnn

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

        # This mode is to evaluate the pipeline and paramaters using annotated data.
        self.evaluation_mode = self.main_cfg.get("evaluation_mode", False)

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
            io.safe_create_dir(expt_path / "correspondences")
            io.safe_create_dir(expt_path / "figures")

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


class ExptOutline:
    def get_training_data(
        self, X_expt_dict, training_expt_names, get_default_training_labels
    ):
        X_train_list = []
        y_train_list = []

        for ann_expt_name in training_expt_names:
            expt_path = self.expt_path_dict[ann_expt_name]
            expt_record = self._load_joblib_object(expt_path, "expt_record.z")
            y_ann = self._load_numpy_array(expt_path, "annotations.npy")

            X_train = X_expt_dict[ann_expt_name]
            y_train = np.zeros(y_ann.shape, dtype=int)

            if self.label_conversion_dict:
                for (
                    new_label,
                    behavior_list,
                ) in self.label_conversion_dict.items():
                    for behavior in behavior_list:
                        old_label = expt_record.behavior_to_label[behavior]
                        y_train[y_ann == old_label] = new_label
            else:
                y_train = get_default_training_labels(y_train, y_ann, expt_record)

            X_train_list.append(X_train[expt_record.mask_dormant])
            y_train_list.append(y_train[expt_record.mask_dormant])
        return X_train_list, y_train_list


class ExptDormantEpochs(Project, ExptOutline):
    def __init__(self, main_cfg_path, use_supervised_learning=False, **kwargs):
        Project.__init__(self, main_cfg_path, **kwargs)
        self.use_supervised_learning = use_supervised_learning
        self.init_dormant_epochs_kwargs(self.fps, use_supervised_learning, **kwargs)

    @misc.timeit
    def outline_dormant_epochs(self):
        X_expt_dict = dict()
        annotated_expt_names = list(self.annotation_path_dict.keys())
        all_expt_names = list(self.expt_path_dict.keys())

        if self.evaluation_mode:
            expt_names = annotated_expt_names
        else:
            expt_names = all_expt_names

        pbar = tqdm(expt_names)
        for expt_name in pbar:
            expt_path = self.expt_path_dict[expt_name]
            pbar.set_description(
                f"Computing feature values of {expt_name} for outlining dormant epochs"
            )

            expt_record = self._load_joblib_object(expt_path, "expt_record.z")
            delta_stft = self._load_pandas_dataframe(expt_path, "delta_stft.pkl")
            ftname_to_deltaft = self._load_yaml_dictionary(
                expt_path, "ftname_to_deltaft.yaml"
            )

            if not self.datums:
                datums = list(ftname_to_deltaft.values())
            else:
                datums = self.datums

            X = DormantEpochs.get_datums_values(
                delta_stft,
                misc.reverse_dict(ftname_to_deltaft),
                datums=datums,
                winsize=self.datums_winsize,
            ).reshape((-1, 1))

            if self.log_scale:
                X = np.log2(X + 1)
            if self.scale:
                X = robust_scale(X)

            X_expt_dict[expt_name] = X

        dormant_epochs = DormantEpochs()

        if self.use_supervised_learning and not self.evaluation_mode:
            self.logger.direct_info(
                "Training the decision tree for dormant bouts with all."
            )
            training_expt_names = annotated_expt_names
            X_train_list, y_train_list = self.get_training_data(
                X_expt_dict,
                training_expt_names,
                DormantEpochs.get_default_training_labels,
            )
            dormant_epochs.construct_dormant_epochs_decision_tree(
                X_train_list, y_train_list, **self.decision_tree_kwargs
            )

        pbar = tqdm(expt_names)
        for expt_name in pbar:
            expt_path = self.expt_path_dict[expt_name]
            pbar.set_description(f"Outlining dormant epochs of {expt_name}")

            X = X_expt_dict[expt_name]
            expt_record = self._load_joblib_object(expt_path, "expt_record.z")

            if self.use_supervised_learning:
                if self.evaluation_mode:
                    self.logger.direct_info(
                        "Training the decision tree for dormant epochs bouts."
                    )
                    training_expt_names = annotated_expt_names
                    training_expt_names.remove(expt_name)
                    X_train_list, y_train_list = self.get_training_data(
                        X_expt_dict,
                        training_expt_names,
                        DormantEpochs.get_default_training_labels,
                    )
                    dormant_epochs.construct_dormant_epochs_decision_tree(
                        X_train_list, y_train_list, **self.decision_tree_kwargs
                    )
                mask_dormant = dormant_epochs.predict_dormant_epochs(
                    X,
                    min_dormant=self.min_dormant,
                    winsize=self.post_processing_winsize,
                    wintype=self.post_processing_wintype,
                )
            else:
                threshold = DormantEpochs.get_threshold(
                    X,
                    self.threshold_key,
                    self.num_gmm_comp,
                    self.threshold_idx,
                )
                mask_dormant, final_labels = DormantEpochs.compute_dormant_epochs(
                    X,
                    threshold,
                    self.min_dormant,
                    self.tol_duration,
                    self.tol_percent,
                    self.epoch_winsize,
                )

            expt_record.mask_dormant = mask_dormant
            assert expt_record.mask_dormant.any()

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
        annotated_expt_names = list(self.annotation_path_dict.keys())
        all_expt_names = list(self.expt_path_dict.keys())

        if self.evaluation_mode:
            expt_names = annotated_expt_names
        else:
            expt_names = all_expt_names

        pbar = tqdm(expt_names)
        for expt_name in pbar:
            expt_path = self.expt_path_dict[expt_name]
            pbar.set_description(
                f"Computing feature values of {expt_name} for outlining active bouts"
            )

            expt_record = self._load_joblib_object(expt_path, "expt_record.z")
            wsft = self._load_numpy_array(expt_path, "wsft.npy")
            ftname_to_snapft = self._load_yaml_dictionary(
                expt_path, "ftname_to_snapft.yaml"
            )

            df_coefs = ActiveBouts.get_df_coefs_summary(
                wsft,
                ftname_to_snapft,
                log=self.log_scale,
                method=self.coefs_summary_method,
            )
            del wsft

            if not any(self.datums_list):
                self.datums_list = [
                    [ftname_to_snapft[datum]] for datum in df_coefs.columns
                ]  # [[]]
            else:
                self.datums_list = self.datums_list

            values = []
            for datums in self.datums_list:
                values.append(
                    ActiveBouts.get_datums_values(
                        df_coefs,
                        misc.reverse_dict(ftname_to_snapft),
                        datums=datums,
                        winsize=self.datums_winsize,
                    )
                )
            del df_coefs

            X = np.stack(values, axis=1)

            if self.scale:
                X = robust_scale(X)

            X_expt_dict[expt_name] = X
            del values

        active_bouts = ActiveBouts()

        if self.use_supervised_learning and not self.evaluation_mode:
            self.logger.direct_info(
                "Training the decision tree for active bouts with all."
            )
            training_expt_names = annotated_expt_names
            X_train_list, y_train_list = self.get_training_data(
                X_expt_dict,
                training_expt_names,
                ActiveBouts.get_default_training_labels,
            )
            active_bouts.construct_active_bouts_decision_tree(
                X_train_list, y_train_list, **self.decision_tree_kwargs
            )

        pbar = tqdm(expt_names)
        for expt_name in pbar:
            expt_path = self.expt_path_dict[expt_name]
            pbar.set_description(f"Outlining active bouts of {expt_name}")

            X = X_expt_dict[expt_name]
            expt_record = self._load_joblib_object(expt_path, "expt_record.z")

            if self.use_supervised_learning:
                if self.evaluation_mode:
                    self.logger.direct_info(
                        "Training the decision tree for active bouts."
                    )
                    training_expt_names = annotated_expt_names
                    training_expt_names.remove(expt_name)
                    X_train_list, y_train_list = self.get_training_data(
                        X_expt_dict,
                        training_expt_names,
                        ActiveBouts.get_default_training_labels,
                    )
                    active_bouts.construct_active_bouts_decision_tree(
                        X_train_list, y_train_list, **self.decision_tree_kwargs
                    )
                mask_active = active_bouts.predict_active_bouts(
                    X,
                    winsize=self.post_processing_winsize,
                    wintype=self.post_processing_wintype,
                )
            else:
                thresholds = []
                for idx, datums in enumerate(self.datums_list):
                    thresholds.append(
                        ActiveBouts.get_threshold(
                            X[expt_record.mask_dormant, idx, np.newaxis],
                            self.threshold_key,
                            self.num_gmm_comp,
                            self.threshold_idx,
                        )
                    )
                mask_active = ActiveBouts.compute_active_bouts(
                    X,
                    thresholds,
                    winsize=self.post_processing_winsize,
                    wintype=self.post_processing_wintype,
                )

            expt_record.mask_active = mask_active
            if expt_record.has_annotation:
                expt_record.mask_active = np.logical_and(
                    expt_record.mask_active, np.logical_not(expt_record.mask_noise)
                )
            assert expt_record.mask_active.any()
            assert np.logical_and(
                expt_record.mask_active, expt_record.mask_dormant
            ).any()

            self._save_joblib_object(
                expt_record,
                expt_path,
                "expt_record.z",
                "with a mask for active bouts",
            )
