import logging
import sys

import joblib
import numpy as np
import pandas as pd
import tqdm

import basty.utils.io as io


class ProgressBarHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class ParameterHandler:
    def __init__(self):
        pass

    def init_annotation_kwargs(self, **kwargs):
        self.annotation_priority = kwargs.pop("annotation_priority", [])
        self.inactive_annotation = kwargs.pop("inactive_annotation", "Idle&Other")
        self.noise_annotation = kwargs.pop("noise_annotation", "Noise")
        self.arouse_annotation = kwargs.pop("arouse_annotation", "Moving")

    def init_preprocessing_kwargs(self, **kwargs):
        self.compute_oriented_pose = kwargs.pop("compute_oriented_pose", True)
        self.compute_egocentric_frames = kwargs.pop("compute_egocentric_frames", False)
        self.save_likelihood = kwargs.pop("save_likelihood", False)

        self.local_outlier_threshold = kwargs.pop("local_outlier_threshold", 15)
        assert self.local_outlier_threshold > 0
        self.local_outlier_winsize = kwargs.pop("local_outlier_winsize", 15)

        self.decreasing_llh_winsize = kwargs.pop("decreasing_llh_winsize", 0)
        self.decreasing_llh_lower_z = kwargs.pop("decreasing_llh_lower_z", -3)

        self.low_llh_threshold = kwargs.pop("low_llh_threshold", 0.05)
        assert self.low_llh_threshold < 1

        self.median_filter_winsize = kwargs.pop("median_filter_winsize", 3)
        self.boxcar_filter_winsize = kwargs.pop("boxcar_filter_winsize", 0)

        self.jump_quantile = kwargs.pop("jump_quantile", 0)

        self.interpolation_method = kwargs.pop("interpolation_method", "linear")
        self.interpolation_kwargs = kwargs.pop("interpolation_kwargs", {})
        self.kalman_filter_kwargs = kwargs.pop("kalman_filter_kwargs", {})

    def init_behavioral_reprs_kwargs(self, **kwargs):
        self.use_cartesian_blent = kwargs.pop("use_cartesian_blent", False)
        self.norm = kwargs.pop("norm", "l1")
        assert self.norm in ["l1", "l2", "max"]

    @staticmethod
    def gradient_boosting_default_kwargs(**kwargs):
        gradient_boosting_kwargs = {}
        gradient_boosting_kwargs["n_estimators"] = kwargs.pop("n_estimators", 10)
        gradient_boosting_kwargs["max_depth"] = kwargs.pop("max_depth", 5)
        gradient_boosting_kwargs["min_samples_leaf"] = kwargs.pop(
            "min_samples_leaf", 10**3
        )
        gradient_boosting_kwargs["max_features"] = kwargs.pop("max_features", "sqrt")
        gradient_boosting_kwargs["criterion"] = kwargs.pop("criterion", "friedman_mse")
        return gradient_boosting_kwargs

    @staticmethod
    def decision_tree_default_kwargs(**kwargs):
        decision_tree_kwargs = {}
        decision_tree_kwargs["n_estimators"] = kwargs.pop("n_estimators", 5)
        decision_tree_kwargs["max_depth"] = kwargs.pop("max_depth", 5)
        decision_tree_kwargs["min_samples_leaf"] = kwargs.pop(
            "min_samples_leaf", 10**2
        )
        decision_tree_kwargs["max_features"] = kwargs.pop("max_features", "sqrt")
        decision_tree_kwargs["criterion"] = kwargs.pop("criterion", "gini")
        decision_tree_kwargs["class_weight"] = kwargs.pop(
            "class_weight", "balanced_subsample"
        )
        return decision_tree_kwargs

    @staticmethod
    def nearest_neighbors_default_kwargs(**kwargs):
        nearest_neighbors_kwargs = {}
        nearest_neighbors_kwargs["n_neighbors"] = kwargs.pop("n_neighbors", 50)
        nearest_neighbors_kwargs["weights"] = kwargs.pop("weights", "distance")
        nearest_neighbors_kwargs["algorithm"] = kwargs.pop("algorithm", "auto")
        nearest_neighbors_kwargs["n_jobs"] = kwargs.pop("n_jobs", -1)
        return nearest_neighbors_kwargs

    def init_dormant_epochs_kwargs(self, fps, use_supervised_learning, **kwargs):
        self.datums = kwargs.pop("datums", [])
        self.datums_winsize = kwargs.pop("datums_winsize", fps * 1)
        self.scale = kwargs.pop("scale", False)
        self.log_scale = kwargs.pop("log_scale", False)

        self.min_dormant = kwargs.pop("min_dormant", 300 * fps)

        if not use_supervised_learning:
            self.num_gmm_comp = kwargs.pop("num_gmm_comp", 2)
            self.threshold_key = kwargs.pop("threshold_key", "local_max")
            # Indices start from 0.
            self.threshold_idx = kwargs.pop("threshold_idx", 1)

            self.epoch_winsize = kwargs.pop("epoch_winsize", 180 * fps)
            self.tol_duration = kwargs.pop("tol_duration", 90 * fps)
            self.tol_percent = kwargs.pop("tol_percent", 0.4)
        else:
            self.post_processing_winsize = kwargs.pop(
                "post_processing_winsize", fps * 2
            )
            self.post_processing_wintype = kwargs.pop(
                "post_processing_wintype", "boxcar"
            )
            self.classifier_kwargs = self.__class__.decision_tree_default_kwargs(
                **kwargs
            )
            # self.classifier_kwargs = self.__class__.nearest_neighbors_default_kwargs(
            #     **kwargs
            # )
            self.label_conversion_dict = kwargs.pop("label_conversion_dict", {})

    def init_active_bouts_kwargs(self, fps, use_supervised_learning, **kwargs):
        self.datums_list = kwargs.pop("datums_list", [[]])
        self.datums_winsize = kwargs.pop("datums_winsize", fps // 10)
        self.coefs_summary_method = kwargs.pop("coefs_summary_method", "max")
        self.scale = kwargs.pop("scale", True)
        self.log_scale = kwargs.pop("log_scale", True)

        self.post_processing_winsize = kwargs.pop("post_processing_winsize", fps * 1)
        self.post_processing_wintype = kwargs.pop("post_processing_wintype", "boxcar")

        if not use_supervised_learning:
            self.num_gmm_comp = kwargs.pop("num_gmm_comp", 3)
            self.threshold_key = kwargs.pop("threshold_key", "local_max")
            # Indices start from 0.
            self.threshold_idx = kwargs.pop("threshold_idx", 1)
        else:
            self.classifier_kwargs = self.__class__.decision_tree_default_kwargs(
                **kwargs
            )
            # self.classifier_kwargs = self.__class__.nearest_neighbors_default_kwargs(
            #     **kwargs
            # )
            self.label_conversion_dict = kwargs.pop("label_conversion_dict", {})

    def init_behavior_embeddings_kwargs(self, **kwargs):
        self.UMAP_kwargs = {}
        self.UMAP_kwargs["n_neighbors"] = kwargs.pop("embedding_n_neighbors", 90)
        self.UMAP_kwargs["min_dist"] = kwargs.pop("embedding_min_dist", 0.0)
        self.UMAP_kwargs["spread"] = kwargs.pop("embedding_spread", 1.0)
        self.UMAP_kwargs["n_components"] = kwargs.pop("embedding_n_components", 2)
        self.UMAP_kwargs["metric"] = kwargs.pop("embedding_metric", "hellinger")
        self.UMAP_kwargs["low_memory"] = kwargs.pop("embedding_low_memory", True)
        self.use_annotations_to_mask = kwargs.pop(
            "use_annotations_to_mask", (False, False)
        )
        assert (
            isinstance(self.use_annotations_to_mask, tuple)
            and len(self.use_annotations_to_mask) == 2
        )

    def init_behavior_clustering_kwargs(self, **kwargs):
        self.HDBSCAN_kwargs = {}
        self.HDBSCAN_kwargs["prediction_data"] = kwargs.pop("prediction_data", True)
        self.HDBSCAN_kwargs["approx_min_span_tree"] = kwargs.pop(
            "approx_min_span_tree", True
        )
        self.HDBSCAN_kwargs["cluster_selection_method"] = kwargs.pop(
            "cluster_selection_method", "eom"
        )
        self.HDBSCAN_kwargs["cluster_selection_epsilon"] = kwargs.pop(
            "cluster_selection_epsilon", 0.0
        )
        self.HDBSCAN_kwargs["min_cluster_size"] = kwargs.pop("min_cluster_size", 500)
        self.HDBSCAN_kwargs["min_samples"] = kwargs.pop("min_cluster_samples", 5)

    def init_behavior_mapping_postprocessing_kwargs(self, **kwargs):
        pass

    def init_behavior_correspondence_kwargs(self, **kwargs):
        pass


class Logger:
    def __init__(self, stdout=False, filepath=None, logformat=None):
        self.stdout = stdout
        self.filepath = filepath
        self.logformat = logformat
        self.handlers = []
        self.handlers.append(ProgressBarHandler())

        if self.logformat is None:
            self.logformat = "[%(asctime)s] %(levelname)s - %(message)s"
            self.datefmt = "%Y-%m-%d %H:%M:%S"
        if self.stdout:
            stdout_handler = logging.StreamHandler(sys.stdout)
            self.handlers.append(stdout_handler)
        if self.filepath is not None:
            io.ensure_file_dir(self.filepath)
            file_handler = logging.FileHandler(filename=filepath)
            self.handlers.append(file_handler)

        logging.basicConfig(
            format=self.logformat,
            handlers=self.handlers,
            level=logging.INFO,
            datefmt=self.datefmt,
        )
        self.logger = logging.getLogger("main")

    def direct_error(self, err):
        self.logger.error(err)

    def direct_warning(self, warn):
        self.logger.warning(warn)

    def direct_info(self, info):
        self.logger.info(info)

    @staticmethod
    def _get_path_str(path, depth):
        return "/".join(path.parts[-depth:-1])

    # Log Messages:
    def file_already_exists(self, name, path, depth=2):
        output_path = self._get_path_str(path, depth)
        self.logger.warning(f"FAE - {name} already exists in {output_path}, skipped.")

    def file_saved(self, name, path, depth=2):
        output_path = self._get_path_str(path, depth)
        self.logger.info(f"FS - {name} has been saved to {output_path}.")

    def file_missing(self, name, path, depth=2):
        output_path = self._get_path_str(path, depth)
        self.logger.error(f"FM - {name} can not be found in {output_path}.")

    def not_initialized(self, name):
        self.logger.error(f"NI - {name} is not initialized yet.")

    def invalid_value_warning(self, name, count):
        self.logger.warning(f"IVW - {name} contains {str(count)} inf/nan values.")

    def type_error(self, name, type_name):
        self.logger.error(f"TE - type of '{name}' must be {type_name}.")

    def missing_cfg_key_error(self, name):
        self.logger.error(
            f"MCKE - key {name} can not be found in the configuration file."
        )

    def missing_cfg_key_warning(self, name):
        self.logger.warning(
            f"MCKW - key {name} can not be found in the configuration file, "
            f"default value will be used."
        )


class LoadingHelper:
    def __init__(self, logger):
        self.logger = logger

    def _load_numpy_array(self, path, name):
        npy_path = path / name
        try:
            npy = np.load(npy_path)
        except FileNotFoundError as err:
            self.logger.direct_error(err)
            self.logger.file_missing(f"array {name}", npy_path)
            raise
        return npy

    def _load_joblib_object(self, path, name):
        obj_path = path / name
        try:
            obj = joblib.load(obj_path)
        except FileNotFoundError as err:
            self.logger.direct_error(err)
            self.logger.file_missing(f"object {name}", obj_path)
            raise
        return obj

    def _load_pandas_dataframe(self, path, name):
        df_path = path / name
        try:
            df = pd.read_pickle(df_path)
        except FileNotFoundError as err:
            self.logger.direct_error(err)
            self.logger.file_missing(f"dataframe {name}", df_path)
            raise
        return df

    def _load_yaml_dictionary(self, expt_path, name):
        dict_path = expt_path / name
        try:
            dict_ = io.read_yaml(dict_path)
        except FileNotFoundError as err:
            self.logger.direct_error(err)
            self.logger.file_missing(f"dictionary {name}", dict_path)
            raise
        return dict_


class SavingHelper:
    def __init__(self, logger):
        self.logger = logger

    def _save_numpy_array(self, npy, path, name, msg="", depth=2):
        npy_path = path / name
        np.save(npy_path, npy)
        log_msg = f"{name} {msg}" if len(msg) > 0 else name
        self.logger.file_saved(f"array {log_msg}", npy_path, depth=depth)

    def _save_joblib_object(self, obj, path, name, msg="", depth=2):
        obj_path = path / name
        joblib.dump(obj, obj_path)
        log_msg = f"{name} {msg}" if len(msg) > 0 else name
        self.logger.file_saved(f"object {log_msg}", obj_path, depth=depth)

    def _save_pandas_dataframe(self, df, path, name, msg="", depth=2):
        df_path = path / name
        df.to_pickle(df_path)
        log_msg = f"{name} {msg}" if len(msg) > 0 else name
        self.logger.file_saved(f"dataframe {log_msg}", df_path, depth=depth)

    def _save_yaml_dictionary(self, dict_obj, expt_path, name, msg="", depth=2):
        dict_path = expt_path / name
        io.dump_yaml(dict_obj, dict_path)
        log_msg = f"{name} {msg}" if len(msg) > 0 else name
        self.logger.file_saved(f"yaml file {log_msg}", dict_path, depth=depth)
