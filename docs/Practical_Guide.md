# Practical Guide (with examples)
In this part of the documentation, usage of the package and wrapper scripts will be described in detail.
As *basty* is a pipeline that consists of relatively well-separated stages, it would be easier to describe each stage separately in correspondence with its script.
The main scripts to create a project and perform behavior mapping are located in [`scripts/project`](../scripts/project/).
The scripts should be run after carefully setting the configuration files and paths.
Default parameters are configured in the [`basty/project/helper`](../basty/project/helper.py).
If you want to set a different set of parameters, you need to edit scripts.

## Initialization
Before everything, a project must be initialized.
Each project has its own directory where its files are stored.
Since experiment recordings are usually quite long, it can easily get unfeasible to keep all the data in the random access memory.
Thus, results are stored in `.pkl` (`pands.DataFrame`s) or `.npy` (`numpy` array) and `ExptRecord` instances (stored as `.z` files with `joblib`) are updated after each stage.

Initialization of the projects is done with the `init_project.py` script, as the name suggests.
All it does is initialize a `Project` class instance from `basty.project.experiment_processing` with suitable parameters and a path of `main_cfg.yaml`.
Keywords arguments `annotation_priority`, `inactive_annotation`, `noise_annotation` and `arouse_annotation` are just to specify how to deal with annotation files.
For instance, `inactive_annotation` determines the string correspondence of label `0`.
```bash
python init_project.py --main-cfg-path /path/to/main_cfg.yaml
```

## Reading the Data and Feature Extraction
The second script that one should proceed with is `extract_features.py`. This script wraps around four steps:
* Computing pose values (includes reading the experiment data and pre-processing) and storing `pose.pkl` ~ `--compute-pose-values`,
* Computing spatio-temporal features and storing `snap_stft.pkl` & `delta_stft.pkl` ~ `--compute-spatio-temporal-features`,
* Applying wavelet transformation to `snap_stft.pkl` and storing `wsft.npy` ~ `--compute-postural-dynamics`,
* Performing flattening, frame normalization, and storing  `behavioral_reprs.npy` ~  `--compute-behavioral-representations`.

You can simply pass the `--compute-all` argument to run all four steps together in order.
The below two commands are equivalent.
```bash
python extract_features.py --main-cfg-path /path/to/main_cfg.yaml --compute-all
python extract_features.py --main-cfg-path /path/to/main_cfg.yaml \
  --compute-pose-values --compute-spatio-temporal-features --compute-postural-dynamics --compute-behavioral-representations
```
Since the preprocessing consists of many steps such as interpolation, smoothing, adaptive filtering based on confidence scores, etc., there are many parameters.
Unless you're not dealing with something extreme, the below default values should work fine.
Kalman filter is implemented, but not recommended as it has lots of parameters that affect the resulting signal a lot.
You can turn off a filter by setting its threshold or window size to 0.
```python
    pose_prep_kwargs = {
        "compute_oriented_pose": True,
        "compute_egocentric_frames": False,
        "local_outlier_threshold": 9,
        "local_outlier_winsize": 15,
        "decreasing_llh_winsize": 30,
        "decreasing_llh_lower_z": 0,
        "low_llh_threshold": 0.0,
        "median_filter_winsize": 6,
        "boxcar_filter_winsize": 6,
        "jump_quantile": 0,
        "interpolation_method": "linear",
        "interpolation_kwargs": {},
        "kalman_filter_kwargs": {},
    }
```
It is possible compute snap spatio-temporal features without computing delta spatio-temporal features (or vice versa) by changing the values of the below arguments.
```python
stft_kwargs = {
		"delta": compute_delta,
		"snap": compute_snap,
}
```
The `use_cartesian_blent` argument of `FeatureExtraction.compute_behavioral_representations()` allow to average over `x` and `y` components of pose spatio-temporal features after wavelet transformation.
If you do not make frames egocentric (which amplifies erroneous tracking), this is helpful for making feature representations orientation-independent.
Another argument of `FeatureExtraction.compute_behavioral_representations()` is `norm` and it determines to type of frame normalization (such as `L1`, `L2`, `max`).
`L1` normalization is recommended.

## Detecting Activities

The next stage of the pipeline is detecting activity bouts and sleep epochs (i.e., long bouts), this stage is called "experiment outlining" in the pipeline.
There exist two steps in this stage, namely "detecting dormancy epochs" (`--outline-dormant-epochs`) and "detecting activity bouts" (`--outline-active-bouts`).
```bash
python outline_experiments.py --main-cfg-path /path/to/main_cfg.yaml --outline-all
python outline_experiments.py --main-cfg-path /path/to/main_cfg.yaml \
  --outline-dormant-epochs --outline-active-bouts
```
Both of these steps can be performed in an unsupervised or supervised fashion.
You can determine which approach to follow by using the boolean parameter `use_supervised_learning`.
It is recommended to perform the unsupervised approach for outlining dormancy epochs and supervised learning for detecting activity bouts.
The `label_conversion_dict` parameter is used to convert annotations to more general categories for supervised learning purposes.
For instance, in the below `label_conversion_dict` configuration for `ExptDormantEpochs.outline_dormant_epochs`, label `0` corresponds to the dormancy, and label `1` corresponds to arouse state.
```python
label_conversion_dict = {
  0: [
    "Idle&Other",
    "HaltereSwitch",
    "Feeding",
    "Grooming",
    "ProboscisPump",
  ],
  1: [
    "Moving",
  ],
  2: [
    "Noise",
  ],
}
```
Similarly, for `ExptDormantEpochs.outline_active_bouts`, we can configure it as follows, where `0` is for quiescence and `1` is for micro-activities.
```python
label_conversion_dict = {
  0: [
    "Idle&Other",
  ],
  1: [
    "Feeding",
    "Grooming",
    "ProboscisPump",
    "HaltereSwitch",
    "Moving",
  ],
  2: [
    "Noise",
  ]
}
```

## Mapping Activities Behavioral Embedding Spaces
```bash
python map_behaviors.py --main-cfg-path /path/to/main_cfg.yaml --compute-semisupervised-pair-embeddings
```

## Computing Behavioral Scores & Predicting Behavioral Categories
```bash
python predict_behavior_categories.py --main-cfg-path /path/to/main_cfg.yaml \
  --num-neighbors 15 --neighbor-weights distance --neighbor-weights-norm log_count \
  --activation standard --voting soft
```

```bash
python export_behavior_predictions.py --main-cfg-path /path/to/main_cfg.yaml
```
