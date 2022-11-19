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
The python script `map_behaviors.py` provides some critical and important functionality for behavior mapping, together with many redundant and impractical ones.
We will use `map_behaviors.py` to compute behavioral embedding spaces with different approaches (e.g., joint and disparate behavioral embeddings, using supervised or unsupervised dimensionality reduction).
The other not-so-useful functionalities this script provides are related to clustering and analysis of clusters.
As **basty** evolved to be a supervised or semi-supervised pipeline, we did not evaluate the performances of clustering-related components.

Our behavioral nearest neighbor prediction scheme requires the computation of behavioral embedding spaces called semi-supervised pair embeddings.
In this approach, we run a semi-supervised extension of UMAP with each annotated and unannotated behavioral experiment pair.
As a result for $R^{+}$ annotated and $R^{-}$ unannotated experiments, $R^{+} \times R^{-}$ many low dimensional behavioral space is generated.
```bash
python map_behaviors.py --main-cfg-path /path/to/main_cfg.yaml --compute-semisupervised-pair-embeddings
```
For more detailed and accurate explanations of the dimensionality reduction parameters please see [UMAP documentation](https://umap-learn.readthedocs.io/en/latest/api.html).
The below defaults should work well in most cases.
We recommend sticking to Hellinger distance as behavioral representations are $\textnormal{L}_1$ normalized vectors, treating them as probability distributions make more sense.
```python
UMAP_kwargs = {}
UMAP_kwargs["embedding_n_neighbors"] = 75
UMAP_kwargs["embedding_min_dist"] = 0.0
UMAP_kwargs["embedding_spread"] = 1.0
UMAP_kwargs["embedding_n_components"] = 2
UMAP_kwargs["embedding_metric"] = "hellinger"
UMAP_kwargs["embedding_low_memory"] = False
use_annotations_to_mask = (True, False)
embedding_kwargs = {
    **UMAP_kwargs,
    "use_annotations_to_mask": use_annotations_to_mask,
}
```
The parameter `use_annotations_to_mask` is pair of boolean values, which controls the subset of frames to be used in the computation of behavioral embeddings.
If the first value is `True`, then only the annotated frames of the supervised experiment will be used.
Otherwise, all frames detected to be dormant and active in the previous stage of the pipeline will be used.
The same is true for the unsupervised experiment with respect to the second value of the tuple, but only if `evaluation_mode` is `true` in the main configuration (as a matter of fact, an unsupervised experiment is supposed to lack annotations in the first place).

The below parameters are for density-based clustering.
See [HDBSCAN documentation](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html), if you want to experiment with clustering in behavioral embedding spaces.
```python
HDBSCAN_kwargs = {}
HDBSCAN_kwargs["prediction_data"] = True
HDBSCAN_kwargs["approx_min_span_tree"] = True
HDBSCAN_kwargs["cluster_selection_method"] = "eom"
HDBSCAN_kwargs["cluster_selection_epsilon"] = 0.0
HDBSCAN_kwargs["min_cluster_size"] = 500
HDBSCAN_kwargs["min_cluster_samples"] = 5
clustering_kwargs = {**HDBSCAN_kwargs}
```

## Computing Behavioral Scores & Predicting Behavioral Categories
Lastly, `predict_behavioral_categories.py` script computes behavioral scores and predicts corresponding behavioral categories by performing a nearest neighbor based analysis in the semi-supervised behavior embeddings.
It can be configured by using different command line arguments.
To see available arguments and their possible values, one can run the below command.
```bash
python predict_behavioral_categories.py --help
```
The output is the following.
```
usage: predict_behavioral_categories.py [-h] --main-cfg-path MAIN_CFG_PATH --num-neighbors NUM_NEIGHBORS [--neighbor-weights {distance,sq_distance}]
                                        [--neighbor-weights-norm {count,log_count,sqrt_count,proportion}] [--activation {softmax,standard}]
                                        [--voting-weights {entropy,uncertainity}] [--voting {hard,soft}] [--save-weights | --no-save-weights]
```
Here, as the name suggests, `num-neighbors` is basically the number of nearest neighbors contributing to the behavioral score.
This value is typically in the wide interval of $5$ to $100$.
However, as the value of `num-neighbors` increases, imbalanced classes (if exists) become a  more serious problem.
As a matter of fact, behavioral categories usually have a very imbalanced number of occurrence distributions.
To deal with this, one might want to use `--neighbor-weights-norm` option.
For instance, `--neighbor-weights-norm log_count` will normalize each score with the $\log$ number of occurrences of the corresponding behavioral category.
As a default, this script does not apply any such normalization.
The argument `--neighbor-weights` can be used to weight points by the inverse of their distance, i.e., closer neighbors will have a greater influence than neighbors which are further away.
Before combining behavioral weights contributed by each annotated experiment, one also may want to map those weights to interval $(0,1)$ to avoid complications related to differences in the scales.
This can be achieved by `--activation softmax` or `--activation standard`.
The other two arguments, namely `--voting-weights` and `--voting`, configure how the behavioral weights contributed by different annotated experiments are combined to compute a final behavioral score.
If `--voting soft` is passed, then the behavioral weights are directly included in the summation. Else, if `--voting hard` is passed, then behavioral weights are binarized before the summation as maximum category being $1$ and others being $0$.
It is also possible to adjust the contribution of each annotated experiment with respect to the level of uncertainty by using `--voting-weights`.
In other words, if the behavioral weight mainly favors one category and is certain about it, then their contribution will have a greater influence on the final score.

A sensible example list of arguments for `predict_behavior_categories.py` is given below.
```bash
python predict_behavior_categories.py --main-cfg-path /path/to/main_cfg.yaml \
  --num-neighbors 15 --neighbor-weights distance --neighbor-weights-norm log_count \
  --activation standard --voting soft
```

Finally, a report of predictions can simply be generated with the `export_behavior_predictions.py` script as below.
```bash
python export_behavior_predictions.py --main-cfg-path /path/to/main_cfg.yaml
```
