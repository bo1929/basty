# Folder Structure
* 📁 [`basty/`](#basty)
  * 📁 [`behavior_mapping/`](#behavior_mapping)
      * 📄 [`behavioral_windows.py`](#behavioral_windowspy)
      * 📄 [`behavioral_states.py`](#behavioral_statespy)
  * 📁[`experiment_processing/`](#experiment_processing)
      * 📄[`experiment_info.py`](#experiment_infopy)
      * 📄[`experiment_outlining.py`](#experiment_outliningpy)
  * 📁[`feature_extraction/`](#feature_extraction)
      * 📄[`body_pose.py`](#body_posepy)
      * 📄[`spatiotemporal_features.py`](#spatiotemporal_featurespy)
      * 📄[`wavelet_transformation.py`](#wavelet_transformationpy)
  * 📁[`project/`](#project)
      * 📄[`behavior_mapping.py`](#behavior_mappingpy)
      * 📄[`experiment_processing.py`](#experiment_processingpy)
      * 📄[`feature_extraction.py`](#feature_extractionpy)
      * 📄[`helper.py`](#helperpy)
* 📁[`scripts/`](#scripts)
  * 📁[`evaluation/`](#evaluation)
      * 📄[`evaluate_outlining_performance.py`](#evaluate_outlining_performancepy)
      * 📄[`evaluate_prediction_performance.py`](#evaluate_prediction_performancepy)
  * 📁[`misc/`](#misc)
      * 📄[`export_behavior_predictions.py`](#export_behavior_predictionspy)
      * 📄[`export_stft_to_csv.py`](#export_stft_to_csvpy)
  * 📁[`visualization/`](#visualization)
      * 📄[`generate_embedding_spaces.py`](#generate_embedding_spacespy)
      * 📄[`generate_ethograms.py`](#generate_ethogramspy)
      * 📄[`style.py`](#stylepy)
  * 📁[`project/`](#project)
    * 📄[`extract_features.py`](#extract_featurespy)
    * 📄[`init_project.py`](#init_projectpy)
    * 📄[`map_behaviors.py`](#map_behaviorspy)
    * 📄[`outline_experiments.py`](#outline_experimentspy)
    * 📄[`predict_behavior_categories.py`](#predict_behavior_categoriespy)
    * 📄[`process_annotations.py`](#process_annotationspy)
    * 📄[`utils.py`](#utilspy)

## 📁`basty/`

### 📁`behavior_mapping/`

#### 📄`behavioral_windows.py`
This file contains the class `BehavioralWindows`, which provides methods for sliding window based analysis of the behavior predictions/annotations and is not related to the behavior mapping.
By setting a window size and a step size, one can analyze different temporal characteristics of the behavioral repertoire (such as co-occurrences of behaviors, and temporal organization through the sleep cycle).

#### 📄`behavioral_states.py`
This is WIP.

### 📁`experiment_processing/`

#### 📄`experiment_info.py`
The `ExptRecord` class is in this file, which inherits the `ExptInfo` and `AnnotationInfo` classes.
The job of the `ExptRecord` class is basically recording some important information about each experiment such as;
 * FPS,
 * number of frames (i.e., data points),
 * mapping from behaviors to annotation labels,
 * mapping from feature names to feature matrix indices,
 * a boolean variable indicating if an experiment is annotated, etc...

For each experiment (i.e., a full, uninterrupted video recording of an animal at a specific date), an instance of `ExptRecord` is initiated at the beginning and is saved as a `joblib` object.
As the pipeline proceeds, arrays indicating micro-activity epochs and sleep epochs are also added as attributes to `ExptRecord` instances, and corresponding saved objects are updated.

#### 📄`experiment_outlining.py`
This file contains classes that are responsible for sleep/activity detection.
There exist two main classes in this file, namely `ActiveBouts` and `DormantEpochs`.
The first one is for detecting micro-activity bouts regardless of the sleep/wake state of the animal.
The latter one handles computing time intervals in which the animal is **mostly** dormant (i.e., likely to be asleep).
Both classes offer supervised and unsupervised alternatives, and usage of these classes can be seen in the wrapper module [`project/experiment_info.py`](#experiment_infopy) from the `project` sub-package.

### 📁`feature_extraction/`

#### 📄`body_pose.py`
`BodyPose` class' task is to format the pose estimation data read from the `.csv` files.
It contains methods to make data egocentric and to compute the orientation of the animal using confidence scores.
After computing orientations and modifying the data based on the pose configuration (which configures which body parts will be used and the relations between body parts, e.g., being right-left counterparts), `BodyPose.get_oriented_pose` returns a reduced version of pose estimation data as a `pandas.DataFrame`.

#### 📄`spatiotemporal_features.py`
This file contains the `SpatioTemporal` class, whose task is to compute basic spatio-temporal features from the reduced & processed pose values.
 `SpatioTemporal.get_snap_stft` and `SpatioTemporal.get_delta_stft` are two essential methods of this class.
  As described in [Description of the Pipeline](Description_of_the_Pipeline.md), there exist two types of spatio-temporal features: ones that are computed based on the instantaneous values (snapshot features) and ones that are computed based on the changing values (delta features).
  For instance, snapshot features include distances, angles, and relative positions; delta features include velocities, angular velocities, and changes in distances.

#### 📄`wavelet_transformation.py`
`WaveletTransformation`, as its name suggests, performs the wavelet transformation, and is simply a wrapper around the `pywt` module.
In addition to being a wrapper, it also provides some functionality for determining wavelet scale spectrum (`linearly`, `dyadically`) and normalization of the power spectrum across the different frequency channels.

### 📁`project/`

#### 📄`behavior_mapping.py`
`behavior_mapping.py` might be the most encrypted file in the project, as it includes many methods which have relatively unimportant use cases and do not provide good insight into the behavioral data (they are included just for the sake of regularity and completeness).
But, this file also contains one of the most essential (and novel) parts of the pipeline.
For example, the `BehaviorEmbedding.compute_semisupervised_pair_embeddings` method generates semi-supervised pair behavioral embeddings, which are used to perform behavior mapping with nearest neighbor analysis [^1].
The `BehaviorEmbedding` class provides behavioral embedding generation methods of all kinds, i.e., unsupervised, semi-supervised, and supervised.
Methods whose name have the `disparate` embed experiments separately and generates, a behavioral embedding space for each experiment.
On the other hand, ones with `joint` in their name generate a joint embedding to which all available experiments are projected.
As the number of distinct experiments in an embedding space increases, it usually becomes harder to interpret.
The `BehaviorClustering` class is especially useful for performing unsupervised analysis.

[^1]: If annotated data is available and a path is given in the configuration file, a behavioral embedding is generated for each pair of annotated and unannotated experiments.

#### 📄`experiment_processing.py`
`experiment_processing.py` has two main tasks: project management and activity detection (i.e., experiment outlining), which are provided by the `Project` class and `ExptActiveBouts` & `ExptDormantEpochs` classes, respectively.
`Project` instance initialization creates a project directory, initiates `ExptRecord` objects and saves them, and reads annotations (if available).
`ExptActiveBouts` and `ExptDormantEpochs` are basically wrappers around the methods provided by `ActiveBouts` and `DormantEpochs` classes (as mentioned in [`experiment_outlining.py`](#experiment_outliningpy)), and perform activity detection, and updates saved `ExptRecord` objects for each experiment by adding activity indicator arrays as attributes.

#### 📄`feature_extraction.py`
`FeatureExtraction.compute_pose_values` read `.csv` files, transforms raw pose estimation data to pose value format (using methods from `BodyPose` class, see [`feature_extraction/body_pose.py`](#body_posepy))), and then perform preprocessing (filtering and imputation).
The resulting `pandas.DataFrame` is saved to the experiment's directory in the project folder as `pose.pkl`.
Then, based on the feature configuration (`feature_cfg.yaml`), spatio-temporal features are computed (`pandas.DataFrame`s of delta features and snap features are respectively saved as `delta_stft.pkl` and `snap_stft.pkl`) by `FeatureExtraction.compute_spatiotemporal_features` (utilizing methods from  [`feature_extraction/spatiotempral_features.py`](#spatiotempral_featurespy)).
After that, `FeatureExtraction.compute_postural_dynamics` performs wavelet continuous transformation, and returns a tensor of features, (# of time steps, # of frequency channels, # of snapshot features).
The returned tensor is saved as a `numpy` array named `wsft.npy`.
At the end `feature_extraction.compute_behavioral_representations` flattens `wsft.npy` to make it a matrix (of size # of time steps, # of frequency channels times # of snapshot features), and apply $\textnormal{L}_1$ normalization to each frame.
The resulting matrix is a high-dimensional behavioral representation, saved as a `numpy` array named `behavioral_reprs.npy`.

#### 📄`helper.py`
Default parameters of the pipeline are hard-coded here.
It would be helpful to check these out to determine the best values for your project.
Other than that, there exist some helpful utilities for saving/loading files and logging & log messages.

## 📁`scripts/`

### 📁`evaluation/`

#### 📄`evaluate_outlining_performance.py`
If `evaluation_mode` is `true` in `main_cfg.yaml` and there exists an annotated experiment, this script report performance of activity detection (i.e., experiment outlining).

#### 📄`evaluate_prediction_performance.py`
If `evaluation_mode` is `true` in `main_cfg.yaml` and there exist at least two annotated experiments, this script provides cross-validation (leave-one-out) and reports the performance of behavior mapping.

### 📁`misc/`

#### 📄`export_behavior_predictions.py`
This script exports predicted behavioral categories as a `.csv` file for each experiment (columns are behavior name, beginning of the behavior bout, and end of the behavior bout).

#### 📄`export_stft_to_csv.py`
This script exports spatio-temporal feature `pandas.DataFrame`s (`snap_stft.pkl` and `delta_stft.pkl`) as `.csv` files for each experiments.

#### 📄`process_annotations.py`
`process_annotations.py` merges annotations of different behavioral categories which are provided in separate `.csv` files.
This script is useful when only the behaviors of interest are annotated, and the rest of the time points are left unannotated.
What it does is basically fill unannotated time intervals with the value of the `LABEL_UNANNOTATED` variable.
So, this is just an ad-hoc solution for converting the annotations we have to a more structured format.

### 📁`visualization/`

#### 📄`generate_embedding_spaces.py`
This script is for generating scatter plots of behavioral embeddings, you can choose the type of embedding (supervised/unsupervised/semi-supervised, joint/disparate/pair) to visualize.
There are also many visualization options that can be specified with command line arguments.

#### 📄`generate_ethograms.py`
This script generates ethograms from given categorical labels such as behavior annotations and behavior predictions.

#### 📄`style.py`
This file contains the style configuration for the Altair package.

### 📁`project/`

#### 📄`init_project.py`
This script initializes a project based on a given main configuration path. For example, it creates necessary directories and `ExptRecord` instances and reads annotations and configurations.

#### 📄`extract_features.py`
`extract_features.py` script handles preprocessing, constructing pose values (`--compute-pose-values`), computing spatio-temporal features (`--compute-postural-dynamics`), wavelet transformation (`--compute-postural-dynamics`), and frame normalization (`--compute-behavioral-representations`).

#### 📄`outline_experiments.py`
This script is responsible for detecting activities and quiescence (i.e., experiment outlining).
You can use the `--outline-dormant-epochs` argument to predict long dormancy and moving/walking epochs.
Similarly, the `--outline-active-bouts` argument can be given to detect micro-activity bouts.

#### 📄`map_behaviors.py`
`map_behaviors.py` can be used for generating embeddings, clustering, and computing correspondences/compositions of clusters. Many options are available (supervised/unsupervised/semi-supervised, joint/disparate/pair).
Before nearest neighbor analysis and behavioral category prediction, this script must be run with the `--compute-semisupervised-pair-embeddings` command line argument.

#### 📄`predict_behavior_categories.py`
This script is for predicting behaviors by performing a nearest-neighbor analysis on semi-supervised pair behavioral embeddings.
There are many options and parameters which might affect the performance a lot, for details and mathematical description of the algorithm please see the publication or the thesis given in the main README file.
One can use the `--save-weights` argument to save computed NN-weights in addition to predicted categorical labels.

#### 📄`utils.py`
This file contains `log_params` and `backup_old_project` functions.
`log_params` saves used parameters to the project directory when the script is run.
