# Configuration Files

## Main Configuration ([main_cfg.yaml](configurations/main_cfg.yaml))
`main_cfg.yaml` sets the main project directory path, experiment data paths, and annotation paths (if available), see below.
For annotated experiments, the key (name of the experiment e.g., `FlyF-Test` of pose estimation data path and key of the annotation path must match.
```yaml
# Please always use absolute paths in this configuration file.
project_path: "./project-test"

experiment_data_paths:
  # name: path to the directory
  #   Input file format is name-index.csv, other files will be ignored.
  # An index is a number, consecutive files will be concatenated.  If
  #   indeces are not consecutive will, raise an error. Indices can start
  #   from any nonnegative integer, time-stamps will be handled
  #   accordingly.
  FlyF-Test1: "./datasets/"
  FlyF-Test2: "./datasets/"
  FlyF-Test3: "./datasets/"
  FlyF-Test4: "./datasets/"

# Optional, if there exists an annotation file for the experiment.
annotation_paths:
  # name: path to file
  FlyF-Test1: "./annotations/FlyF-Test1.csv"
  FlyF-Test2: "./annotations/FlyF-Test2.csv"
```

It is also necessary to specify the paths of the other three configuration files as follows.
```yaml
configuration_paths:
  feature_cfg: "./configurations/feature_cfg.yaml"
  temporal_cfg: "./configurations/temporal_cfg.yaml"
  pose_cfg: "./configurations/pose_cfg.yaml"
```

## Pose Configuration ([pose_cfg.yaml](configurations/pose_cfg.yaml))
Pose configuration's main task declaration of body parts that will be used for spatio-temporal feature configuration, and defines the relationships between body parts for preprocessing and determining the animal's orientation.
If a body part does not have a role in the computation of a spatio-temporal feature, then it should not be included in the `pose_cfg.yaml` (to keep the size of the `pose.pkl` smaller).

`singles` key is for body parts that do not have a right or left counterpart.
For example, there is only a single proboscis, there is no right proboscis or left proboscis.
```yaml
# Required.
singles:
  - prob
  - headtip
  - thor_ant
  - thor_post
```

Body parts that have a left-right counterpart, such as haltere (left haltere, `halt_l`, and right haltere `halt_r`) should be configured and the key `counterparts`.
A name that will be used for the counterpart needs to be provided as a key.
While computing a spatio-temporal feature, this new name will be used (also in the `feature_cfg.yaml`).
```yaml
# Drop one of the left-right counterparts based on confidence scores.
# If you want to keep both, list them under the 'single' key.
# Optional.
counterparts:  # new_name: [name_left, name_right]
  head: [head_l, head_r]
  halt: [halt_l, halt_r]
  t1_tip: [t1_l_tip, t1_r_tip]
  t2_tip: [t2_l_tip, t2_r_tip]
  t3_tip: [t3_l_tip, t3_r_tip]
```

There might somehow analogous and similar body parts, for example legs.
In this case, it may be desired to compute spatio-temporal feature values capturing all of them.
For example, we might be interested in velocity of leg tips, then average velocity for all three right legs can be computed as a single value.
For such cases, we can configure `groups` as below.
```yaml
# Group definitions for analogous body parts.
# Optional.
groups:
  joint:
  - joint1
  - joint2
  - joint3
  tip:
  - t1_tip
  - t2_tip
  - t3_tip
```

Another useful configuration key is `connected_parts`. As the name suggests, it can be used to define body parts that are tied to each other.
For example, the thorax posterior and the thorax anterior can not move independently.
Similarly, the independent motion of body parts belonging to the leg is limited to some extent.
These definitions are only used in preprocessing, and are optional (and do not provide much benefit).
```yaml
# Optional.
connected_parts:
  - [joint1_ltop, joint1_lmid, joint1_l, t1_l_tip]
  - [joint2_ltop, joint2_lmid, joint2_l, t2_l_tip]
  - [joint3_ltop, joint_3lmid, joint3_l, t3_l_tip]
```

You can also define new points using the `defined_points` key.
For instance, `midpoint` is the centroid of all the body parts listed under.
```yaml
### User-defined Points ###
# --->
# For better pose representations.
# Optional.
defined_points: # p = (x1 + x2 + ... xn)/n + (y1 + y2 + ... yn)/n
  midpoint:
  - joint2_top
  - joint3_top
  - thor_ant
  - thor_post
```

If you want to make frames egocentric, you need to define a line through two body parts to be the reference.
```yaml
# Construct the spine through two body parts.
# Give only two body parts, and a line representing the spine will pass through them.
# Optional for making frames egocentric.
centerline: [thor_ant, thor_post]
```

## Feature Configuration ([feature_cfg.yaml](configurations/feature_cfg.yaml))
There exists two types of features: delta features (`pose_delta`, `angle_delta`, `distance_delta`) and snap features (`pose`, `angle`, `distance`).
For `pose` (and `delta_pose`) features, it is sufficient to use one body part.
The Below configuration defines three different pose features.
Delta features and snap features do not need to be the same.
```yaml
# Use cartesian components (x,y values) of given body parts, one body part.
pose:
    - prob
    - halt
    - thor_post

# Use given body parts to compute velocity values, one body part.
pose_delta:
    - head
    - prob
    - thor_post
```

For `distance` (and `distance_delta`) features, two body part needs to be given as below.
```yaml
# Use given body parts to compute distance values, two body parts.
distance:
    - [origin, halt]
    - [head, prob]
    - avg:
      - [thor_post, joint1]
      - [thor_post, joint2]
      - [thor_post, joint3]
```

Optionally, you can compute averages, minimums maximums of a set of features as follows.
```yaml
# Only for angle, distance, delta distance, and delta angles.
# - avg: [feature 1, feature 2, ..., feature n]
# - min: [feature 1, feature 2, ..., feature n]
# - max: [feature 1, feature 2, ..., feature n]
```

Similarly, `angle` (and `angle_delta`) features are defined as below.
```yaml
# Use given body parts to compute angle values, three body-parts.
angle:
  - [headtip, thor_post, atip]
  - avg:
    - [t1_tip, joint1, joint1_top]
    - [t2_tip, joint2, joint2_top]
    - [t3_tip, joint3, joint3_top]
# Use given body parts to compute angular velocity values, three body-parts.
# Let's say we don't define any delta features of angles.
angle_delta: []
```

In order to use delta features, you need to define time scales (in milliseconds) for them to be computed over.
Multiple time scales can be given as well.
In that case, delta features will be computed for each.
```yaml
# Given delta scales, the delta feature set will be smoothed on each scale.
# Scales values should be millisecond values.
# Number of frames will be calculated based on the FPS value given in the temporal configuration file.
delta_scales: [33, 66]
```

## Temporal Configuration ([temporal_cfg.yaml](configurations/temporal_cfg.yaml))
This configuration file is mainly for time related configurations such as wavelet transformation and sliding window analysis.

FPS value of the video recordings should be defined as below.
```yaml
# For time-stamps, figures, and time scale related computations.
fps: 30
```

Continuous wavelet transformation needs to be carefully configured, `wavelet` and `padding` are directly passed to the `pywt` package.
Values of `num_channels`, `freq_up`, `freq_low`, and `freq_spacing` are used to determine the spectrum of wavelet scales (see Chapter 3, Section 4.2 of the thesis for more details).
`normalization_option` is for choosing the approach for making different scales comparable, to get an insight see Chapter 3, Section 4.2.1 of the thesis, and [`feature_extraction/wavelet_transformation.py`](../basty/feature_extraction/wavelet_transformation.py).
```yaml
wt:
    wavelet: morl
    padding: reflect
    num_channels: 20
    freq_up: 10
    freq_low: 0.1
    freq_spacing: dyadically
    normalization_option: 2

# Optinal, if above 'wt' config should be ignored for that a specific snap feature.
# 'wt' config can be overwritten for specific features by defining a separate one under 'wt_detached'.
# wt_detached:
#   2: # This is the snap name key, which can be found in snap_names_dict.yaml
#     wavelet: morl
#     padding: reflect
#     num_channels: 20
#     freq_up: 10
#     freq_low: 0.1
#     freq_spacing: dyadically
```

The below parameters are to configure methods of the `BehavioralWindows` class, which is used to analyze mapping or annotations with a sliding window approach.
```yaml
windows:
  winsize: 600
  stepsize: 60
  # method: 'count' or 'tf-idf'
  method: tf-idf
  # For scipy.signal.get_window (must require no parameters).
  # If the method is tf-idf then this must be boxcar.
  # Default is the boxcar.
  wintype: boxcar
  # norm: (Optional) 'l1' or 'l2' or 'max' if method is count.
  # If the method is count or binary then the default is None.
  # norm: (Optional) 'l1' or 'l2' if method is tf-idf.
  # If the method is tf-idf then the default is 'l1'
  norm: null
```
