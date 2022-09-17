# basty
## 🪰 Overview
*basty* (Automated **B**ehavioral **A**nalysis of A**s**leep Frui**t** Fl**y**) is a software designed to analyze behavioral correlates of sleep in *Drosophila Melanogaster*.
The software consists of an **end-to-end & multi-stage pipeline** and a couple of useful scripts for behavioral analysis.
*basty* is specifically designed for unique **challenges related to the characteristics of sleep**, so can deal with hard-to-detect behaviors exhibited rarely during long sleep cycles.
It allows both unsupervised and semi-supervised analysis and is able to map the animal tracking data to given behavioral categories with **limited supervision** (*basty* does not require a large labeled data set!).
*basty* also offers an extensive set of pre-processing procedures, and therefore, can directly operate on noisy pose estimation data (e.g., DeepLabCut[^1]).
Similar to many other behavior mapping pipelines, *basty* starts by computing a meaningful high-dimensional spatio-temporal representation from the filtered and imputed pose estimation data.
Utilizing high-dimensional time-series representations, our software can detect **sleep epochs**, **micro-activities** (e.g., postural adjustment and short-duration grooming behaviors), and **macro-activities** (e.g., feeding and walking).
Then, *basty* performs **high-performance behavior mapping** and allows a fine-grained categorization of the micro-activities by generating behavior embeddings and using a novel nearest neighbor-based prediction scheme.

## ⚡️ Quickstart
1. Clone the repository, and change current directory to `basty/`.
  ``` bash
  git clone https://github.com/bo1929/basty.git
  cd basty
  ```
2. Create a new virtual environment with Python version 3.9 with using any version management tool, such as [`conda`](https://www.anaconda.com/products/distribution) and [`pyenv`](https://github.com/pyenv/pyenv).
    * You can use following `conda` commands.
    ``` bash
    conda create -n basty python=3.9.0
    conda activate basty
    ```
    * Alternatively, `pyenv` sets a local application-specific Python version by writing the version name to a file called `.python-version`, and automatically switches to that version when you are in the directory.
    ``` bash
    pyenv local 3.9.0
    ```
3. Use [poetry](https://python-poetry.org/docs/) to install dependencies in the `basty/` directory, and spawn a new shell.
  ```bash
  poetry install
  poetry shell
  ```
4. Then, after creating your configuration files, initialize a project using the below command.
  For a detailed description of the main configuration file (`main_cfg.yaml`), see [Configuration Files](docs/ConfigurationFiles.md).
  ```bash
  python init_project.py --main-cfg-path /path/to/main_cfg.yaml
  ```
5. Extract spatio-temporally meaningful features.
  ```bash
  python extract_features.py --main-cfg-path /path/to/main_cfg.yaml --compute-all
  ```
6. Detect sleep epochs, micro-activity and macro-activity bouts by running the following command.
  ```bash
  python outline_experiments.py --main-cfg-path /path/to/main_cfg.yaml --outline-all
  ```
7. Generate behavioral embedding space(s).
  At this step, you have different options (e.g., unsupervised or semi-supervised dimensionality reduction).
  For a detailed description and some tips, please see [Pipeline Description](docs/PipelineDescription.md) and [Usage Example](docs/UsageExample.md).
  Assuming labeled data is available for semi-supervised behavior mapping, the following command generates semi-supervised pair embeddings (see [Nomenclature](docs/Nomenclature.md) for the definitions).
  ```bash
  python map_behaviors.py --main-cfg-path /path/to/main_cfg.yaml --compute-semisupervised-pair-embeddings
  ```
8. Finally, you can map your unlabeled tracking data to defined behavior categories (with some sensible default parameters), and export predictions to a `.csv` file.
  ```bash
  python predict_behavior_categories.py --main-cfg-path /path/to/main_cfg.yaml \
    --num-neighbors 15 --neighbor-weights distance --neighbor-weights-norm log_count \
    --activation standard --voting soft
  python export_behavior_predictions.py --main-cfg-path /path/to/main_cfg.yaml
  ```

## Guide & Documentation
* [Description of the Pipeline](docs/Description_of_the_Pipeline.md)
* [Folder Structure](docs/Folder_Structure.md)
* [Configuration Files](docs/Configuration_Files.md)
* [Usage Example](docs/Usage_Example.md)
* [Nomenclature](docs/Nomenclature.md)

## References
* [1] : Mathis, Alexander, et al. "DeepLabCut: markerless pose estimation of user-defined body parts with deep learning." Nature neuroscience 21.9 (2018): 1281-1289.
