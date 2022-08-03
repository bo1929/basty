# basty

*basty* (Automated **B**ehavioral **A**nalysis of A**s**leep Frui**t** Fl**y**) is a software designed to analyze behavioral correlates of sleep in *Drosophila*. It is **WIP**!

## ⚡️ Quickstart

Create a new virtual environment with [conda](https://www.anaconda.com/products/distribution).
``` bash
conda create -n basty python=3.9.0
```
Then activate created environment.
``` bash
conda activate basty
```

Install [poetry](https://python-poetry.org/docs/) and using poetry install dependencies after changing the current directory into the cloned folder.
```bash
poetry install
```
Then, initialize a project using the below command.
```bash
python init_project.py --main-cfg-path ~/basty/examples/configurations/main_cfg.yaml
```
