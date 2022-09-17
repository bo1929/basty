# basty

*basty* (Automated **B**ehavioral **A**nalysis of A**s**leep Frui**t** Fl**y**) is a software designed to analyze behavioral correlates of sleep in *Drosophila*. It is **WIP**!

## ⚡️ Quickstart
1. Clone the repository, and change current directory to `basty`.
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
3. Use [poetry](https://python-poetry.org/docs/) to install dependencies after changing the current directory into the cloned folder.
```bash
poetry install
```
4. Then, initialize a project using the below command.
```bash
python init_project.py --main-cfg-path ~/basty/docs/configurations/main_cfg.yaml
```
