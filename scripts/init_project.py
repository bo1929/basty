import argparse

from pathlib import Path

from basty.utils.io import read_config
from basty.project.experiment_processing import Project

parser = argparse.ArgumentParser(
    description="Initialize project  based on a given main configuration."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
parser.add_argument(
    "--reset-project",
    action=argparse.BooleanOptionalAction,
    help="Option to create a new project.",
)

args = parser.parse_args()


def backup_old_project(main_cfg_path):
    main_cfg = read_config(main_cfg_path)
    project_path = main_cfg.get("project_path")
    old_proj_dir = Path(project_path)
    if old_proj_dir.exists():
        suffix = 1
        while old_proj_dir.with_suffix(f".{str(suffix)}").exists():
            suffix += 1
        old_proj_dir.replace(old_proj_dir.with_suffix(f".{str(suffix)}"))


if __name__ == "__main__":
    """
    Add suffix or increment NUM to the project created by previous tests.
    """
    if args.reset_project:
        backup_old_project(args.main_cfg_path)

    proj = Project(
        args.main_cfg_path,
        annotation_priority=[],
        inactive_annotation="Idle",
        noise_annotation="Noise",
        arouse_annotation="Moving",
    )
