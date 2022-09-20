import argparse

from utils import backup_old_project, log_params

from basty.project.experiment_processing import Project

parser = argparse.ArgumentParser(
    description="Initialize project based on a given main configuration."
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


if __name__ == "__main__":
    """
    Add suffix or increment NUM to the project created by previous tests.
    """
    if args.reset_project:
        backup_old_project(args.main_cfg_path)
    init_kwargs = dict(
        annotation_priority=[],
        inactive_annotation="Idle&Other",
        noise_annotation="Noise",
        arouse_annotation="Moving",
    )
    log_params(args.main_cfg_path, "init_params", init_kwargs)
    proj = Project(args.main_cfg_path, **init_kwargs)
